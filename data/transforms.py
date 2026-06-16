"""
data/transforms.py
------------------
TriModalTransform — joint augmentation pipeline that preserves spatial alignment.
"""
from __future__ import annotations

__all__ = ["TriModalTransform", "build_transform"]

import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class TriModalTransform:
    """
    Joint augmentation pipeline for RGB, Thermal, LiDAR numpy arrays.

    The same random parameters are applied to all three modalities to
    preserve spatial alignment. Colour jitter is applied to RGB only.
    Gaussian noise is applied to Thermal and LiDAR only.

    All inputs are expected as np.ndarray uint8 (H, W, C) or (H, W).
    Outputs are float32 torch.Tensors: (C, H, W).
    """

    def __init__(self, cfg: dict, is_train: bool = True) -> None:
        self.is_train = is_train
        img_cfg = cfg.get("image", {})
        aug_cfg = cfg.get("augmentation", {}) if is_train else {}
        cor_cfg = cfg.get("corruption", {}) if is_train else {}

        self.target_h: int = img_cfg.get("height", 256)
        self.target_w: int = img_cfg.get("width", 256)

        # Normalisation
        self.rgb_mean   = img_cfg.get("rgb_mean",     [0.485, 0.456, 0.406])
        self.rgb_std    = img_cfg.get("rgb_std",      [0.229, 0.224, 0.225])
        self.th_mean    = img_cfg.get("thermal_mean", [0.5])
        self.th_std     = img_cfg.get("thermal_std",  [0.5])
        self.li_mean    = img_cfg.get("lidar_mean",   [0.5])
        self.li_std     = img_cfg.get("lidar_std",    [0.5])

        # Augmentation flags (train only)
        if is_train and aug_cfg.get("enabled", False):
            self.hflip_p    = aug_cfg.get("random_hflip", 0.0)
            self.vflip_p    = aug_cfg.get("random_vflip", 0.0)
            self.rot_cfg    = aug_cfg.get("random_rotate", {"enabled": False})
            self.cj_cfg     = aug_cfg.get("color_jitter",  {"enabled": False})
            self.noise_cfg  = aug_cfg.get("gaussian_noise", {"enabled": False})
        else:
            self.hflip_p    = 0.0
            self.vflip_p    = 0.0
            self.rot_cfg    = {"enabled": False}
            self.cj_cfg     = {"enabled": False}
            self.noise_cfg  = {"enabled": False}

        # Corruption simulation (train only)
        if is_train and cor_cfg.get("enabled", False):
            self.corrupt_p    = cor_cfg.get("probability", 0.0)
            self.corrupt_types = cor_cfg.get("types", [])
            self.dropout_p    = cor_cfg.get("dropout_prob", 0.1)
        else:
            self.corrupt_p    = 0.0
            self.corrupt_types = []
            self.dropout_p    = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def __call__(
        self,
        rgb: np.ndarray,
        thermal: np.ndarray,
        lidar: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply joint transforms.

        Args:
            rgb:     (H, W, 3) uint8 numpy array
            thermal: (H, W)    or (H, W, 1) uint8 numpy array
            lidar:   (H, W)    or (H, W, 1) uint8 numpy array

        Returns:
            (rgb_t, thermal_t, lidar_t) float32 tensors (C, H, W)
        """
        rgb_img = Image.fromarray(rgb.astype(np.uint8))
        th_img  = Image.fromarray(
            thermal.squeeze().astype(np.uint8) if thermal.ndim == 3 else thermal.astype(np.uint8),
            mode="L"
        )
        li_img  = Image.fromarray(
            lidar.squeeze().astype(np.uint8)   if lidar.ndim == 3   else lidar.astype(np.uint8),
            mode="L"
        )

        # 1) Resize
        rgb_img = TF.resize(rgb_img, [self.target_h, self.target_w], antialias=True)
        th_img  = TF.resize(th_img,  [self.target_h, self.target_w], antialias=True)
        li_img  = TF.resize(li_img,  [self.target_h, self.target_w], antialias=True)

        if self.is_train:
            rgb_img, th_img, li_img = self._apply_geometry(rgb_img, th_img, li_img)

        # 2) To tensor (HWC → CHW, [0,1] float)
        rgb_t = TF.to_tensor(rgb_img)   # (3, H, W)
        th_t  = TF.to_tensor(th_img)    # (1, H, W)
        li_t  = TF.to_tensor(li_img)    # (1, H, W)

        # 3) Colour jitter on RGB only (train)
        if self.is_train and self.cj_cfg.get("enabled", False):
            rgb_t = self._color_jitter(rgb_t)

        # 4) Gaussian noise on Thermal + LiDAR only (train)
        if self.is_train and self.noise_cfg.get("enabled", False):
            std = self.noise_cfg.get("std", 0.01)
            th_t = (th_t + torch.randn_like(th_t) * std).clamp(0.0, 1.0)
            li_t = (li_t + torch.randn_like(li_t) * std).clamp(0.0, 1.0)

        # 5) Corruption simulation (train)
        if self.is_train and self.corrupt_p > 0:
            rgb_t, th_t, li_t = self._simulate_corruption(rgb_t, th_t, li_t)

        # 6) Normalise
        rgb_t = TF.normalize(rgb_t, self.rgb_mean, self.rgb_std)
        th_t  = TF.normalize(th_t,  self.th_mean,  self.th_std)
        li_t  = TF.normalize(li_t,  self.li_mean,  self.li_std)

        return rgb_t, th_t, li_t

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _apply_geometry(self, rgb, th, li):
        """Apply identical geometric transforms to all three modalities."""
        # Horizontal flip
        if random.random() < self.hflip_p:
            rgb = TF.hflip(rgb)
            th  = TF.hflip(th)
            li  = TF.hflip(li)

        # Vertical flip
        if random.random() < self.vflip_p:
            rgb = TF.vflip(rgb)
            th  = TF.vflip(th)
            li  = TF.vflip(li)

        # Random rotation
        if self.rot_cfg.get("enabled", False):
            angle = random.uniform(
                -self.rot_cfg.get("degrees", 10),
                self.rot_cfg.get("degrees", 10),
            )
            rgb = TF.rotate(rgb, angle)
            th  = TF.rotate(th,  angle)
            li  = TF.rotate(li,  angle)

        return rgb, th, li

    def _color_jitter(self, rgb_t: torch.Tensor) -> torch.Tensor:
        brightness = self.cj_cfg.get("brightness", 0.0)
        contrast   = self.cj_cfg.get("contrast",   0.0)
        saturation = self.cj_cfg.get("saturation", 0.0)
        hue        = self.cj_cfg.get("hue",        0.0)

        # Apply jitter ops in random order
        fns = []
        if brightness > 0:
            bfactor = random.uniform(max(0.0, 1 - brightness), 1 + brightness)
            fns.append(lambda t: TF.adjust_brightness(t, bfactor))
        if contrast > 0:
            cfactor = random.uniform(max(0.0, 1 - contrast), 1 + contrast)
            fns.append(lambda t: TF.adjust_contrast(t, cfactor))
        if saturation > 0:
            sfactor = random.uniform(max(0.0, 1 - saturation), 1 + saturation)
            fns.append(lambda t: TF.adjust_saturation(t, sfactor))
        if hue > 0:
            hfactor = random.uniform(-hue, hue)
            fns.append(lambda t: TF.adjust_hue(t, hfactor))

        random.shuffle(fns)
        for fn in fns:
            rgb_t = fn(rgb_t)
        return rgb_t.clamp(0.0, 1.0)

    def _simulate_corruption(
        self,
        rgb: torch.Tensor,
        th: torch.Tensor,
        li: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly apply a corruption to one modality (dropout / noise / blur)."""
        if random.random() > self.corrupt_p:
            return rgb, th, li

        target = random.choice(["rgb", "thermal", "lidar"])
        mode   = random.choice(self.corrupt_types) if self.corrupt_types else "noise"

        def _corrupt(t: torch.Tensor) -> torch.Tensor:
            if mode == "dropout":
                mask = (torch.rand_like(t) > self.dropout_p).float()
                return t * mask
            elif mode == "noise":
                return (t + torch.randn_like(t) * 0.05).clamp(0.0, 1.0)
            elif mode == "blur":
                # Simple box blur via unfold — no external dependency
                pad = torch.nn.functional.pad(t.unsqueeze(0), [1, 1, 1, 1], mode="reflect")
                unf = pad.unfold(2, 3, 1).unfold(3, 3, 1)
                return unf.contiguous().view(*t.shape[:1], t.shape[1], t.shape[2], -1).mean(-1)
            return t

        if target == "rgb":
            rgb = _corrupt(rgb)
        elif target == "thermal":
            th = _corrupt(th)
        else:
            li = _corrupt(li)

        return rgb, th, li


def build_transform(data_cfg: dict, split: str) -> TriModalTransform:
    """Build a TriModalTransform for the given data split."""
    is_train = split == "train"
    return TriModalTransform(data_cfg, is_train=is_train)

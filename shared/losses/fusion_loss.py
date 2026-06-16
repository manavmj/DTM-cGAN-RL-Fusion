"""
shared/losses/fusion_loss.py
------------------------------
L_fusion — SSIM + perceptual + L1 pixel fusion quality loss.
"""
from __future__ import annotations

__all__ = ["FusionLoss"]

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from shared.metrics.fusion_metrics import ssim


class FusionLoss(nn.Module):
    """
    Fusion quality loss combining:
        1. (1 - SSIM)    structural similarity loss         [always on]
        2. Perceptual    VGG16-feature-space L1 distance    [optional]
        3. L1 pixel      mean absolute error                [optional]

    Args:
        use_perceptual: Include VGG perceptual loss term.
        use_l1:         Include L1 pixel loss term.
        weights:        dict with keys ssim, perceptual, l1 (default 1.0 each)
    """

    def __init__(
        self,
        use_perceptual: bool = True,
        use_l1: bool = True,
        weights: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.use_perceptual = use_perceptual
        self.use_l1 = use_l1
        self.weights = weights or {"ssim": 1.0, "perceptual": 1.0, "l1": 0.5}

        if use_perceptual:
            vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
            # Extract features up to relu3_3 (indices 0..16)
            self.vgg_features = nn.Sequential(*list(vgg.features.children())[:17])
            for p in self.vgg_features.parameters():
                p.requires_grad_(False)
        else:
            self.vgg_features = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) generated fused image.
            target: (B, 3, H, W) reference (RGB) image.
                    Both expected in [-1, 1] or [0, 1] — normalised appropriately.

        Returns:
            Scalar loss tensor (lower = higher fusion quality).
        """
        # Ensure both in [0, 1] for metrics
        pred_01   = (pred   + 1.0) / 2.0 if pred.min() < -0.01   else pred
        target_01 = (target + 1.0) / 2.0 if target.min() < -0.01 else target

        total = torch.tensor(0.0, device=pred.device)

        # 1) SSIM loss: (1 - mean_SSIM)
        ssim_val  = ssim(pred_01, target_01).mean()   # scalar
        ssim_loss = 1.0 - ssim_val
        total = total + self.weights.get("ssim", 1.0) * ssim_loss

        # 2) Perceptual loss (VGG features)
        if self.use_perceptual and self.vgg_features is not None:
            # VGG expects [0,1] — already satisfied by pred_01/target_01
            # Ensure 3-channel (expand grayscale if needed)
            p3 = pred_01   if pred_01.shape[1] == 3   else pred_01.expand(-1, 3, -1, -1)
            t3 = target_01 if target_01.shape[1] == 3 else target_01.expand(-1, 3, -1, -1)

            with torch.amp.autocast("cuda", enabled=False):
                feat_p = self.vgg_features(p3.float())
                feat_t = self.vgg_features(t3.float())

            perc_loss = F.l1_loss(feat_p, feat_t)
            total = total + self.weights.get("perceptual", 1.0) * perc_loss

        # 3) L1 pixel loss
        if self.use_l1:
            l1_loss = F.l1_loss(pred_01, target_01)
            total = total + self.weights.get("l1", 0.5) * l1_loss

        return total

"""
shared/encoder/multimodal_encoder.py
--------------------------------------
Multimodal Encoder — feature extractors for RGB, Thermal, and LiDAR.
"""
from __future__ import annotations

__all__ = ["MultimodalEncoder"]

import torch
import torch.nn as nn

from shared.encoder.confidence_estimator import ConfidenceEstimator


class SimpleConvBlock(nn.Module):
    """Basic Conv-BN-ReLU block."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultimodalEncoder(nn.Module):
    """
    Extracts features and confidence scores for RGB, Thermal, and LiDAR.
    
    Returns:
        dict with features {rgb, thermal, lidar} and confidence {rgb, thermal, lidar}
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        
        out_dim = cfg.get("embed_dim", 64)
        
        # RGB Encoder (3 channels)
        self.rgb_encoder = nn.Sequential(
            SimpleConvBlock(3, 32, stride=2),
            SimpleConvBlock(32, out_dim, stride=2)
        )
        self.rgb_conf = ConfidenceEstimator(out_dim)
        
        # Thermal Encoder (1 channel)
        self.th_encoder = nn.Sequential(
            SimpleConvBlock(1, 32, stride=2),
            SimpleConvBlock(32, out_dim, stride=2)
        )
        self.th_conf = ConfidenceEstimator(out_dim)
        
        # LiDAR Encoder (1 channel, typically dense depth map or BEV projected to camera view)
        self.li_encoder = nn.Sequential(
            SimpleConvBlock(1, 32, stride=2),
            SimpleConvBlock(32, out_dim, stride=2)
        )
        self.li_conf = ConfidenceEstimator(out_dim)

    def forward(
        self, 
        rgb: torch.Tensor, 
        thermal: torch.Tensor, 
        lidar: torch.Tensor,
        tau: float = 1.0
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Args:
            rgb:     (B, 3, H, W)
            thermal: (B, 1, H, W)
            lidar:   (B, 1, H, W)
            tau:     Temperature for softmax over confidences.
            
        Returns:
            features: dict of (B, C, H', W')
            confidences: dict of (B, 1) normalized values summing to 1.
        """
        f_rgb = self.rgb_encoder(rgb)
        f_th  = self.th_encoder(thermal)
        f_li  = self.li_encoder(lidar)
        
        c_rgb = self.rgb_conf(f_rgb)
        c_th  = self.th_conf(f_th)
        c_li  = self.li_conf(f_li)
        
        # Softmax normalize across modalities
        c_stack = torch.cat([c_rgb, c_th, c_li], dim=1) # (B, 3)
        c_norm  = torch.softmax(c_stack / tau, dim=1)   # (B, 3)
        
        features = {"rgb": f_rgb, "thermal": f_th, "lidar": f_li}
        confidences = {
            "rgb":     c_norm[:, 0:1], 
            "thermal": c_norm[:, 1:2], 
            "lidar":   c_norm[:, 2:3]
        }
        
        return features, confidences

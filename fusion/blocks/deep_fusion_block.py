"""
fusion/blocks/deep_fusion_block.py
------------------------------------
Deep Fusion Block — heavily parameterised cross-modal reasoning.
"""
from __future__ import annotations

__all__ = ["DeepFusionBlock"]

import torch
import torch.nn as nn

from fusion.attention.cross_attention import CrossModalAttention


class DeepFusionBlock(nn.Module):
    """
    Performs deep fusion using cross-modal attention.
    Calculates CMA for RGB<-Thermal, RGB<-LiDAR, etc.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.cma_rgb_th = CrossModalAttention(dim)
        self.cma_rgb_li = CrossModalAttention(dim)
        
        self.merge = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(
        self, 
        f_rgb: torch.Tensor, 
        f_th: torch.Tensor, 
        f_li: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            f_rgb, f_th, f_li: (B, C, H, W)
            
        Returns:
            (B, C, H, W) fused features
        """
        # Cross attention to RGB
        att_th = self.cma_rgb_th(target=f_rgb, source=f_th)
        att_li = self.cma_rgb_li(target=f_rgb, source=f_li)
        
        # Concatenate and merge
        combined = torch.cat([f_rgb, att_th, att_li], dim=1)
        return self.merge(combined)

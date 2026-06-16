"""
fusion/blocks/light_fusion_block.py
-------------------------------------
Light Fusion Block — lightweight gated fusion path.
"""
from __future__ import annotations

__all__ = ["LightFusionBlock"]

import torch
import torch.nn as nn


class LightFusionBlock(nn.Module):
    """
    Performs fast, lightweight fusion using gated summation.
    
    F_light = sigma(W_l[F_rgb, F_th, F_li]) * sum_m (C_m * F_m)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate_conv = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(
        self, 
        f_rgb: torch.Tensor, 
        f_th: torch.Tensor, 
        f_li: torch.Tensor,
        c_rgb: torch.Tensor,
        c_th: torch.Tensor,
        c_li: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            f_*: (B, C, H, W)
            c_*: (B, 1) or (B, 1, 1, 1) confidences
            
        Returns:
            (B, C, H, W) fused features
        """
        # Ensure confidences have spatial dims
        if c_rgb.dim() == 2:
            c_rgb = c_rgb.view(-1, 1, 1, 1)
            c_th  = c_th.view(-1, 1, 1, 1)
            c_li  = c_li.view(-1, 1, 1, 1)
            
        # Gating signal
        concat_f = torch.cat([f_rgb, f_th, f_li], dim=1)
        gate = torch.sigmoid(self.gate_conv(concat_f)) # (B, C, H, W)
        
        # Confidence-weighted sum
        weighted_sum = (f_rgb * c_rgb) + (f_th * c_th) + (f_li * c_li)
        
        return gate * weighted_sum

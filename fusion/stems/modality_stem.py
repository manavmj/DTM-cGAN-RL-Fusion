"""
fusion/stems/modality_stem.py
-------------------------------
Modality Stem — initial shallow processing of modalities before fusion.
"""
from __future__ import annotations

__all__ = ["ModalityStem"]

import torch
import torch.nn as nn


class ModalityStem(nn.Module):
    """
    Applies initial processing to project inputs into a common channel dimension
    before fusion layers.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
            
        Returns:
            (B, out_channels, H, W)
        """
        return self.stem(x)

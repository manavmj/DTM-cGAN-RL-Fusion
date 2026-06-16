"""
shared/encoder/confidence_estimator.py
----------------------------------------
Confidence Estimator — predicts sensor reliability from features.
"""
from __future__ import annotations

__all__ = ["ConfidenceEstimator"]

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceEstimator(nn.Module):
    """
    Estimates a confidence score Cm ∈ [0, 1] for a given modality feature map.
    
    Uses a small CNN + MLP over the dense features to predict reliability,
    which is then softmax-normalized across all modalities in the main model.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) modality feature map
            
        Returns:
            (B, 1) raw logit for confidence (pre-softmax)
        """
        x = self.conv(features)        # (B, hidden_dim, 1, 1)
        x = x.view(x.shape[0], -1)     # (B, hidden_dim)
        logit = self.mlp(x)            # (B, 1)
        return logit

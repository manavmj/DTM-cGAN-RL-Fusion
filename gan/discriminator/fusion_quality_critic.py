"""
gan/discriminator/fusion_quality_critic.py
--------------------------------------------
Fusion Quality Critic — evaluates perceptual and informational realism.
"""
from __future__ import annotations

__all__ = ["FusionQualityCritic"]

import torch
import torch.nn as nn


class FusionQualityCritic(nn.Module):
    """
    Evaluates whether the fused image looks like a realistic, high-quality
    composite. Outputs a score Q_f ∈ [0, 1].
    
    Architecture is a standard PatchGAN-style convolutional network.
    """

    def __init__(self, in_channels: int = 3, ndf: int = 64) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            # (B, 3, H, W) -> (B, ndf, H/2, W/2)
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> (B, ndf*2, H/4, W/4)
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> (B, ndf*4, H/8, W/8)
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> (B, ndf*8, H/8, W/8) - stride=1
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> (B, 1, H/8, W/8) Patch output
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, 3, H, W) fused image or real reference
            
        Returns:
            (B, 1) global quality score in [0, 1] (average of patch scores)
        """
        patch_out = self.net(img)                 # (B, 1, H', W')
        # We output a sigmoid value for the RL reward, but standard adversarial loss 
        # usually prefers raw logits. For RL reward Q_f we use sigmoid.
        # This implementation returns the aggregated sigmoid score.
        score = torch.sigmoid(patch_out).mean(dim=[1, 2, 3]) # (B,)
        return score

"""
gan/discriminator/discriminator.py
------------------------------------
Multi-Critic Evaluation Framework wrapper.
"""
from __future__ import annotations

__all__ = ["MultiCriticDiscriminator"]

import torch
import torch.nn as nn

from gan.discriminator.fusion_quality_critic import FusionQualityCritic
from gan.discriminator.latency_critic import LatencyCritic


class MultiCriticDiscriminator(nn.Module):
    """
    Wraps the standard PatchGAN discriminator (FusionQualityCritic)
    with the Latency critic for a unified interface.
    """

    def __init__(self, cfg: dict, action_dim: int) -> None:
        super().__init__()
        
        ndf = cfg.get("ndf", 64)
        
        self.quality_critic = FusionQualityCritic(in_channels=3, ndf=ndf)
        self.latency_critic = LatencyCritic(action_dim=action_dim)

    def forward(
        self, 
        img: torch.Tensor, 
        action: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Evaluates the fused image and chosen action.
        
        Args:
            img:    (B, 3, H, W) fused image or real reference
            action: (B, action_dim) required for fake images, optional for real
            
        Returns:
            Dict containing:
                - q_f: Fusion Quality score (B,)
                - t_l: Latency penalty proxy (B,) [only if action provided]
                - raw_logits: PatchGAN raw logits for adversarial loss (B, N, H, W)
        """
        # Get raw logits from quality critic's internal layers for adversarial loss
        patch_logits = self.quality_critic.net(img)
        q_f = torch.sigmoid(patch_logits).mean(dim=[1, 2, 3])
        
        out = {
            "q_f": q_f,
            "raw_logits": patch_logits.view(img.shape[0], -1) # Flattened for BCE
        }
        
        if action is not None:
            out["t_l"] = self.latency_critic(action)
            
        return out

"""
shared/losses/generator_loss_engine.py
----------------------------------------
Generator Loss Engine — combines all four loss terms:

    L_G = λ1·L_adv + λ2·L_fusion + λ3·L_latency + λ4·L_task
"""
from __future__ import annotations

__all__ = ["GeneratorLossEngine"]

import torch
import torch.nn as nn

from shared.losses.adversarial_loss import AdversarialLoss
from shared.losses.fusion_loss       import FusionLoss
from shared.losses.latency_loss      import LatencyLoss
import lpips


class GeneratorLossEngine(nn.Module):
    """
    Combines four loss terms into the total generator loss:

        L_G = λ1·L_adv  +  λ2·L_fusion  +  λ3·L_latency

    where λ1..λ3 come from model_config.yaml → loss_weights.

    Args:
        loss_weights_cfg: dict with keys:
            lambda_adv, lambda_fusion, lambda_latency
        adv_loss_type:    "bce" | "hinge" | "wgan" (default "bce")
        use_perceptual:   Include VGG perceptual term in L_fusion.
        use_l1:           Include L1 pixel term in L_fusion.
    """

    def __init__(
        self,
        loss_weights_cfg: dict,
        adv_loss_type: str = "bce",
        use_perceptual: bool = True,
        use_l1: bool = True,
    ) -> None:
        super().__init__()

        self.lambda_adv     = float(loss_weights_cfg.get("lambda_adv",     1.0))
        self.lambda_fusion  = float(loss_weights_cfg.get("lambda_fusion",  10.0))
        self.lambda_latency = float(loss_weights_cfg.get("lambda_latency", 0.5))

        self.adv_loss     = AdversarialLoss(loss_type=adv_loss_type)
        self.fusion_loss  = FusionLoss(use_perceptual=use_perceptual, use_l1=use_l1)
        self.latency_loss = LatencyLoss()
        
        self.lpips_loss = lpips.LPIPS(net='vgg')
        for param in self.lpips_loss.parameters():
            param.requires_grad = False

    def forward(
        self,
        fake_scores: torch.Tensor,    # (B,)   discriminator patch scores for G(z)
        fused: torch.Tensor,          # (B, 3, H, W) generated image
        reference: torch.Tensor,      # (B, 3, H, W) RGB reference
        tl_pred: torch.Tensor,        # (B,)   latency proxy from LatencyCritic
    ) -> dict[str, torch.Tensor]:
        """
        Compute L_G and all sub-terms.

        Returns:
            dict with keys:
                total    — total generator loss (scalar, differentiable)
                ladv     — adversarial loss component
                lfusion  — fusion quality loss component
                llatency — latency loss component
        """
        ladv     = self.adv_loss(fake_scores)
        lfusion  = self.fusion_loss(fused, reference)
        llatency = self.latency_loss(tl_pred)
        
        # Add LPIPS to fusion loss component (lpips expects [-1, 1] inputs)
        lpips_val = self.lpips_loss(fused, reference).mean()
        lfusion = lfusion + lpips_val

        total = (
            self.lambda_adv     * ladv     +
            self.lambda_fusion  * lfusion  +
            self.lambda_latency * llatency
        )

        return {
            "total":    total,
            "ladv":     ladv,
            "lfusion":  lfusion,
            "llatency": llatency,
        }


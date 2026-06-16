"""
shared/losses/adversarial_loss.py
-----------------------------------
L_adv — GAN adversarial loss terms for both generator and discriminator.
"""
from __future__ import annotations

__all__ = ["AdversarialLoss"]

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    """
    GAN adversarial loss supporting BCE, Hinge, and WGAN variants.

    Generator loss (call with fake=True):
        BCE:   -log(D(G(z)))          non-saturating
        Hinge: -mean(D(G(z)))
        WGAN:  -mean(D(G(z)))

    Discriminator loss (call with is_disc_loss=True):
        BCE:   -mean(log D(real) + log(1 - D(fake)))
        Hinge: mean(relu(1 - D(real)) + relu(1 + D(fake)))
        WGAN:  mean(D(fake)) - mean(D(real))
    """

    SUPPORTED = {"bce", "hinge", "wgan"}

    def __init__(self, loss_type: str = "bce") -> None:
        super().__init__()
        loss_type = loss_type.lower()
        if loss_type not in self.SUPPORTED:
            raise ValueError(
                f"loss_type must be one of {self.SUPPORTED}, got '{loss_type}'"
            )
        self.loss_type = loss_type

    # ------------------------------------------------------------------
    # Generator loss (fooling D)
    # ------------------------------------------------------------------
    def generator_loss(self, fake_scores: torch.Tensor) -> torch.Tensor:
        """
        Generator adversarial loss — maximise D(G(z)).

        Args:
            fake_scores: (B,) or (B, 1) raw discriminator scores for fake images.

        Returns:
            Scalar loss (lower = generator winning).
        """
        fake_scores = fake_scores.view(-1)

        if self.loss_type == "bce":
            # Non-saturating: -E[log σ(D(G(z)))]
            return F.binary_cross_entropy_with_logits(
                fake_scores,
                torch.ones_like(fake_scores),
            )
        elif self.loss_type in ("hinge", "wgan"):
            return -fake_scores.mean()

    # ------------------------------------------------------------------
    # Discriminator loss (distinguish real from fake)
    # ------------------------------------------------------------------
    def discriminator_loss(
        self,
        real_scores: torch.Tensor,
        fake_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminator adversarial loss.

        Args:
            real_scores: (B,) raw scores for real images.
            fake_scores: (B,) raw scores for G(z) (detached in the D-update).

        Returns:
            Scalar discriminator loss.
        """
        real_scores = real_scores.view(-1)
        fake_scores = fake_scores.view(-1)

        if self.loss_type == "bce":
            real_loss = F.binary_cross_entropy_with_logits(
                real_scores, torch.ones_like(real_scores)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_scores, torch.zeros_like(fake_scores)
            )
            return (real_loss + fake_loss) * 0.5

        elif self.loss_type == "hinge":
            real_loss = F.relu(1.0 - real_scores).mean()
            fake_loss = F.relu(1.0 + fake_scores).mean()
            return (real_loss + fake_loss) * 0.5

        elif self.loss_type == "wgan":
            return fake_scores.mean() - real_scores.mean()

    # ------------------------------------------------------------------
    # Default forward = generator loss (for backward compat with engine)
    # ------------------------------------------------------------------
    def forward(self, fake_scores: torch.Tensor) -> torch.Tensor:
        return self.generator_loss(fake_scores)

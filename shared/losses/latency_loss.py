"""
shared/losses/latency_loss.py
-------------------------------
L_latency — Differentiable latency proxy loss.
"""
from __future__ import annotations

__all__ = ["LatencyLoss"]

import torch
import torch.nn as nn


class LatencyLoss(nn.Module):
    """
    Wraps the LatencyCritic's predicted latency score Tl ∈ [0, 1] into
    a differentiable scalar loss.

    The loss is simply the mean predicted latency — gradients flow back
    through the LatencyCritic's MLP proxy, enabling the generator to
    learn efficiency-aware fusion.

    L_latency = mean(Tl)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, tl_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tl_pred: (B,) normalised predicted latency from LatencyCritic ∈ [0, 1].

        Returns:
            Scalar latency loss (lower = faster predicted inference).
        """
        return tl_pred.mean()

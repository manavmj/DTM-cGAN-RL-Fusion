"""
shared/reward/reward_engine.py
--------------------------------
PPO Reward Engine — combines critic signals into a scalar RL reward.

R_t = α * Q_{f,t} - γ * T_{l,t} - δ * C_{t}
"""
from __future__ import annotations

__all__ = ["RewardEngine"]

import torch
import torch.nn as nn


class RewardEngine(nn.Module):
    """
    Computes the scalar reward for the PPO agent based on feedback
    from the Multi-Critic Evaluator.

    R_t = α * Q_f - γ * T_l - δ * C_res

    Where:
        Q_f:   Fusion Quality (from FusionQualityCritic)
        T_l:   Latency proxy (from LatencyCritic)
        C_res: Computational cost proxy (derived from action budgets)
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        # Load weights from config
        self.alpha = float(cfg.get("alpha", 1.0))
        self.gamma = float(cfg.get("gamma", 0.5))
        self.delta = float(cfg.get("delta", 0.1))

    def forward(
        self,
        fusion_quality: torch.Tensor,
        latency_proxy: torch.Tensor,
        resource_cost: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total scalar reward for each sample in the batch.

        Args:
            fusion_quality: (B,) predicted quality score in [0, 1]
            latency_proxy:  (B,) predicted latency penalty in [0, 1]
            resource_cost:  (B,) actual FLOP/Memory penalty mapped to [0, 1]

        Returns:
            (B,) reward values.
        """
        reward = (
            self.alpha * fusion_quality -
            self.gamma * latency_proxy -
            self.delta * resource_cost
        )
        return reward

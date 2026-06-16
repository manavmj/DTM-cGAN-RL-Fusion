"""
rl/agent/critic.py
--------------------
PPO Critic Network — evaluates state value.
"""
from __future__ import annotations

__all__ = ["CriticNetwork"]

import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    """
    Predicts the expected return V(S_t) for the PPO agent.
    """

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim)
            
        Returns:
            (B, 1) State value prediction
        """
        return self.net(state)

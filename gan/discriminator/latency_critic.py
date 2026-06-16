"""
gan/discriminator/latency_critic.py
-------------------------------------
Latency Critic — differentiates latency estimation for generator gradients.
"""
from __future__ import annotations

__all__ = ["LatencyCritic"]

import torch
import torch.nn as nn


class LatencyCritic(nn.Module):
    """
    Predicts the computational latency T_l ∈ [0, 1] from the chosen
    RL action vector and current state.
    
    This acts as a differentiable proxy for actual hardware timing,
    allowing gradients to flow back to the generator's topology selections.
    """

    def __init__(self, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # Normalised latency proxy
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action: (B, action_dim) the PPO action defining the topology
            
        Returns:
            (B,) latency penalty proxy in [0, 1]
        """
        return self.net(action).squeeze(-1)

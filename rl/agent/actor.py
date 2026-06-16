"""
rl/agent/actor.py
-------------------
PPO Actor Network — maps state to action distribution.
"""
from __future__ import annotations

__all__ = ["ActorNetwork"]

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    """
    Predicts a continuous action vector defining the generator's topology.
    Outputs mean and log_std for a diagonal Gaussian distribution.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True)
        )
        
        self.mu = nn.Linear(128, action_dim)
        # Log standard deviation parameter (state-independent, standard practice in PPO for continuous domains)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state: torch.Tensor) -> Normal:
        """
        Args:
            state: (B, state_dim)
            
        Returns:
            Normal distribution from which actions can be sampled.
        """
        x = self.net(state)
        mu = self.mu(x)
        
        # Action space is typically bounded [-1, 1], so we use tanh on the mean
        mu = torch.tanh(mu)
        
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

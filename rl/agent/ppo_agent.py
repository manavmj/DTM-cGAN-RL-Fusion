"""
rl/agent/ppo_agent.py
----------------------
PPO Agent — Actor-Critic wrapper.
"""
from __future__ import annotations

__all__ = ["PPOAgent"]

import torch
import torch.nn as nn

from rl.agent.actor import ActorNetwork
from rl.agent.critic import CriticNetwork


class PPOAgent(nn.Module):
    """
    Actor-Critic PPO agent combining policy and value networks.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)

    def act(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Sample action from current policy. Used during rollout collection.
        
        Args:
            state: (B, state_dim)
            
        Returns:
            dict containing action, log_prob, and state value.
        """
        dist = self.actor(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(state).squeeze(-1)
        
        return {
            "action": action,     # (B, A)
            "log_prob": log_prob, # (B,)
            "value": value        # (B,)
        }

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Evaluate stored actions — used during PPO update.
        
        Args:
            state:  (B, state_dim)
            action: (B, action_dim)
            
        Returns:
            dict containing log_prob, entropy, and state value.
        """
        dist = self.actor(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state).squeeze(-1)
        
        return {
            "log_prob": log_prob, # (B,)
            "entropy": entropy,   # (B,)
            "value": value        # (B,)
        }

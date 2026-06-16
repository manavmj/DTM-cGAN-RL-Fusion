"""
rl/memory/rollout_buffer.py
-----------------------------
Rollout Buffer — stores PPO experience trajectories.
"""
from __future__ import annotations

__all__ = ["RolloutBuffer"]

import torch


class RolloutBuffer:
    """
    Stores transitions (state, action, log_prob, reward, value, done)
    for PPO updates.
    """

    def __init__(self) -> None:
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.is_terminals: list[bool] = []

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        is_terminal: bool = False
    ) -> None:
        """Stores a single transition."""
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(reward.detach())
        self.values.append(value.detach())
        self.is_terminals.append(is_terminal)

    def clear(self) -> None:
        """Clears all stored data."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.is_terminals.clear()
        
    def get_data(self) -> dict[str, torch.Tensor]:
        """Returns stacked tensors of all stored experience."""
        # Note: In a batched environment, state is (B, dim), so we cat along dim=0
        return {
            "states": torch.cat(self.states, dim=0),
            "actions": torch.cat(self.actions, dim=0),
            "log_probs": torch.cat(self.log_probs, dim=0),
            "rewards": torch.cat(self.rewards, dim=0),
            "values": torch.cat(self.values, dim=0)
        }

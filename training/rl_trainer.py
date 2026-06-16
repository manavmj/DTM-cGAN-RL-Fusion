"""
training/rl_trainer.py
------------------------
RL Trainer — manages rollout collection and PPO updates.
"""
from __future__ import annotations

__all__ = ["RLTrainer"]

import torch
import torch.nn as nn

from rl.agent.ppo_agent import PPOAgent
from rl.memory.rollout_buffer import RolloutBuffer
from rl.update.ppo_update import PPOUpdater
from shared.reward.reward_engine import RewardEngine


class RLTrainer:
    """
    Handles PPO trajectory collection and policy updates.
    """

    def __init__(
        self,
        agent: PPOAgent,
        updater: PPOUpdater,
        buffer: RolloutBuffer,
        reward_engine: RewardEngine,
        device: torch.device
    ) -> None:
        self.agent = agent
        self.updater = updater
        self.buffer = buffer
        self.reward_engine = reward_engine
        self.device = device

    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        q_f: torch.Tensor,
        t_l: torch.Tensor,
        resource_cost: torch.Tensor
    ) -> float:
        """
        Computes reward and stores the transition in the PPO buffer.
        
        Returns:
            Mean reward for logging.
        """
        with torch.no_grad():
            reward = self.reward_engine(
                fusion_quality=q_f,
                latency_proxy=t_l,
                resource_cost=resource_cost
            )
            
        self.buffer.push(state, action, log_prob, reward, value)
        return reward.mean().item()

    def update_policy(self) -> dict[str, float]:
        """
        Executes the PPO update using collected rollouts.
        Clears the buffer afterwards.
        """
        # Ensure we have data
        if len(self.buffer.states) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            
        rollout_data = self.buffer.get_data()
        metrics = self.updater.update(rollout_data)
        self.buffer.clear()
        
        return metrics

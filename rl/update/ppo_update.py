"""
rl/update/ppo_update.py
-------------------------
PPO Update Routine — GAE and clipped surrogate objective.
"""
from __future__ import annotations

__all__ = ["PPOUpdater"]

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from rl.agent.ppo_agent import PPOAgent


class PPOUpdater:
    """
    Performs PPO updates on the Actor and Critic networks using Generalized 
    Advantage Estimation (GAE).
    """

    def __init__(self, cfg: dict, agent: PPOAgent, optimizer: torch.optim.Optimizer) -> None:
        self.agent = agent
        self.optimizer = optimizer
        
        # Hyperparameters
        self.clip_ratio = cfg.get("clip_ratio", 0.2)
        self.gamma = cfg.get("gamma", 0.99)
        self.lam = cfg.get("gae_lambda", 0.95)
        self.epochs = cfg.get("ppo_epochs", 4)
        self.batch_size = cfg.get("ppo_batch_size", 64)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)
        self.value_coef = cfg.get("value_coef", 0.5)

    def compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        next_value: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes GAE and Returns.
        
        Args:
            rewards: (N,)
            values: (N,)
            
        Returns:
            advantages: (N,)
            returns: (N,)
        """
        # Append next_value for bootstrap
        values = torch.cat([values, torch.tensor([next_value], device=values.device)])
        
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0.0
        
        # Iterate backwards
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.lam * lastgaelam
            
        returns = advantages + values[:-1]
        
        return advantages, returns

    def update(self, rollout_data: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Executes PPO update over rollout data.
        
        Args:
            rollout_data: dict of stacked tensors (states, actions, log_probs, rewards, values)
            
        Returns:
            dict with average losses (policy_loss, value_loss, entropy)
        """
        states = rollout_data["states"]
        actions = rollout_data["actions"]
        old_log_probs = rollout_data["log_probs"]
        rewards = rollout_data["rewards"]
        values = rollout_data["values"]
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        total_p_loss = 0.0
        total_v_loss = 0.0
        total_ent = 0.0
        
        for _ in range(self.epochs):
            for b_states, b_actions, b_old_log_probs, b_returns, b_advantages in loader:
                
                eval_res = self.agent.evaluate(b_states, b_actions)
                new_log_probs = eval_res["log_prob"]
                new_values = eval_res["value"]
                entropy = eval_res["entropy"].mean()
                
                # Policy ratio
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(new_values, b_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()
                
                total_p_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                total_ent += entropy.item()
                
        num_batches = self.epochs * len(loader)
        
        return {
            "policy_loss": total_p_loss / num_batches,
            "value_loss": total_v_loss / num_batches,
            "entropy": total_ent / num_batches
        }

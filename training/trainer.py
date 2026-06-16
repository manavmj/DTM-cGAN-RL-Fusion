"""
training/trainer.py
---------------------
Master Trainer — coordinates the joint optimization of RL and GAN.
"""
from __future__ import annotations

__all__ = ["MasterTrainer"]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gan.generator.generator import DynamicGenerator
from gan.discriminator.discriminator import MultiCriticDiscriminator
from rl.agent.ppo_agent import PPOAgent
from scene.state_builder import StateBuilder

from training.gan_trainer import GANTrainer
from training.rl_trainer import RLTrainer


class MasterTrainer:
    """
    Coordinates the bi-level optimization loop.
    
    For each batch:
      1. Forward RL to get action topology.
      2. Forward GAN (with action) to get fusion.
      3. Evaluate GAN to get critics feedback.
      4. Step GAN optimization.
      5. Step RL rollout collection (using critics feedback as reward).
      6. Periodically step RL optimization (PPO update).
    """

    def __init__(
        self,
        generator: DynamicGenerator,
        discriminator: MultiCriticDiscriminator,
        ppo_agent: PPOAgent,
        state_builder: StateBuilder,
        gan_trainer: GANTrainer,
        rl_trainer: RLTrainer,
        device: torch.device,
        ppo_update_freq: int = 4
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.ppo_agent = ppo_agent
        self.state_builder = state_builder
        
        self.gan_trainer = gan_trainer
        self.rl_trainer = rl_trainer
        
        self.device = device
        self.ppo_update_freq = ppo_update_freq
        self.global_step = 0

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """Runs one full epoch of joint training."""
        self.generator.train()
        self.discriminator.train()
        self.ppo_agent.train()
        self.state_builder.train()
        
        epoch_metrics: dict[str, float] = {}
        batch_count = 0
        
        # State tracking for Meta Fusion Bank (RNN-like dependency across batches isn't strictly
        # valid without sequential data, but we use previous_state from last batch as an approximation
        # for continuous learning context).
        previous_state = None
        
        for batch in dataloader:
            rgb = batch["rgb"].to(self.device)
            th  = batch["thermal"].to(self.device)
            li  = batch["lidar"].to(self.device)
            
            # ===============================================================
            # 1. RL Forward: Build State & Sample Action
            # ===============================================================
            with torch.no_grad():
                # We need features/confidences to build the state
                features, confidences = self.generator.encoder(rgb, th, li)
                
                # Mock resource stats (e.g., current hardware load)
                # In practice, this could be read from utils/latency_metrics.py
                B = rgb.shape[0]
                resource_stats = torch.zeros(B, 3, device=self.device) # [latency, flops, mem]
                
                # Build state
                current_state, _ = self.state_builder(
                    features, confidences, resource_stats, previous_state
                )
                previous_state = current_state.detach()
            
            # Sample action
            rl_out = self.ppo_agent.act(current_state)
            action = rl_out["action"]
            log_prob = rl_out["log_prob"]
            value = rl_out["value"]
            
            # ===============================================================
            # 2. GAN Forward & Update
            # ===============================================================
            gan_metrics = self.gan_trainer.train_step(rgb, th, li, action)
            
            # ===============================================================
            # 3. RL Rollout Collection
            # ===============================================================
            # Calculate mock resource cost (could be derived from action's topological weight)
            # Higher deep fusion weight = higher cost
            # Assuming action[:, 5] is w_deep (from AdaptiveConstructor)
            w_deep = ((action[:, 5] + 1.0) / 2.0).detach()
            resource_cost = w_deep 
            
            q_f = torch.tensor(gan_metrics["q_f_mean"], device=self.device).expand(B)
            t_l = torch.tensor(gan_metrics["t_l_mean"], device=self.device).expand(B)
            
            reward_mean = self.rl_trainer.store_transition(
                state=current_state,
                action=action,
                log_prob=log_prob,
                value=value,
                q_f=q_f,
                t_l=t_l,
                resource_cost=resource_cost
            )
            
            gan_metrics["PPO_reward"] = reward_mean
            
            # ===============================================================
            # 4. Periodically Update PPO
            # ===============================================================
            self.global_step += 1
            if self.global_step % self.ppo_update_freq == 0:
                ppo_metrics = self.rl_trainer.update_policy()
                gan_metrics.update(ppo_metrics)
                
            # Accumulate metrics
            for k, v in gan_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            batch_count += 1
            
        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= batch_count
            
        return epoch_metrics

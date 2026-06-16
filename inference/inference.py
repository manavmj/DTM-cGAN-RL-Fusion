"""
inference/inference.py
------------------------
Inference Pipeline — evaluates test images with frozen RL policy and GAN Generator.
"""
from __future__ import annotations

__all__ = ["InferencePipeline"]

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gan.generator.generator import DynamicGenerator
from rl.agent.ppo_agent import PPOAgent
from scene.state_builder import StateBuilder
from utils.visualization import denormalize, save_image_grid, make_image_grid


class InferencePipeline:
    """
    Runs evaluation on a dataset using the trained DTM-RL-GAN.
    """

    def __init__(
        self,
        generator: DynamicGenerator,
        ppo_agent: PPOAgent,
        state_builder: StateBuilder,
        device: torch.device
    ) -> None:
        self.generator = generator
        self.ppo_agent = ppo_agent
        self.state_builder = state_builder
        self.device = device
        
        # Ensure eval mode
        self.generator.eval()
        self.ppo_agent.eval()
        self.state_builder.eval()

    @torch.no_grad()
    def run(self, dataloader: DataLoader, output_dir: str | None = None) -> dict[str, float]:
        """
        Evaluate the dataset.
        
        Args:
            dataloader: Test/Val DataLoader
            output_dir: If provided, saves fused image grids here.
            
        Returns:
            Metrics dict including average inference latency.
        """
        import os
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        latencies = []
        previous_state = None
        
        for batch_idx, batch in enumerate(dataloader):
            rgb = batch["rgb"].to(self.device)
            th  = batch["thermal"].to(self.device)
            li  = batch["lidar"].to(self.device)
            
            torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
            t0 = time.perf_counter()
            
            # 1. Forward features and confidences
            features, confidences = self.generator.encoder(rgb, th, li)
            
            # 2. Build State
            B = rgb.shape[0]
            resource_stats = torch.zeros(B, 3, device=self.device)
            current_state, _ = self.state_builder(features, confidences, resource_stats, previous_state)
            previous_state = current_state
            
            # 3. PPO Action (Deterministic for inference: use mean of distribution)
            dist = self.ppo_agent.actor(current_state)
            action = dist.loc # use mean for deterministic eval
            
            # 4. Generate fused representation
            fused_image, _, _ = self.generator(rgb, th, li, action)
            
            torch.cuda.synchronize(self.device) if self.device.type == "cuda" else None
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # in ms
            
            # Visualization
            if output_dir and batch_idx < 10: # save up to 10 batches
                # fused_image is in [-1, 1], others might be [0, 1] depending on norm
                # The visualisation utilities handle it.
                grid = make_image_grid(rgb, th, li, fused_image)
                save_path = os.path.join(output_dir, f"batch_{batch_idx:04d}.png")
                save_image_grid(grid, save_path)
                
        avg_latency = sum(latencies) / max(1, len(latencies))
        
        return {
            "avg_latency_ms": avg_latency,
            "fps": 1000.0 / avg_latency if avg_latency > 0 else 0.0
        }

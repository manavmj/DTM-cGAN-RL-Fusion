"""
training/gan_trainer.py
-------------------------
GAN Trainer — coordinates Generator and Discriminator updates.
"""
from __future__ import annotations

__all__ = ["GANTrainer"]

import torch
import torch.nn as nn
from torch.optim import Optimizer

from gan.generator.generator import DynamicGenerator
from gan.discriminator.discriminator import MultiCriticDiscriminator
from shared.losses.generator_loss_engine import GeneratorLossEngine
from shared.losses.adversarial_loss import AdversarialLoss


class GANTrainer:
    """
    Handles the standard GAN min-max optimization.
    """

    def __init__(
        self,
        generator: DynamicGenerator,
        discriminator: MultiCriticDiscriminator,
        opt_g: Optimizer,
        opt_d: Optimizer,
        gen_loss_engine: GeneratorLossEngine,
        adv_loss_module: AdversarialLoss,
        device: torch.device
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.gen_loss_engine = gen_loss_engine
        self.adv_loss_module = adv_loss_module
        self.device = device

    def train_step(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        lidar: torch.Tensor,
        action: torch.Tensor
    ) -> dict[str, float]:
        """
        Executes one GAN training step (Discriminator then Generator).
        
        Args:
            rgb, thermal, lidar: (B, C, H, W) references/inputs
            action: (B, A) PPO action
            
        Returns:
            Dictionary of loss values.
        """
        # ====================================================================
        # 1. Forward Generator (with gradient, since we need to update G)
        # ====================================================================
        fused_fake, _, _ = self.generator(rgb, thermal, lidar, action)
        
        # ====================================================================
        # 2. Train Discriminator
        # ====================================================================
        self.opt_d.zero_grad()
        
        # Real pass (RGB is considered the high-quality target reference for realism)
        real_eval = self.discriminator(rgb)
        
        # Fake pass (detach generator output so gradients don't flow to G yet)
        fake_eval = self.discriminator(fused_fake.detach(), action)
        
        # Discriminator adversarial loss
        d_loss = self.adv_loss_module.discriminator_loss(
            real_scores=real_eval["raw_logits"],
            fake_scores=fake_eval["raw_logits"]
        )
        
        d_loss.backward()
        self.opt_d.step()
        
        # ====================================================================
        # 3. Train Generator
        # ====================================================================
        self.opt_g.zero_grad()
        
        # Re-evaluate fake images with updated discriminator
        # This time we DO NOT detach fused_fake, so gradients flow to G
        fake_eval_for_g = self.discriminator(fused_fake, action)
        
        g_losses = self.gen_loss_engine(
            fake_scores=fake_eval_for_g["raw_logits"],
            fused=fused_fake,
            reference=rgb,
            tl_pred=fake_eval_for_g["t_l"]
        )
        
        g_loss_total = g_losses["total"]
        g_loss_total.backward()
        self.opt_g.step()
        
        # Return detached losses for logging
        return {
            "D_loss": d_loss.item(),
            "G_loss_total": g_loss_total.item(),
            "G_loss_adv": g_losses["ladv"].item(),
            "G_loss_fusion": g_losses["lfusion"].item(),
            "G_loss_latency": g_losses["llatency"].item(),
            "q_f_mean": fake_eval_for_g["q_f"].mean().item(),
            "t_l_mean": fake_eval_for_g["t_l"].mean().item()
        }

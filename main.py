"""
main.py
-------
Entry point for DTM-RL-GAN Tri-Modal Sensor Fusion.
"""
import os
import argparse
import torch

from utils.config import load_config, merge_configs, get_device
from utils.seed import set_seed
from data.utils import build_dataloaders
from shared.losses.generator_loss_engine import GeneratorLossEngine
from shared.losses.adversarial_loss import AdversarialLoss
from shared.reward.reward_engine import RewardEngine
from gan.generator.generator import DynamicGenerator
from gan.discriminator.discriminator import MultiCriticDiscriminator
from rl.agent.ppo_agent import PPOAgent
from rl.memory.rollout_buffer import RolloutBuffer
from rl.update.ppo_update import PPOUpdater
from scene.state_builder import StateBuilder
from training.gan_trainer import GANTrainer
from training.rl_trainer import RLTrainer
from training.trainer import MasterTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="DTM-RL-GAN Training")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Normally we'd load configs from yaml, for now we mock
    print("Initializing DTM-RL-GAN...")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ---------------------------------------------------------
    # 1. Initialize Modules
    # ---------------------------------------------------------
    cfg_mock = {"embed_dim": 64, "ndf": 64}
    action_dim = 10 # As per paper
    state_dim = 256
    
    generator = DynamicGenerator(cfg_mock, action_dim).to(device)
    discriminator = MultiCriticDiscriminator(cfg_mock, action_dim).to(device)
    ppo_agent = PPOAgent(state_dim, action_dim).to(device)
    state_builder = StateBuilder(feature_dim=64, action_dim=action_dim, final_state_dim=state_dim).to(device)
    
    # ---------------------------------------------------------
    # 2. Losses & Optimizers
    # ---------------------------------------------------------
    loss_weights = {"lambda_adv": 1.0, "lambda_fusion": 10.0, "lambda_latency": 0.5, "lambda_task": 5.0}
    gen_loss_engine = GeneratorLossEngine(loss_weights, use_perceptual=False).to(device)
    adv_loss_module = AdversarialLoss("bce").to(device)
    reward_engine = RewardEngine({}).to(device)
    
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    opt_ppo = torch.optim.Adam(ppo_agent.parameters(), lr=1e-4)
    
    # ---------------------------------------------------------
    # 3. Trainers
    # ---------------------------------------------------------
    gan_trainer = GANTrainer(generator, discriminator, opt_g, opt_d, gen_loss_engine, adv_loss_module, device)
    
    rollout_buffer = RolloutBuffer()
    ppo_updater = PPOUpdater({"ppo_batch_size": 2, "ppo_epochs": 2}, ppo_agent, opt_ppo)
    rl_trainer = RLTrainer(ppo_agent, ppo_updater, rollout_buffer, reward_engine, device)
    
    master_trainer = MasterTrainer(
        generator, discriminator, ppo_agent, state_builder, 
        gan_trainer, rl_trainer, device, ppo_update_freq=2
    )
    
    print("Initialization complete. Ready for training.")

if __name__ == "__main__":
    main()

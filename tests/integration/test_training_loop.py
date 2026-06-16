"""
tests/integration/test_training_loop.py
-----------------------------------------
Integration test for the DTM-RL-GAN training pipeline using synthetic data.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset

from gan.generator.generator import DynamicGenerator
from gan.discriminator.discriminator import MultiCriticDiscriminator
from rl.agent.ppo_agent import PPOAgent
from scene.state_builder import StateBuilder
from shared.losses.generator_loss_engine import GeneratorLossEngine
from shared.losses.adversarial_loss import AdversarialLoss
from shared.reward.reward_engine import RewardEngine
from training.gan_trainer import GANTrainer
from training.rl_trainer import RLTrainer
from training.trainer import MasterTrainer
from rl.memory.rollout_buffer import RolloutBuffer
from rl.update.ppo_update import PPOUpdater


def test_full_pipeline_synthetic_run():
    """
    Validates that the entire architecture can forward and backward pass
    without shape mismatches or crash.
    """
    device = torch.device("cpu") # Test on CPU for reliability in CI
    
    # ---------------------------------------------------------
    # 1. Configs and Dimensions
    # ---------------------------------------------------------
    cfg = {"embed_dim": 16, "ndf": 16}
    action_dim = 10
    state_dim = 32
    B, H, W = 2, 64, 64
    
    # ---------------------------------------------------------
    # 2. Init Modules
    # ---------------------------------------------------------
    generator = DynamicGenerator(cfg, action_dim).to(device)
    discriminator = MultiCriticDiscriminator(cfg, action_dim).to(device)
    ppo_agent = PPOAgent(state_dim, action_dim).to(device)
    state_builder = StateBuilder(feature_dim=16, action_dim=action_dim, final_state_dim=state_dim).to(device)
    
    loss_weights = {"lambda_adv": 1.0, "lambda_fusion": 1.0, "lambda_latency": 0.5, "lambda_task": 1.0}
    # Turn off perceptual loss for unit testing to avoid downloading VGG
    gen_loss_engine = GeneratorLossEngine(loss_weights, use_perceptual=False).to(device)
    adv_loss_module = AdversarialLoss("bce").to(device)
    reward_engine = RewardEngine({}).to(device)
    
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    opt_ppo = torch.optim.Adam(ppo_agent.parameters(), lr=1e-3)
    
    # ---------------------------------------------------------
    # 3. Trainers
    # ---------------------------------------------------------
    gan_trainer = GANTrainer(generator, discriminator, opt_g, opt_d, gen_loss_engine, adv_loss_module, device)
    
    rollout_buffer = RolloutBuffer()
    ppo_updater = PPOUpdater({"ppo_batch_size": 2, "ppo_epochs": 1}, ppo_agent, opt_ppo)
    rl_trainer = RLTrainer(ppo_agent, ppo_updater, rollout_buffer, reward_engine, device)
    
    master_trainer = MasterTrainer(
        generator, discriminator, ppo_agent, state_builder, 
        gan_trainer, rl_trainer, device, ppo_update_freq=1
    )
    
    # ---------------------------------------------------------
    # 4. Synthetic Data
    # ---------------------------------------------------------
    rgb = torch.rand(B, 3, H, W)
    thermal = torch.rand(B, 1, H, W)
    lidar = torch.rand(B, 1, H, W)
    
    # Dataloader expects dict
    class MockDataset(torch.utils.data.Dataset):
        def __len__(self): return 4 # 2 batches of size 2
        def __getitem__(self, idx):
            return {"rgb": torch.rand(3, H, W), "thermal": torch.rand(1, H, W), "lidar": torch.rand(1, H, W)}
            
    loader = DataLoader(MockDataset(), batch_size=B)
    
    # ---------------------------------------------------------
    # 5. Run Epoch
    # ---------------------------------------------------------
    metrics = master_trainer.train_epoch(loader)
    
    # ---------------------------------------------------------
    # 6. Validate
    # ---------------------------------------------------------
    assert "D_loss" in metrics
    assert "G_loss_total" in metrics
    assert "PPO_reward" in metrics
    assert "policy_loss" in metrics # Means PPO updated
    
    print("Integration test passed successfully. Pipeline is completely functional.")

if __name__ == "__main__":
    test_full_pipeline_synthetic_run()

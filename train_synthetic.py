import os
import torch
import torchvision.utils as vutils
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

def run_synthetic_training():
    print("Initializing DTM-RL-GAN Training Pipeline on Synthetic Data...")
    device = torch.device("cpu")
    
    # Configs
    cfg = {"embed_dim": 32, "ndf": 32}
    action_dim = 10
    state_dim = 64
    B, H, W = 4, 128, 128
    epochs = 3  # short training for demo
    
    # Init Modules
    generator = DynamicGenerator(cfg, action_dim).to(device)
    discriminator = MultiCriticDiscriminator(cfg, action_dim).to(device)
    ppo_agent = PPOAgent(state_dim, action_dim).to(device)
    state_builder = StateBuilder(feature_dim=32, action_dim=action_dim, final_state_dim=state_dim).to(device)
    
    loss_weights = {"lambda_adv": 1.0, "lambda_fusion": 10.0, "lambda_latency": 0.5, "lambda_task": 1.0}
    gen_loss_engine = GeneratorLossEngine(loss_weights, use_perceptual=False).to(device)
    adv_loss_module = AdversarialLoss("bce").to(device)
    reward_engine = RewardEngine({}).to(device)
    
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-3)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-3)
    opt_ppo = torch.optim.Adam(ppo_agent.parameters(), lr=1e-3)
    
    # Trainers
    gan_trainer = GANTrainer(generator, discriminator, opt_g, opt_d, gen_loss_engine, adv_loss_module, device)
    
    rollout_buffer = RolloutBuffer()
    ppo_updater = PPOUpdater({"ppo_batch_size": 4, "ppo_epochs": 1}, ppo_agent, opt_ppo)
    rl_trainer = RLTrainer(ppo_agent, ppo_updater, rollout_buffer, reward_engine, device)
    
    master_trainer = MasterTrainer(
        generator, discriminator, ppo_agent, state_builder, 
        gan_trainer, rl_trainer, device, ppo_update_freq=1
    )
    
    # Synthetic Data
    class MockDataset(torch.utils.data.Dataset):
        def __len__(self): return 8 # 2 batches of size 4
        def __getitem__(self, idx):
            return {"rgb": torch.rand(3, H, W), "thermal": torch.rand(1, H, W), "lidar": torch.rand(1, H, W)}
            
    loader = DataLoader(MockDataset(), batch_size=B)
    
    print("Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        metrics = master_trainer.train_epoch(loader)
        
        g_loss = metrics.get('G_loss_total', 0.0)
        d_loss = metrics.get('D_loss', 0.0)
        ppo_reward = metrics.get('PPO_reward', 0.0)
        policy_loss = metrics.get('policy_loss', 0.0)
        
        print(f"Generator Loss:     {g_loss:.4f}")
        print(f"Discriminator Loss: {d_loss:.4f}")
        print(f"PPO Reward:         {ppo_reward:.4f}")
        print(f"PPO Policy Loss:    {policy_loss:.4f}")

    print("\nTraining complete! Generating sample output...")
    
    generator.eval()
    ppo_agent.eval()
    
    # Generate one sample
    rgb = torch.rand(1, 3, H, W)
    thermal = torch.rand(1, 1, H, W)
    lidar = torch.rand(1, 1, H, W)
    
    # PPO routing action
    with torch.no_grad():
        features, confidences = generator.encoder(rgb, thermal, lidar)
        scene_state, _ = state_builder(features, confidences, torch.zeros(1, 3, device=device))
        action_dict = ppo_agent.act(scene_state)
        action = action_dict["action"]
        
        fused_image, _, _ = generator(rgb, thermal, lidar, action)
        
    fused_image_01 = (fused_image + 1.0) / 2.0
    
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "trained_synthetic_demo_fused.png")
    
    vutils.save_image(fused_image_01, out_path)
    print(f"Output saved to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    run_synthetic_training()

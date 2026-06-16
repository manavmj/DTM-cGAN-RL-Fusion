import os
import torch
import torchvision.utils as vutils
from utils.config import load_config
from data.utils import build_dataloaders

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

def run_real_training():
    print("Initializing DTM-RL-GAN Training Pipeline on Real Data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configs
    model_cfg = load_config("configs/model_config.yaml")
    data_cfg = load_config("configs/data_config.yaml")
    train_cfg = load_config("configs/training_config.yaml")
    
    # We will use smaller dimensions to ensure it runs quickly for a demo
    cfg = {"embed_dim": 64, "ndf": 64}
    action_dim = 10
    state_dim = 256
    epochs = 1  # 1 epoch for demo
    
    # Init Modules
    generator = DynamicGenerator(cfg, action_dim).to(device)
    discriminator = MultiCriticDiscriminator(cfg, action_dim).to(device)
    ppo_agent = PPOAgent(state_dim, action_dim).to(device)
    state_builder = StateBuilder(feature_dim=64, action_dim=action_dim, final_state_dim=state_dim).to(device)
    
    loss_weights = {"lambda_adv": 1.0, "lambda_fusion": 20.0, "lambda_latency": 0.5, "lambda_task": 2.0}
    gen_loss_engine = GeneratorLossEngine(loss_weights, use_perceptual=True).to(device)
    adv_loss_module = AdversarialLoss("bce").to(device)
    reward_engine = RewardEngine({}).to(device)
    
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
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
    
    # Load Real Data
    print("Building DataLoaders from actual dataset...")
    # Make sure num_workers is 0 for Windows to avoid multiprocess issues during demo
    train_cfg["num_workers"] = 0
    loaders = build_dataloaders(data_cfg, train_cfg)
    train_loader = loaders["train"]
    test_loader = loaders["test"]
    
    print(f"Full dataset size: Train={len(train_loader.dataset)} Test={len(test_loader.dataset)}")
    
    # Subset for fast demo while preserving enough data for graphs
    train_subset = torch.utils.data.Subset(train_loader.dataset, range(160))
    test_subset = torch.utils.data.Subset(test_loader.dataset, range(40))
    
    from torch.utils.data import DataLoader
    from data.utils import trimodal_collate_fn
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, collate_fn=trimodal_collate_fn)
    test_loader = DataLoader(test_subset, batch_size=8, shuffle=False, collate_fn=trimodal_collate_fn)
    
    print(f"Using subset dataset: Train={len(train_loader.dataset)} Test={len(test_loader.dataset)}")
    
    print("Starting Training Loop...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        metrics = master_trainer.train_epoch(train_loader)
        
        g_loss = metrics.get('G_loss_total', 0.0)
        d_loss = metrics.get('D_loss', 0.0)
        ppo_reward = metrics.get('PPO_reward', 0.0)
        policy_loss = metrics.get('policy_loss', 0.0)
        
        print(f"Generator Loss:     {g_loss:.4f}")
        print(f"Discriminator Loss: {d_loss:.4f}")
        print(f"PPO Reward:         {ppo_reward:.4f}")
        print(f"PPO Policy Loss:    {policy_loss:.4f}")

    print("\nTraining complete! Generating sample output from the test set...")
    
    # Save the trained model checkpoint
    os.makedirs("outputs", exist_ok=True)
    torch.save({
        'generator': generator.state_dict(),
        'ppo_agent': ppo_agent.state_dict(),
    }, "outputs/trained_model.pt")
    
    generator.eval()
    ppo_agent.eval()
    
    # Fetch a single batch from the test set
    batch = next(iter(test_loader))
    rgb = batch["rgb"].to(device)
    thermal = batch["thermal"].to(device)
    lidar = batch["lidar"].to(device)
    stem = batch["stem"][0]
    
    # Take only the first image in the batch
    rgb = rgb[0:1]
    thermal = thermal[0:1]
    lidar = lidar[0:1]
    
    # Generate PPO routing action
    with torch.no_grad():
        features, confidences = generator.encoder(rgb, thermal, lidar)
        scene_state, _ = state_builder(features, confidences, torch.zeros(1, 3, device=device))
        action_dict = ppo_agent.act(scene_state)
        action = action_dict["action"]
        
        fused_image, _, _ = generator(rgb, thermal, lidar, action)
        
    fused_image_01 = (fused_image + 1.0) / 2.0
    
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"trained_real_demo_{stem}.png")
    
    # We will save the original RGB, Thermal, LiDAR and Fused output side by side
    combined = torch.cat([rgb, thermal.repeat(1, 3, 1, 1), lidar.repeat(1, 3, 1, 1), fused_image_01.cpu()], dim=3)
    vutils.save_image(combined, out_path)
    print(f"Output saved to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    run_real_training()

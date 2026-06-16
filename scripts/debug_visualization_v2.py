import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

from utils.config import load_config
from data.utils import build_dataloaders
from gan.generator.generator import DynamicGenerator
from rl.agent.ppo_agent import PPOAgent
from scene.state_builder import StateBuilder

def unnormalize(tensor, mean, std):
    """Un-normalize an image tensor using mean and std."""
    tensor = tensor.clone().detach().cpu()
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)

def save_tensor_as_image(tensor, path, cmap=None, is_feature=False):
    """Save a tensor as an image, handling feature maps and gray/RGB."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # remove batch dim
    
    tensor = tensor.detach().cpu()
    
    if is_feature:
        # Take the mean across channels to visualize feature maps
        heatmap = tensor.mean(dim=0).numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap, cmap=cmap if cmap else 'viridis')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        # Image
        if tensor.shape[0] == 3: # RGB
            img = tensor.permute(1, 2, 0).numpy()
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close()
        elif tensor.shape[0] == 1: # Grayscale
            img = tensor.squeeze(0).numpy()
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap=cmap if cmap else 'gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close()

def print_tensor_stats(name, tensor):
    print(f"--- {name} ---")
    print(f"Shape: {list(tensor.shape)}")
    print(f"Min:   {tensor.min().item():.4f}")
    print(f"Max:   {tensor.max().item():.4f}")
    print(f"Mean:  {tensor.mean().item():.4f}\n")

def visualize_and_verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Advanced Debug Verification on {device}...\n")
    
    os.makedirs("outputs/debug", exist_ok=True)
    
    model_cfg = load_config("configs/model_config.yaml")
    data_cfg = load_config("configs/data_config.yaml")
    train_cfg = load_config("configs/training_config.yaml")
    train_cfg["num_workers"] = 0
    
    loaders = build_dataloaders(data_cfg, train_cfg)
    dataloader = loaders["test"]
    
    cfg = {"embed_dim": 64, "ndf": 64}
    action_dim = 10
    state_dim = 256
    
    generator = DynamicGenerator(cfg, action_dim).to(device)
    ppo_agent = PPOAgent(state_dim, action_dim).to(device)
    state_builder = StateBuilder(feature_dim=64, action_dim=action_dim, final_state_dim=state_dim).to(device)
    
    if os.path.exists("outputs/trained_model.pt"):
        checkpoint = torch.load("outputs/trained_model.pt", map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        ppo_agent.load_state_dict(checkpoint['ppo_agent'])
        print("Loaded trained model checkpoint.\n")
    else:
        print("No trained model found. Using untrained weights for debugging.\n")
        
    generator.eval()
    ppo_agent.eval()
    state_builder.eval()
    
    # Get ONE sample
    batch = next(iter(dataloader))
    raw_rgb_t = batch["rgb"][0:1].to(device)
    raw_thermal_t = batch["thermal"][0:1].to(device)
    raw_lidar_t = batch["lidar"][0:1].to(device)
    stem = batch["stem"][0]
    
    print(f"VERIFICATION: Stem loaded: {stem}")
    print("="*40)
    
    # 1. Un-normalize the raw inputs
    # From data_config.yaml:
    rgb_mean = data_cfg["image"]["rgb_mean"]
    rgb_std = data_cfg["image"]["rgb_std"]
    thermal_mean = data_cfg["image"]["thermal_mean"]
    thermal_std = data_cfg["image"]["thermal_std"]
    lidar_mean = data_cfg["image"]["lidar_mean"]
    lidar_std = data_cfg["image"]["lidar_std"]
    
    unnorm_rgb = unnormalize(raw_rgb_t, rgb_mean, rgb_std)
    unnorm_thermal = unnormalize(raw_thermal_t, thermal_mean, thermal_std)
    unnorm_lidar = unnormalize(raw_lidar_t, lidar_mean, lidar_std)
    
    print_tensor_stats("Raw RGB (Un-normalized)", unnorm_rgb)
    print_tensor_stats("Raw Thermal (Un-normalized)", unnorm_thermal)
    print_tensor_stats("Raw LiDAR (Un-normalized)", unnorm_lidar)
    
    save_tensor_as_image(unnorm_rgb, "outputs/debug/raw_rgb.png")
    save_tensor_as_image(unnorm_thermal, "outputs/debug/raw_thermal.png", cmap="inferno")
    save_tensor_as_image(unnorm_lidar, "outputs/debug/raw_lidar.png", cmap="plasma")
    
    # 2. Extract Intermediate Features
    with torch.no_grad():
        # Encoder
        features, confidences = generator.encoder(raw_rgb_t, raw_thermal_t, raw_lidar_t)
        
        print_tensor_stats("RGB Encoder Output", features["rgb"])
        print_tensor_stats("Thermal Encoder Output", features["thermal"])
        print_tensor_stats("LiDAR Encoder Output", features["lidar"])
        
        save_tensor_as_image(features["rgb"], "outputs/debug/rgb_encoder_output.png", is_feature=True)
        save_tensor_as_image(features["thermal"], "outputs/debug/thermal_encoder_output.png", is_feature=True)
        save_tensor_as_image(features["lidar"], "outputs/debug/lidar_encoder_output.png", is_feature=True)
        
        # State Builder
        scene_state, knowledge_retrieval = state_builder(features, confidences, torch.zeros(1, 3, device=device))
        
        print_tensor_stats("Meta Fusion Knowledge Bank Output", knowledge_retrieval)
        
        # Save Knowledge Retrieval as bar plot
        retrieval_np = knowledge_retrieval.squeeze(0).cpu().numpy()
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(retrieval_np)), retrieval_np, color='purple')
        plt.title("Meta Fusion Knowledge Bank Output Vector")
        plt.savefig("outputs/debug/meta_fusion_bank_output.png")
        plt.close()
        
        # PPO Agent
        action_dict = ppo_agent.act(scene_state)
        action = action_dict["action"]
        
        # Adaptive Gating
        topology_controls = generator.adaptive_constructor(action)
        w_deep = topology_controls.get("w_deep", torch.tensor(0.5)).item()
        w_light = topology_controls.get("w_light", torch.tensor(0.5)).item()
        
        # Gating Weights
        weights_df = pd.DataFrame([{"w_deep": w_deep, "w_light": w_light}])
        weights_df.to_csv("outputs/debug/gating_weights.csv", index=False)
        print(f"--- Adaptive Gating Coefficients ---")
        print(f"w_deep:  {w_deep:.4f}")
        print(f"w_light: {w_light:.4f}\n")
        
        plt.figure(figsize=(6, 6))
        plt.bar(["w_deep", "w_light"], [w_deep, w_light], color=['blue', 'orange'])
        plt.ylim(0, 1)
        plt.title("Adaptive Gating Coefficients")
        plt.savefig("outputs/debug/adaptive_gating_coefficients.png")
        plt.close()
        
        # Fusion
        fused_features = generator.topology(
            features["rgb"], features["thermal"], features["lidar"],
            confidences["rgb"], confidences["thermal"], confidences["lidar"],
            topology_controls
        )
        
        print_tensor_stats("Dynamic Tri-Modal Output (Fused Features)", fused_features)
        save_tensor_as_image(fused_features, "outputs/debug/dynamic_tri_modal_output.png", is_feature=True)
        
        # Decoder (Actual generator output tensor)
        fused_image = generator.decoder(fused_features)
        
        # Normalise output from [-1, 1] to [0, 1]
        fused_01 = (fused_image + 1.0) / 2.0
        fused_01 = fused_01.clamp(0, 1)
        
        print_tensor_stats("Final Fused Image (Output Tensor normalized to 0-1)", fused_01)
        save_tensor_as_image(fused_01, "outputs/debug/final_fused_image.png")
        
    print("All debug visualization artifacts have been saved to outputs/debug/")

if __name__ == "__main__":
    visualize_and_verify()

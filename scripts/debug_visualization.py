import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import lpips
import time

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from shared.metrics.fusion_metrics import mutual_information

from utils.config import load_config
from data.utils import build_dataloaders
from gan.generator.generator import DynamicGenerator
from rl.agent.ppo_agent import PPOAgent
from scene.state_builder import StateBuilder

def visualize_and_verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Debug Visualization on {device}...")
    
    # 1. Load Configurations and Dataloader
    model_cfg = load_config("configs/model_config.yaml")
    data_cfg = load_config("configs/data_config.yaml")
    train_cfg = load_config("configs/training_config.yaml")
    train_cfg["num_workers"] = 0
    
    loaders = build_dataloaders(data_cfg, train_cfg)
    dataloader = loaders["test"]
    
    # 2. Instantiate Architecture
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
        print("Loaded trained model checkpoint.")
    else:
        print("No trained model found. Using untrained weights for debugging.")
        
    generator.eval()
    ppo_agent.eval()
    state_builder.eval()
    
    # 3. Setup Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    # 4. Get ONE sample
    batch = next(iter(dataloader))
    rgb = batch["rgb"][0:1].to(device)
    thermal = batch["thermal"][0:1].to(device)
    lidar = batch["lidar"][0:1].to(device)
    stem = batch["stem"][0]
    
    # VERIFICATION: Print exact file paths
    root_path = os.path.join(data_cfg["dataset"]["root"], "test")
    print(f"\n--- FILE VERIFICATION ---")
    print(f"Stem loaded: {stem}")
    print(f"RGB path:     {os.path.join(root_path, 'rgb', stem + '.png')}")
    print(f"Thermal path: {os.path.join(root_path, 'thermal', stem + '.png')}")
    print(f"LiDAR path:   {os.path.join(root_path, 'lidar', stem + '.png')}")
    print(f"-------------------------\n")
    
    # 5. Extract Intermediate Features (Hooks)
    intermediate = {}
    def get_hook(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                if isinstance(output[0], dict):
                    intermediate[name] = output[0]["rgb"].detach()
                else:
                    intermediate[name] = output[0].detach()
            else:
                intermediate[name] = output.detach()
        return hook
    
    # Hook for DTM features (output of MultimodalEncoder)
    generator.encoder.register_forward_hook(get_hook('dtm_features'))
    
    # Run inference
    with torch.no_grad():
        t0 = time.time()
        
        # Generator Encoder
        features, confidences = generator.encoder(rgb, thermal, lidar)
        
        # State Builder (RL State & Meta Fusion Bank)
        scene_state, knowledge_retrieval = state_builder(features, confidences, torch.zeros(1, 3, device=device))
        
        # PPO Agent
        action_dict = ppo_agent.act(scene_state)
        action = action_dict["action"]
        value = action_dict["value"]
        
        # Generator Adaptive Constructor
        topology_controls = generator.adaptive_constructor(action)
        w_deep = topology_controls.get("w_deep", torch.tensor(0.5)).item()
        w_light = topology_controls.get("w_light", torch.tensor(0.5)).item()
        
        # Generator Fusion & Decoder
        fused_features = generator.topology(
            features["rgb"], features["thermal"], features["lidar"],
            confidences["rgb"], confidences["thermal"], confidences["lidar"],
            topology_controls
        )
        fused_image = generator.decoder(fused_features)
        
        t1 = time.time()
        latency = (t1 - t0) * 1000

    # 6. Calculate Metrics
    fused_01 = (fused_image + 1.0) / 2.0
    rgb_01 = rgb
    
    psnr_val = psnr_metric(fused_01, rgb_01).item()
    ssim_val = ssim_metric(fused_01, rgb_01).item()
    mi_val = mutual_information(fused_01, rgb_01).item()
    lpips_val = loss_fn_vgg(fused_image, rgb_01 * 2 - 1.0).item()
    ppo_reward = value.item() # Value proxy for reward since it predicts return
    
    print(f"--- SAMPLE METRICS ---")
    print(f"SSIM:    {ssim_val:.4f}")
    print(f"PSNR:    {psnr_val:.4f} dB")
    print(f"MI:      {mi_val:.4f}")
    print(f"LPIPS:   {lpips_val:.4f}")
    print(f"Latency: {latency:.2f} ms")
    print(f"Reward:  {ppo_reward:.4f} (Predicted Value)")
    print(f"----------------------\n")
    
    # 7. Generate Visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Debugging Visualization - Stem: {stem}", fontsize=16)
    
    def tensor_to_img(t, is_rgb=False):
        t = t.squeeze(0).cpu().numpy()
        if is_rgb:
            t = np.transpose(t, (1, 2, 0))
            return t
        else:
            return t[0] # Return first channel for grayscale
            
    def feature_to_heatmap(feat):
        # Mean across channel dimension
        heatmap = feat.squeeze(0).mean(dim=0).cpu().numpy()
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    # Row 1: Inputs
    axes[0, 0].imshow(tensor_to_img(rgb_01, is_rgb=True))
    axes[0, 0].set_title("RGB Input")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(tensor_to_img(thermal), cmap='inferno')
    axes[0, 1].set_title("Thermal Input")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(tensor_to_img(lidar), cmap='plasma')
    axes[0, 2].set_title("LiDAR Input")
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(tensor_to_img(fused_01, is_rgb=True))
    axes[0, 3].set_title(f"Final Fused Image\n(SSIM: {ssim_val:.3f})")
    axes[0, 3].axis('off')
    
    # Row 2: Intermediate representations
    # DTM Feature (e.g. RGB feature map)
    axes[1, 0].imshow(feature_to_heatmap(features["rgb"]), cmap='viridis')
    axes[1, 0].set_title("DTM RGB Feature Map")
    axes[1, 0].axis('off')
    
    # Meta Fusion Bank (Bar plot of retrieval vector)
    axes[1, 1].bar(range(action_dim), knowledge_retrieval.squeeze(0).cpu().numpy())
    axes[1, 1].set_title("Meta Fusion Bank Output")
    axes[1, 1].set_xlabel("Action Dim")
    
    # Adaptive Gating (Show the weights textually or as pie/bar)
    axes[1, 2].bar(["w_deep", "w_light"], [w_deep, w_light], color=['blue', 'orange'])
    axes[1, 2].set_title(f"Adaptive Gating Weights")
    axes[1, 2].set_ylim(0, 1)
    
    # DTM Fused Feature
    axes[1, 3].imshow(feature_to_heatmap(fused_features), cmap='viridis')
    axes[1, 3].set_title("Pre-Decoder Fused Feature Map")
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    os.makedirs("outputs/debug", exist_ok=True)
    out_path = f"outputs/debug/verification_{stem}.png"
    plt.savefig(out_path)
    print(f"Visualization saved to: {out_path}")

if __name__ == "__main__":
    visualize_and_verify()

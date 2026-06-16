import os
import torch
import torchvision.utils as vutils
from gan.generator.generator import DynamicGenerator

def main():
    print("Initializing DTM-RL-GAN Generator...")
    device = torch.device("cpu")
    
    # Mock configs matching your architecture
    cfg_mock = {"embed_dim": 64, "ndf": 64}
    action_dim = 10
    
    # Initialize an untrained generator
    generator = DynamicGenerator(cfg_mock, action_dim).to(device)
    generator.eval()
    
    # Create synthetic input data (batch size 1, 256x256 resolution)
    # Simulating what real sensors would capture
    print("Generating synthetic sensor data...")
    rgb = torch.rand(1, 3, 256, 256)
    thermal = torch.rand(1, 1, 256, 256)
    lidar = torch.rand(1, 1, 256, 256)
    
    # The PPO agent would normally provide this action (routing weights for fusion)
    # We'll mock an action with shape (1, action_dim)
    action = torch.rand(1, action_dim)
    
    print("Passing through Dynamic Tri-Modal Generator...")
    with torch.no_grad():
        fused_image, _, _ = generator(rgb, thermal, lidar, action)
        
    # The output is in [-1, 1] because of the Tanh activation.
    # Convert it to [0, 1] for saving
    fused_image_01 = (fused_image + 1.0) / 2.0
    
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "synthetic_demo_fused.png")
    
    # Save the generated image
    vutils.save_image(fused_image_01, out_path)
    print(f"Output saved to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()

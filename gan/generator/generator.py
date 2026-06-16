"""
gan/generator/generator.py
----------------------------
Dynamic Tri-Modal Generator (DTM-G) — conditionally instantiates fusion topology.
"""
from __future__ import annotations

__all__ = ["DynamicGenerator"]

import torch
import torch.nn as nn

from shared.encoder.multimodal_encoder import MultimodalEncoder
from fusion.topology.adaptive_constructor import AdaptiveConstructor
from fusion.topology.topology_constructor import TopologyConstructor


class DynamicGenerator(nn.Module):
    """
    The Dynamic Tri-Modal Generator.
    
    1. Extracts modality features and confidences.
    2. Parses the RL action into topology controls.
    3. Fuses features using the dynamically configured topology.
    4. Decodes the fused features back to image space.
    """

    def __init__(self, cfg: dict, action_dim: int) -> None:
        super().__init__()
        
        embed_dim = cfg.get("embed_dim", 64)
        
        # 1. Encoders (Shared)
        self.encoder = MultimodalEncoder(cfg)
        
        # 2. Topology Construction
        self.adaptive_constructor = AdaptiveConstructor(action_dim)
        self.topology = TopologyConstructor(embed_dim)
        
        # 3. Decoder
        # Fuses back to 3-channel (RGB-like) representation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Normalised to [-1, 1]
        )

    def forward(
        self, 
        rgb: torch.Tensor, 
        thermal: torch.Tensor, 
        lidar: torch.Tensor,
        action: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Args:
            rgb:     (B, 3, H, W)
            thermal: (B, 1, H, W)
            lidar:   (B, 1, H, W)
            action:  (B, action_dim) from PPO
            
        Returns:
            fused_image: (B, 3, H, W) in [-1, 1]
            features:    dict of extracted modality features
            confidences: dict of extracted modality confidences
        """
        # Extract features and confidences
        features, confidences = self.encoder(rgb, thermal, lidar)
        
        # Parse action into topology controls
        topology_controls = self.adaptive_constructor(action)
        
        # Execute dynamic fusion
        fused_features = self.topology(
            features["rgb"], features["thermal"], features["lidar"],
            confidences["rgb"], confidences["thermal"], confidences["lidar"],
            topology_controls
        )
        
        # Decode to image
        fused_image = self.decoder(fused_features)
        
        return fused_image, features, confidences

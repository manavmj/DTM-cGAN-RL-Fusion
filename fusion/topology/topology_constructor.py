"""
fusion/topology/topology_constructor.py
-----------------------------------------
Topology Constructor — executes the dynamic fusion computation graph based on RL action.
"""
from __future__ import annotations

__all__ = ["TopologyConstructor"]

import torch
import torch.nn as nn

from fusion.stems.modality_stem import ModalityStem
from fusion.blocks.deep_fusion_block import DeepFusionBlock
from fusion.blocks.light_fusion_block import LightFusionBlock


class TopologyConstructor(nn.Module):
    """
    Executes the dynamic generator graph based on RL actions.
    Combines the Deep and Light fusion paths dynamically.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        
        self.deep_path = DeepFusionBlock(dim)
        self.light_path = LightFusionBlock(dim)

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_th: torch.Tensor,
        f_li: torch.Tensor,
        c_rgb: torch.Tensor,
        c_th: torch.Tensor,
        c_li: torch.Tensor,
        action: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Executes the fusion topology.
        
        Args:
            f_*: Features (B, C, H, W)
            c_*: Confidences (B, 1)
            action: Dict containing 'w_deep', 'w_light' controlling path selection.
            
        Returns:
            (B, C, H, W) Fused representation
        """
        w_deep = action.get("w_deep", torch.tensor(0.5, device=f_rgb.device))
        w_light = action.get("w_light", torch.tensor(0.5, device=f_rgb.device))
        
        # We assume w_deep and w_light are scalars or (B, 1, 1, 1) tensors
        if w_deep.dim() in [1, 2]:
            w_deep = w_deep.view(-1, 1, 1, 1)
            w_light = w_light.view(-1, 1, 1, 1)
            
        # Execute Deep Path
        f_deep = self.deep_path(f_rgb, f_th, f_li)
        
        # Execute Light Path
        f_light = self.light_path(f_rgb, f_th, f_li, c_rgb, c_th, c_li)
        
        # Combine according to RL action (Dynamic Routing)
        fused = (w_deep * f_deep) + (w_light * f_light)
        
        return fused

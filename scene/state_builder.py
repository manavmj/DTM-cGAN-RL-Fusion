"""
scene/state_builder.py
------------------------
State Builder — compiles features, confidences, and memory into PPO state.
"""
from __future__ import annotations

__all__ = ["StateBuilder"]

import torch
import torch.nn as nn

from scene.scene_graph_network import SceneGraphNetwork
from scene.meta_fusion_bank import MetaFusionBank


class StateBuilder(nn.Module):
    """
    Constructs the reinforcement learning state S_t for the PPO agent.
    
    S_t = concat[
        pooled_rgb, pooled_th, pooled_li,
        conf_rgb, conf_th, conf_li,
        scene_graph_emb,
        knowledge_retrieval,
        resource_stats
    ]
    """

    def __init__(self, feature_dim: int, action_dim: int, final_state_dim: int = 256) -> None:
        super().__init__()
        
        # Pool dense feature maps to flat vectors
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.scene_graph = SceneGraphNetwork(feature_dim * 3)
        self.knowledge_bank = MetaFusionBank(state_dim=final_state_dim, action_dim=action_dim)
        
        # Calculate raw state dimension:
        # 3 modalities * feature_dim (from pooling)
        # + 3 confidences
        # + scene_graph (feature_dim * 3, pooled over nodes)
        # + knowledge_retrieval (action_dim)
        # + resource stats (e.g., 3 dims: latency_budget, flop_budget, mem_budget)
        raw_dim = (feature_dim * 3) + 3 + (feature_dim * 3) + action_dim + 3
        
        self.compress = nn.Sequential(
            nn.Linear(raw_dim, final_state_dim * 2),
            nn.LayerNorm(final_state_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_state_dim * 2, final_state_dim)
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        confidences: dict[str, torch.Tensor],
        resource_stats: torch.Tensor,
        previous_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Builds the current state.
        
        Returns:
            current_state: (B, final_state_dim)
            retrieved_knowledge: (B, action_dim)
        """
        B = features["rgb"].shape[0]
        
        # 1. Modality Features
        f_rgb = self.pool(features["rgb"]).view(B, -1)
        f_th  = self.pool(features["thermal"]).view(B, -1)
        f_li  = self.pool(features["lidar"]).view(B, -1)
        
        # 2. Confidences
        c_rgb = confidences["rgb"]
        c_th  = confidences["thermal"]
        c_li  = confidences["lidar"]
        
        # 3. Scene Graph
        concat_feats = torch.cat([features["rgb"], features["thermal"], features["lidar"]], dim=1)
        sg_nodes = self.scene_graph(concat_feats) # (B, num_nodes, feature_dim * 3)
        sg_emb = sg_nodes.mean(dim=1)             # (B, feature_dim * 3)
        
        # 4. Knowledge Retrieval (using previous state, or zeros if t=0)
        if previous_state is None:
            previous_state = torch.zeros(B, self.compress[-1].out_features, device=f_rgb.device)
        
        retrieval = self.knowledge_bank.retrieve(previous_state)
        
        # 5. Concatenate all
        raw_state = torch.cat([
            f_rgb, f_th, f_li,
            c_rgb, c_th, c_li,
            sg_emb,
            retrieval,
            resource_stats
        ], dim=1)
        
        current_state = self.compress(raw_state)
        
        return current_state, retrieval

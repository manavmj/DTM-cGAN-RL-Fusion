"""
scene/scene_graph_network.py
------------------------------
Scene Graph Network — constructs a relational graph from features.
"""
from __future__ import annotations

__all__ = ["SceneGraphNetwork"]

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneGraphNetwork(nn.Module):
    """
    Constructs a semantic graph representation from multimodal features.
    Uses self-attention (Graph Attention style) over a pooled grid of features
    to represent spatial entity relationships.
    """

    def __init__(self, feature_dim: int, num_nodes: int = 16) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        
        # Adaptive pool to fixed number of grid 'nodes'
        # e.g., num_nodes=16 means a 4x4 spatial grid of features
        grid_size = int(num_nodes ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        
        # Graph Message Passing (Attention)
        self.query = nn.Linear(feature_dim, feature_dim // 2)
        self.key   = nn.Linear(feature_dim, feature_dim // 2)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) combined features (e.g., concatenated modalities)
            
        Returns:
            (B, num_nodes, feature_dim) graph node embeddings
        """
        B, C, _, _ = features.shape
        
        # 1) Extract grid nodes
        # (B, C, G, G) -> (B, C, num_nodes) -> (B, num_nodes, C)
        nodes = self.pool(features).view(B, C, -1).transpose(1, 2)
        
        # 2) Graph Attention Message Passing
        q = self.query(nodes) # (B, N, C/2)
        k = self.key(nodes)   # (B, N, C/2)
        v = self.value(nodes) # (B, N, C)
        
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        msg = torch.bmm(attn_weights, v) # (B, N, C)
        
        # Residual + Norm + MLP
        nodes = self.norm(nodes + msg)
        nodes = nodes + self.mlp(nodes)
        
        return nodes

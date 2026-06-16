"""
fusion/attention/cross_attention.py
-------------------------------------
Cross-Modal Attention mechanism for deep fusion.
"""
from __future__ import annotations

__all__ = ["CrossModalAttention"]

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Computes cross-attention from modality N (source) to modality M (target).
    Target provides queries, Source provides keys and values.
    
    CMA_{M<-N} = Softmax((W_Q F_M)(W_K F_N)^T / sqrt(d_h)) W_V F_N
    """

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target: (B, C, H, W) e.g., RGB
            source: (B, C, H, W) e.g., Thermal or LiDAR
            
        Returns:
            (B, C, H, W) attended features
        """
        B, C, H, W = target.shape
        N = H * W
        
        q = self.q_proj(target).view(B, self.num_heads, self.head_dim, N).transpose(2, 3) # (B, H, N, D)
        k = self.k_proj(source).view(B, self.num_heads, self.head_dim, N).transpose(2, 3)
        v = self.v_proj(source).view(B, self.num_heads, self.head_dim, N).transpose(2, 3)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v) # (B, H, N, D)
        
        # Reshape back to spatial dimensions
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        
        return self.out_proj(out)

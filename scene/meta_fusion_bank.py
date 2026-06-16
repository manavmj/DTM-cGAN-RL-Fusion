"""
scene/meta_fusion_bank.py
---------------------------
Meta Fusion Knowledge Bank — historical retrieval of optimal fusion states.
"""
from __future__ import annotations

__all__ = ["MetaFusionBank"]

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaFusionBank(nn.Module):
    """
    Memory bank storing previous high-reward (state, action) pairs.
    Retrieves a 'knowledge vector' using soft attention over stored states.
    """

    def __init__(self, state_dim: int, action_dim: int, bank_size: int = 1024) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bank_size = bank_size
        
        # Non-learnable memory buffers (updated during RL rollouts)
        self.register_buffer("keys", torch.randn(bank_size, state_dim))
        self.register_buffer("values", torch.zeros(bank_size, action_dim))
        self.register_buffer("rewards", torch.zeros(bank_size))
        
        self.ptr = 0
        self.is_full = False
        
        self.query_proj = nn.Linear(state_dim, state_dim)

    def retrieve(self, current_state: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """
        Retrieve knowledge vector based on current state.
        
        Args:
            current_state: (B, state_dim)
            
        Returns:
            (B, action_dim) retrieved prior action embedding
        """
        B = current_state.shape[0]
        
        if self.ptr == 0 and not self.is_full:
            # Bank is empty, return zeros
            return torch.zeros(B, self.action_dim, device=current_state.device)
            
        valid_size = self.bank_size if self.is_full else self.ptr
        valid_keys = self.keys[:valid_size]     # (V, S)
        valid_values = self.values[:valid_size] # (V, A)
        
        q = self.query_proj(current_state)      # (B, S)
        
        # Cosine similarity
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(valid_keys, dim=-1)
        
        sim = torch.mm(q_norm, k_norm.transpose(0, 1)) # (B, V)
        
        # Top-K soft attention
        topk_sim, topk_idx = torch.topk(sim, min(top_k, valid_size), dim=1)
        attn = F.softmax(topk_sim * 10.0, dim=1) # Temperature scaling
        
        # Gather values
        # topk_idx: (B, K)
        retrieved_v = valid_values[topk_idx] # (B, K, A)
        
        # Aggregate
        out = torch.bmm(attn.unsqueeze(1), retrieved_v).squeeze(1) # (B, A)
        
        return out

    @torch.no_grad()
    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> None:
        """Update bank with new experience (FIFO)."""
        batch_size = states.shape[0]
        for i in range(batch_size):
            self.keys[self.ptr] = states[i]
            self.values[self.ptr] = actions[i]
            self.rewards[self.ptr] = rewards[i]
            
            self.ptr += 1
            if self.ptr >= self.bank_size:
                self.ptr = 0
                self.is_full = True

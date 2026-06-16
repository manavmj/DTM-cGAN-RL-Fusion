"""
fusion/topology/adaptive_constructor.py
-----------------------------------------
Adaptive Constructor — maps RL action vector to topology configuration parameters.
"""
from __future__ import annotations

__all__ = ["AdaptiveConstructor"]

import torch
import torch.nn as nn


class AdaptiveConstructor(nn.Module):
    """
    Parses the raw PPO action vector into structured control signals for the generator.
    """

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim

    def forward(self, action_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Maps continuous PPO action [-1, 1] to generator topology controls.
        
        Args:
            action_vec: (B, action_dim)
            
        Returns:
            Dict containing control signals:
                - w_deep: weight for deep fusion path [0, 1]
                - w_light: weight for light fusion path [0, 1]
                - ... other structural controls
        """
        # Map [-1, 1] to [0, 1] for probabilities/weights
        normalized_action = (action_vec + 1.0) / 2.0
        
        # Extract components (assume specific indices based on paper equation 6)
        # At = [wrgb, wth, wlidar, dt, pt, b_deep, b_light, a_att, p_t, r_bud]
        # We focus on b_deep and b_light for path routing
        
        # In this implementation we assume action_vec has at least 7 dims
        w_deep = normalized_action[:, 5:6]
        w_light = normalized_action[:, 6:7]
        
        # Softmax to ensure they sum to 1 (zeta_t and 1 - zeta_t in paper)
        path_weights = torch.cat([w_deep, w_light], dim=1)
        path_weights = torch.softmax(path_weights, dim=1)
        
        return {
            "w_deep": path_weights[:, 0:1],
            "w_light": path_weights[:, 1:2]
        }

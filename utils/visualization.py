"""
utils/visualization.py
-----------------------
Visualization helpers for fused outputs and training metrics.
"""
from __future__ import annotations

__all__ = ["make_image_grid", "save_image_grid", "plot_metrics", "denormalize"]

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils


# Default ImageNet normalisation constants
_DEFAULT_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_STD  = [0.229, 0.224, 0.225]


def denormalize(
    tensor: torch.Tensor,
    mean: list[float] = _DEFAULT_MEAN,
    std: list[float] = _DEFAULT_STD,
) -> torch.Tensor:
    """
    Reverse ImageNet-style normalisation for display.

    Args:
        tensor: (B, C, H, W) or (C, H, W) normalised tensor.
        mean:   Per-channel mean used during normalisation.
        std:    Per-channel std  used during normalisation.

    Returns:
        Tensor in [0, 1] float32.
    """
    t = tensor.clone().float()
    m = torch.tensor(mean, dtype=torch.float32, device=t.device)
    s = torch.tensor(std,  dtype=torch.float32, device=t.device)

    if t.dim() == 4:   # (B, C, H, W)
        m = m.view(1, -1, 1, 1)
        s = s.view(1, -1, 1, 1)
    elif t.dim() == 3: # (C, H, W)
        m = m.view(-1, 1, 1)
        s = s.view(-1, 1, 1)

    return (t * s + m).clamp(0.0, 1.0)


def _to_display(t: torch.Tensor, is_single_channel: bool = False) -> torch.Tensor:
    """Convert a tensor to display format (3-channel, [0,1])."""
    t = t.detach().cpu().float()
    if t.min() < -0.01:          # likely [-1, 1]
        t = (t + 1.0) / 2.0
    t = t.clamp(0.0, 1.0)
    if is_single_channel and t.shape[-3] == 1:
        t = t.repeat(1, 3, 1, 1) if t.dim() == 4 else t.repeat(3, 1, 1)
    return t


def make_image_grid(
    rgb: torch.Tensor,
    thermal: torch.Tensor,
    lidar: torch.Tensor,
    fused: torch.Tensor,
    nrow: int = 4,
) -> torch.Tensor:
    """
    Arrange raw modalities and fused output side-by-side into a comparison grid.

    Columns (per sample): RGB | Thermal | LiDAR | Fused

    Args:
        rgb:     (B, 3, H, W)
        thermal: (B, 1, H, W)
        lidar:   (B, 1, H, W)
        fused:   (B, 3, H, W)
        nrow:    Number of columns in the output grid.

    Returns:
        grid: (3, H_grid, W_grid) float tensor in [0, 1].
    """
    B = rgb.shape[0]
    r = _to_display(rgb)
    th = _to_display(thermal, is_single_channel=True)
    li = _to_display(lidar,   is_single_channel=True)
    fu = _to_display(fused)

    # Interleave: for each sample concatenate 4 images then stack all
    panels = []
    for i in range(B):
        panels.extend([r[i], th[i], li[i], fu[i]])

    all_imgs = torch.stack(panels)   # (B*4, 3, H, W)
    return vutils.make_grid(all_imgs, nrow=nrow * 4, padding=2, normalize=False)


def save_image_grid(grid: torch.Tensor, path: str | Path) -> None:
    """Save a (3, H, W) float tensor image grid to disk as PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, str(path))


def plot_metrics(
    metrics_dict: dict[str, list[float]],
    save_path: str | Path,
    title: str = "Training Metrics",
) -> None:
    """
    Plot training curves for all metrics in metrics_dict and save as PNG.

    Args:
        metrics_dict: {metric_name: [value_at_step_0, value_at_step_1, ...]}
        save_path:    Output PNG file path.
        title:        Figure title.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(metrics_dict)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for idx, (name, values) in enumerate(metrics_dict.items()):
        ax = axes[idx // cols][idx % cols]
        ax.plot(values, linewidth=1.5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

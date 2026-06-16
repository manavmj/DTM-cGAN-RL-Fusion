"""
utils/logger.py
---------------
Structured training logger backed by TensorBoard SummaryWriter.
"""
from __future__ import annotations

__all__ = ["Logger"]

from pathlib import Path
from typing import Any

import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    TensorBoard-backed training logger.

    Wraps SummaryWriter and provides ergonomic helpers for logging
    scalars, image grids, histograms, and hyperparameters.
    """

    def __init__(self, log_dir: str | Path) -> None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._log_dir = log_dir

    # ------------------------------------------------------------------
    # Scalars
    # ------------------------------------------------------------------
    def log_scalar(self, tag: str, value: float | torch.Tensor, step: int) -> None:
        """Log a single scalar value."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        self._writer.add_scalar(tag, value, global_step=step)

    def log_scalars(self, tag_value_dict: dict[str, Any], step: int) -> None:
        """Log multiple scalar values from a flat dict."""
        for tag, value in tag_value_dict.items():
            self.log_scalar(tag, value, step)

    def log_scalars_grouped(
        self,
        main_tag: str,
        tag_scalar_dict: dict[str, float],
        step: int,
    ) -> None:
        """Log multiple scalars under a shared main_tag (grouped chart)."""
        self._writer.add_scalars(main_tag, tag_scalar_dict, global_step=step)

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    def log_images(
        self,
        tag: str,
        image_tensor: torch.Tensor,
        step: int,
        nrow: int = 4,
    ) -> None:
        """
        Log an image grid to TensorBoard.

        Args:
            tag:          Name for the image panel.
            image_tensor: (N, C, H, W) float tensor in [-1, 1] or [0, 1].
            step:         Global step counter.
            nrow:         Number of images per row in the grid.
        """
        img = image_tensor.detach().cpu().float()
        # Normalise to [0, 1] if values are in [-1, 1]
        if img.min() < -0.01:
            img = (img + 1.0) / 2.0
        img = img.clamp(0.0, 1.0)
        grid = vutils.make_grid(img, nrow=nrow, normalize=False)
        self._writer.add_image(tag, grid, global_step=step)

    # ------------------------------------------------------------------
    # Histograms
    # ------------------------------------------------------------------
    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: int,
    ) -> None:
        """Log a histogram of tensor values."""
        self._writer.add_histogram(tag, values.detach().cpu(), global_step=step)

    def log_model_gradients(self, model: torch.nn.Module, step: int) -> None:
        """Log gradient histograms for all parameters that received gradients."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.log_histogram(f"grads/{name}", param.grad, step)

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    def log_hparams(
        self,
        hparam_dict: dict[str, Any],
        metric_dict: dict[str, float],
    ) -> None:
        """Log hyperparameters to the TensorBoard hparams dashboard."""
        # Flatten nested dicts into dot-separated keys
        flat_hparams = _flatten_dict(hparam_dict)
        # TensorBoard hparams only accepts str/bool/float/int values
        safe_hparams = {
            k: v for k, v in flat_hparams.items()
            if isinstance(v, (str, bool, float, int))
        }
        self._writer.add_hparams(safe_hparams, metric_dict)

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------
    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a text string (useful for config dumps, assumptions, etc.)."""
        self._writer.add_text(tag, text, global_step=step)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def log_dir(self) -> Path:
        return self._log_dir


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

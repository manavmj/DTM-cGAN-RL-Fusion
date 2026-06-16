"""
utils/checkpoint.py
--------------------
Save, load, and resume training checkpoints.
"""
from __future__ import annotations

__all__ = ["save_checkpoint", "load_checkpoint", "resume_from", "keep_last_n"]

import re
import tempfile
from pathlib import Path

import torch


# Checkpoint filename pattern: checkpoint_epoch{N:04d}_step{M:08d}.pt
_CKPT_PATTERN = re.compile(r"checkpoint_epoch(\d+)_step(\d+)\.pt")


def save_checkpoint(path: str | Path, payload: dict) -> None:
    """
    Atomically save a checkpoint dict to disk.

    Uses a write-to-temp + rename strategy to avoid corruption on interrupt.

    Expected payload keys:
        epoch, step, generator, discriminator, ppo_agent,
        opt_gen, opt_disc, opt_ppo, best_metric, config
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory then rename (atomic on most FS)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, suffix=".tmp"
    ) as tmp:
        tmp_path = Path(tmp.name)

    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    """Load a checkpoint dict from disk, mapping tensors to the given device."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def resume_from(checkpoint_dir: str | Path) -> dict | None:
    """
    Find and load the most recent checkpoint in a directory.

    Checkpoints are ranked by (epoch, step) descending.

    Returns:
        Checkpoint dict, or None if the directory contains no checkpoints.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    candidates = []
    for p in checkpoint_dir.glob("checkpoint_epoch*_step*.pt"):
        m = _CKPT_PATTERN.match(p.name)
        if m:
            epoch, step = int(m.group(1)), int(m.group(2))
            candidates.append((epoch, step, p))

    if not candidates:
        return None

    # Sort descending by (epoch, step)
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    latest = candidates[0][2]
    print(f"[checkpoint] Resuming from: {latest}")
    return load_checkpoint(latest)


def keep_last_n(checkpoint_dir: str | Path, n: int) -> None:
    """Delete all but the n most recent checkpoints in a directory."""
    if n <= 0:
        return
    checkpoint_dir = Path(checkpoint_dir)
    candidates = []
    for p in checkpoint_dir.glob("checkpoint_epoch*_step*.pt"):
        m = _CKPT_PATTERN.match(p.name)
        if m:
            epoch, step = int(m.group(1)), int(m.group(2))
            candidates.append((epoch, step, p))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    for _, _, p in candidates[n:]:
        p.unlink(missing_ok=True)


def build_checkpoint_name(epoch: int, step: int) -> str:
    """Build a checkpoint filename from epoch and step."""
    return f"checkpoint_epoch{epoch:04d}_step{step:08d}.pt"

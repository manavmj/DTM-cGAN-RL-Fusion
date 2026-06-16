"""
utils/config.py
---------------
YAML config loader and deep-merger.
"""
from __future__ import annotations

__all__ = ["load_config", "merge_configs", "get_device"]

import copy
from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as a nested dict."""
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(*cfgs: dict) -> dict:
    """
    Deep-merge multiple config dicts.
    Later dicts override earlier ones for scalar values;
    nested dicts are recursively merged.
    """
    result: dict = {}
    for cfg in cfgs:
        _deep_update(result, cfg)
    return result


def _deep_update(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


def get_device(train_cfg: dict) -> torch.device:
    """Return the appropriate torch.device from training config."""
    device_str = train_cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[config] CUDA requested but not available — falling back to CPU.")
        device_str = "cpu"
    return torch.device(device_str)

"""
shared/metrics/latency_metrics.py
-----------------------------------
Hardware performance / latency measurement utilities.
"""
from __future__ import annotations

__all__ = ["measure_latency", "count_flops", "get_gpu_utilization"]

import statistics
import time
from typing import Any

import torch
import torch.nn as nn


def measure_latency(
    model: nn.Module,
    inputs: tuple,
    n_warmup: int = 5,
    n_runs: int = 20,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """
    Measure real wall-clock inference latency.

    Args:
        model:    nn.Module to benchmark.
        inputs:   Tuple of input tensors (already on the right device).
        n_warmup: Number of warmup passes (not counted).
        n_runs:   Number of timed passes.
        device:   Device string or torch.device.

    Returns:
        dict with keys: mean_ms, std_ms, min_ms, max_ms, fps
    """
    model.eval()
    dev = torch.device(device) if isinstance(device, str) else device
    is_cuda = dev.type == "cuda"

    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = model(*inputs)
            if is_cuda:
                torch.cuda.synchronize()

        # Timed runs
        times_ms: list[float] = []
        for _ in range(n_runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(*inputs)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    return {
        "mean_ms": mean_ms,
        "std_ms":  statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms":  min(times_ms),
        "max_ms":  max(times_ms),
        "fps":     1000.0 / (mean_ms + 1e-9),
    }


def count_flops(
    model: nn.Module,
    input_shapes: list[tuple],
    device: str = "cpu",
) -> dict[str, int]:
    """
    Count approximate FLOPs and parameters for a model.

    Uses a hook-based approach to count MACs for Conv2d and Linear layers.
    MACs are multiplied by 2 to get FLOPs.

    Args:
        model:        nn.Module.
        input_shapes: List of (C, H, W) shapes for each input tensor.
        device:       Device to run the forward pass on.

    Returns:
        dict with keys: total_flops, total_macs, total_params
    """
    total_macs: list[int] = [0]
    hooks = []

    def _conv_hook(module, inp, out):
        in_t = inp[0]
        Cin  = in_t.shape[1]
        Cout = out.shape[1]
        kH, kW = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        oH, oW = out.shape[2], out.shape[3]
        groups = module.groups
        macs = (Cin // groups) * kH * kW * Cout * oH * oW
        total_macs[0] += macs

    def _linear_hook(module, inp, out):
        in_t = inp[0]
        macs = in_t.shape[-1] * out.shape[-1]
        total_macs[0] += macs

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    dev = torch.device(device)
    dummy_inputs = [
        torch.zeros(1, *shape, device=dev) for shape in input_shapes
    ]
    with torch.no_grad():
        model.to(dev)(*dummy_inputs)

    for h in hooks:
        h.remove()

    total_params = sum(p.numel() for p in model.parameters())

    return {
        "total_macs":   total_macs[0],
        "total_flops":  total_macs[0] * 2,
        "total_params": total_params,
    }


def get_gpu_utilization() -> dict[str, float]:
    """
    Get current GPU memory utilisation.

    Returns:
        dict with keys: gpu_util_pct (0.0 if not available),
                        mem_used_mb, mem_total_mb, mem_util_pct
    """
    if not torch.cuda.is_available():
        return {
            "gpu_util_pct": 0.0,
            "mem_used_mb":  0.0,
            "mem_total_mb": 0.0,
            "mem_util_pct": 0.0,
        }

    mem_used  = torch.cuda.memory_allocated() / 1024 ** 2
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    mem_util  = mem_used / (mem_total + 1e-9) * 100.0

    # nvidia-smi GPU util is not accessible from pure PyTorch;
    # we return memory-based utilisation as a proxy
    return {
        "gpu_util_pct": mem_util,
        "mem_used_mb":  mem_used,
        "mem_total_mb": mem_total,
        "mem_util_pct": mem_util,
    }

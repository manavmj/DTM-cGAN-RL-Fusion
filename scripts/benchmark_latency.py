"""
scripts/benchmark_latency.py
------------------------------
Standalone latency benchmarking for the generator.

Measures:
    - Mean / std inference time (ms) over N runs
    - FPS at given batch size
    - GPU memory usage
    - Per-branch comparison (Light vs Deep vs Both)

Usage:
    python scripts/benchmark_latency.py
        --checkpoint  outputs/checkpoints/best.pt
        --batch-size  1
        --n-runs      100
        --input-size  256
        --device      cuda
"""
from __future__ import annotations


def main() -> None:
    """Run latency benchmark and print results table."""
    ...


if __name__ == "__main__":
    main()

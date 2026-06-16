"""
scripts/export_onnx.py
-----------------------
Export the Generator backbone to ONNX format.

Exports ONLY the generator backbone (stems + topology constructor +
cross-attention integration), NOT the scene modules or PPO agent.

The action is fixed at export time:
    branch  = 1  (Deep)          — default, configurable via --branch
    weights = [1/3, 1/3, 1/3]   — equal modality blend

Usage:
    python scripts/export_onnx.py
        --checkpoint  outputs/checkpoints/best.pt
        --output      outputs/generator_backbone.onnx
        --branch      1                     # 0=Light, 1=Deep, 2=Both
        --input-size  256                   # spatial resolution
        --opset       17
        --verify                            # run onnxruntime verification
"""
from __future__ import annotations


def main() -> None:
    """Export generator backbone to ONNX."""
    ...


if __name__ == "__main__":
    main()

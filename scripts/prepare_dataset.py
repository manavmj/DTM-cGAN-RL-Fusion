"""
scripts/prepare_dataset.py
---------------------------
Convert raw multi-modal datasets to the expected directory layout.

Supported input formats:
    - FLIR Thermal Dataset (paired RGB + Thermal)
    - KITTI (RGB + LiDAR point clouds → depth maps)
    - Custom triplet folders

Usage:
    python scripts/prepare_dataset.py --source  /path/to/raw
                                      --dest    data/raw/
                                      --format  flir | kitti | custom
                                      --split   0.8 0.1 0.1
"""
from __future__ import annotations


def main() -> None:
    """Entry point for dataset preparation script."""
    ...


if __name__ == "__main__":
    main()

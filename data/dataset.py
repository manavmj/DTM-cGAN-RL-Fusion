"""
data/dataset.py
---------------
TriModalFusionDataset — aligned RGB / Thermal / LiDAR triplets.
"""
from __future__ import annotations

__all__ = ["TriModalFusionDataset"]

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.transforms import TriModalTransform


class TriModalFusionDataset(Dataset):
    """
    Multi-modal dataset returning aligned RGB, Thermal, LiDAR triplets.

    Directory layout under root/<split>/:
        rgb/      *.png   — 3-channel colour
        thermal/  *.png   — 1-channel thermal
        lidar/    *.png   — 1-channel depth / range map
        labels/   *.txt   — YOLO-format boxes (optional)

    Files are matched by filename stem (e.g. '0001' → 0001.png).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[TriModalTransform] = None,
        load_labels: bool = False,
        corruption_cfg: Optional[dict] = None,
    ) -> None:
        self.root = Path(root) / split
        self.transform = transform
        self.load_labels = load_labels
        self.corruption_cfg = corruption_cfg or {}

        # Collect stems that exist in all three modality folders
        rgb_dir = self.root / "rgb"
        th_dir  = self.root / "thermal"
        li_dir  = self.root / "lidar"

        # Validate directories exist
        for d, name in [(rgb_dir, "rgb"), (th_dir, "thermal"), (li_dir, "lidar")]:
            if not d.exists():
                raise FileNotFoundError(
                    f"[TriModalFusionDataset] Directory not found: {d}\n"
                    f"Expected: root/<split>/{name}/*.png"
                )

        rgb_stems = {p.stem for p in sorted(rgb_dir.glob("*.png"))}
        th_stems  = {p.stem for p in sorted(th_dir.glob("*.png"))}
        li_stems  = {p.stem for p in sorted(li_dir.glob("*.png"))}

        common_stems = sorted(rgb_stems & th_stems & li_stems)
        if len(common_stems) == 0:
            raise RuntimeError(
                f"[TriModalFusionDataset] No common stems found across "
                f"rgb/ thermal/ lidar/ in {self.root}"
            )

        self.stems: list[str] = common_stems
        self.rgb_dir = rgb_dir
        self.th_dir  = th_dir
        self.li_dir  = li_dir
        self.label_dir = self.root / "labels" if load_labels else None

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        # ----------------------------------------------------------------
        # Load raw images
        # ----------------------------------------------------------------
        rgb_img = np.array(Image.open(self.rgb_dir / f"{stem}.png").convert("RGB"))
        th_img  = np.array(Image.open(self.th_dir  / f"{stem}.png").convert("L"))
        li_img  = np.array(Image.open(self.li_dir  / f"{stem}.png").convert("L"))

        # ----------------------------------------------------------------
        # Apply transforms
        # ----------------------------------------------------------------
        if self.transform is not None:
            rgb_t, th_t, li_t = self.transform(rgb_img, th_img, li_img)
        else:
            rgb_t = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
            th_t  = torch.from_numpy(th_img).unsqueeze(0).float() / 255.0
            li_t  = torch.from_numpy(li_img).unsqueeze(0).float() / 255.0

        sample: dict = {
            "rgb":     rgb_t,      # (3, H, W)
            "thermal": th_t,       # (1, H, W)
            "lidar":   li_t,       # (1, H, W)
            "stem":    stem,
        }

        # ----------------------------------------------------------------
        # Optionally load YOLO-format labels
        # ----------------------------------------------------------------
        if self.load_labels and self.label_dir is not None:
            label_path = self.label_dir / f"{stem}.txt"
            if label_path.exists():
                sample["labels"] = _load_yolo_labels(label_path)
            else:
                sample["labels"] = torch.zeros((0, 5), dtype=torch.float32)

        return sample


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_yolo_labels(path: Path) -> torch.Tensor:
    """
    Load YOLO-format label file.

    Format per line: class_id  cx  cy  w  h   (all normalised [0,1])

    Returns:
        (N, 5) float32 tensor [class_id, cx, cy, w, h]
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            rows.append([float(x) for x in parts])

    if rows:
        return torch.tensor(rows, dtype=torch.float32)
    return torch.zeros((0, 5), dtype=torch.float32)

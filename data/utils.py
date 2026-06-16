"""
data/utils.py
-------------
DataLoader builders, collate function, and modality stacking.
"""
from __future__ import annotations

__all__ = [
    "trimodal_collate_fn",
    "build_dataset",
    "build_dataloader",
    "build_dataloaders",
    "stack_modalities",
]

import torch
from torch.utils.data import DataLoader

from data.dataset import TriModalFusionDataset
from data.transforms import build_transform


def trimodal_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    Custom collate function that handles variable-length labels gracefully.

    Stacks rgb/thermal/lidar tensors normally.
    For labels, pads with zeros if shapes differ and stacks (max_N, 5).
    """
    keys = batch[0].keys()
    out: dict = {}

    for key in keys:
        if key == "stem":
            out[key] = [s[key] for s in batch]
            continue

        if key == "labels":
            # Pad labels to the same number of boxes
            label_list = [s[key] for s in batch]
            max_n = max(lb.shape[0] for lb in label_list)
            if max_n == 0:
                out[key] = torch.zeros(len(batch), 0, 5, dtype=torch.float32)
            else:
                padded = []
                for lb in label_list:
                    pad_n = max_n - lb.shape[0]
                    if pad_n > 0:
                        lb = torch.cat([lb, torch.zeros(pad_n, 5)], dim=0)
                    padded.append(lb)
                out[key] = torch.stack(padded, dim=0)  # (B, max_N, 5)
            continue

        # Default: stack tensors
        out[key] = torch.stack([s[key] for s in batch], dim=0)

    return out


def stack_modalities(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Concatenate rgb, thermal, lidar along channel dimension.

    Args:
        batch: dict with keys "rgb" (B,3,H,W), "thermal" (B,1,H,W), "lidar" (B,1,H,W)

    Returns:
        (B, 5, H, W) concatenated tensor
    """
    return torch.cat([batch["rgb"], batch["thermal"], batch["lidar"]], dim=1)


def build_dataset(
    data_cfg: dict,
    split: str,
    load_labels: bool = False,
) -> TriModalFusionDataset:
    """Build a TriModalFusionDataset for the given split."""
    transform = build_transform(data_cfg, split)
    return TriModalFusionDataset(
        root=data_cfg["dataset"]["root"],
        split=split,
        transform=transform,
        load_labels=load_labels,
        corruption_cfg=data_cfg.get("corruption"),
    )


def build_dataloader(
    dataset: TriModalFusionDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    **kwargs,
) -> DataLoader:
    """Build a DataLoader from a TriModalFusionDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=trimodal_collate_fn,
        **kwargs,
    )


def build_dataloaders(
    data_cfg: dict,
    train_cfg: dict,
) -> dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from config.

    Returns:
        dict with keys "train", "val", "test"
    """
    load_labels = data_cfg.get("labels", {}).get("format") is not None
    bs          = train_cfg.get("batch_size", 8)
    nw          = train_cfg.get("num_workers", 4)
    pin         = train_cfg.get("pin_memory", True)

    loaders: dict[str, DataLoader] = {}

    for split, shuffle in [("train", True), ("val", False), ("test", False)]:
        split_key = f"{split}_split"
        split_name = data_cfg["dataset"].get(split_key, split)
        ds = build_dataset(data_cfg, split_name, load_labels=load_labels)
        loaders[split] = build_dataloader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=pin,
            drop_last=(split == "train"),
        )

    return loaders

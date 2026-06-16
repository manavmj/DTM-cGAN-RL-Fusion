"""
shared/metrics/task_metrics.py
--------------------------------
Downstream task performance metrics (mAP, precision, recall).
Pure PyTorch — no external metric library dependency.
"""
from __future__ import annotations

__all__ = ["compute_map", "batch_detection_score"]

import torch


def compute_map(
    predictions: list[dict],
    targets: list[dict],
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """
    Compute mean Average Precision (mAP) for object detection.

    Args:
        predictions: List[dict] each with keys:
            boxes   (N, 4) xyxy, normalised [0,1]
            scores  (N,)   confidence
            labels  (N,)   int class IDs
        targets: List[dict] each with keys:
            boxes   (M, 4) xyxy, normalised [0,1]
            labels  (M,)   int class IDs
        iou_thresholds: List of IoU thresholds to evaluate at.
                        Defaults to [0.5] for mAP50 and [0.5:0.95:0.05] for mAP.

    Returns:
        dict{map50, map75, map, precision, recall}
    """
    if iou_thresholds is None:
        iou_thresholds = [round(t, 2) for t in torch.arange(0.5, 1.0, 0.05).tolist()]

    # Collect per-class results
    class_ids = set()
    for t in targets:
        class_ids.update(t["labels"].tolist())
    class_ids = sorted(class_ids)

    aps_per_threshold: dict[float, float] = {}
    precisions_all: list[float] = []
    recalls_all:    list[float] = []

    for iou_th in iou_thresholds:
        aps: list[float] = []
        for cls in class_ids:
            ap, prec, rec = _compute_ap_for_class(predictions, targets, cls, iou_th)
            aps.append(ap)
            if iou_th == 0.5:
                precisions_all.append(prec)
                recalls_all.append(rec)
        aps_per_threshold[iou_th] = float(torch.tensor(aps).mean()) if aps else 0.0

    map50 = aps_per_threshold.get(0.5, 0.0)
    map75 = aps_per_threshold.get(0.75, 0.0)
    mapp  = float(torch.tensor(list(aps_per_threshold.values())).mean()) if aps_per_threshold else 0.0

    mean_prec = float(torch.tensor(precisions_all).mean()) if precisions_all else 0.0
    mean_rec  = float(torch.tensor(recalls_all).mean())    if recalls_all    else 0.0

    return {
        "map50":     map50,
        "map75":     map75,
        "map":       mapp,
        "precision": mean_prec,
        "recall":    mean_rec,
    }


def batch_detection_score(
    predictions: list[dict],
    targets: list[dict],
) -> float:
    """
    Compute mean mAP50 over a batch as a single scalar.

    Args:
        predictions: See compute_map.
        targets:     See compute_map.

    Returns:
        mAP50 scalar float.
    """
    result = compute_map(predictions, targets, iou_thresholds=[0.5])
    return result["map50"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iou_matrix(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    Args:
        boxes_a: (N, 4) xyxy
        boxes_b: (M, 4) xyxy

    Returns:
        (N, M) IoU matrix
    """
    # Intersection
    x1 = torch.max(boxes_a[:, 0].unsqueeze(1), boxes_b[:, 0].unsqueeze(0))
    y1 = torch.max(boxes_a[:, 1].unsqueeze(1), boxes_b[:, 1].unsqueeze(0))
    x2 = torch.min(boxes_a[:, 2].unsqueeze(1), boxes_b[:, 2].unsqueeze(0))
    y2 = torch.min(boxes_a[:, 3].unsqueeze(1), boxes_b[:, 3].unsqueeze(0))

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)   # (N, M)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter
    return inter / (union + 1e-9)


def _compute_ap_for_class(
    predictions: list[dict],
    targets: list[dict],
    class_id: int,
    iou_threshold: float,
) -> tuple[float, float, float]:
    """
    Compute Average Precision for a single class at a given IoU threshold.

    Returns: (AP, mean_precision, mean_recall) floats.
    """
    all_scores:    list[float] = []
    all_tp:        list[int]   = []
    all_fp:        list[int]   = []
    n_gt_total:    int = 0

    for pred, tgt in zip(predictions, targets):
        # Filter by class
        p_mask = pred["labels"] == class_id
        t_mask = tgt["labels"] == class_id

        p_boxes  = pred["boxes"][p_mask]
        p_scores = pred["scores"][p_mask]
        t_boxes  = tgt["boxes"][t_mask]

        n_gt_total += t_boxes.shape[0]

        if p_boxes.shape[0] == 0:
            continue

        # Sort by descending confidence
        order = p_scores.argsort(descending=True)
        p_boxes  = p_boxes[order]
        p_scores = p_scores[order]

        matched = torch.zeros(t_boxes.shape[0], dtype=torch.bool)

        for i in range(p_boxes.shape[0]):
            tp, fp = 0, 0
            if t_boxes.shape[0] > 0:
                iou_vals = _iou_matrix(p_boxes[i:i+1], t_boxes).squeeze(0)  # (M,)
                best_iou, best_j = iou_vals.max(0) if iou_vals.numel() > 0 else (torch.tensor(0.0), torch.tensor(0))
                if best_iou.item() >= iou_threshold and not matched[best_j.item()]:
                    tp = 1
                    matched[best_j.item()] = True
                else:
                    fp = 1
            else:
                fp = 1

            all_scores.append(p_scores[i].item())
            all_tp.append(tp)
            all_fp.append(fp)

    if not all_scores:
        return 0.0, 0.0, 0.0

    # Sort all predictions by score
    order = sorted(range(len(all_scores)), key=lambda i: -all_scores[i])
    tp_sorted = torch.tensor([all_tp[i] for i in order], dtype=torch.float32)
    fp_sorted = torch.tensor([all_fp[i] for i in order], dtype=torch.float32)

    tp_cum = tp_sorted.cumsum(0)
    fp_cum = fp_sorted.cumsum(0)

    recall    = tp_cum / (n_gt_total + 1e-9)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)

    ap = _compute_ap_from_pr(precision, recall)
    return (
        float(ap),
        float(precision[-1]) if precision.numel() > 0 else 0.0,
        float(recall[-1])    if recall.numel()    > 0 else 0.0,
    )


def _compute_ap_from_pr(
    precision: torch.Tensor,
    recall: torch.Tensor,
) -> float:
    """
    Compute AP using 101-point interpolation (COCO style).
    """
    rec_thresholds = torch.linspace(0, 1, 101, device=precision.device)
    ap = 0.0
    for thr in rec_thresholds:
        prec_at = precision[recall >= thr]
        if prec_at.numel() > 0:
            ap += prec_at.max().item()
    return ap / 101.0

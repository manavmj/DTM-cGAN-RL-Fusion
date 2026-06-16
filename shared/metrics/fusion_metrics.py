"""
shared/metrics/fusion_metrics.py
----------------------------------
Differentiable image fusion quality metrics (torch-only, no external libs).

All metrics operate on (B, C, H, W) float32 tensors.
"""
from __future__ import annotations

__all__ = [
    "entropy_score",
    "mutual_information",
    "average_gradient",
    "spatial_frequency",
    "ssim",
    "compute_all",
]

import torch
import torch.nn.functional as F


# ============================================================
# Individual metrics
# ============================================================

def entropy_score(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalised information entropy of an image batch.

    Approximates the histogram via soft binning with 256 bins.

    Args:
        x:   (B, C, H, W) in any range — will be normalised to [0, 1]

    Returns:
        (B,) entropy values in nats (maximise for rich information content)
    """
    B = x.shape[0]
    # Flatten spatial and channel dims, normalise to [0, 1]
    x_flat = x.reshape(B, -1)
    x_flat = (x_flat - x_flat.min(dim=1, keepdim=True).values) / (
        x_flat.max(dim=1, keepdim=True).values
        - x_flat.min(dim=1, keepdim=True).values
        + eps
    )

    # Soft histogram: 256 bins via Gaussian kernels (differentiable)
    num_bins = 256
    bin_centers = torch.linspace(0, 1, num_bins, device=x.device)  # (K,)
    sigma = 1.0 / num_bins

    # (B, N, 1) - (1, 1, K) → (B, N, K) → sum over N → (B, K)
    diffs = (x_flat.unsqueeze(-1) - bin_centers.view(1, 1, -1)) / sigma
    weights = torch.exp(-0.5 * diffs ** 2)
    hist = weights.sum(dim=1)                           # (B, K)
    hist = hist / (hist.sum(dim=1, keepdim=True) + eps) # normalise to PMF

    entropy = -(hist * (hist + eps).log()).sum(dim=1)   # (B,)
    return entropy


def mutual_information(
    x: torch.Tensor,
    y: torch.Tensor,
    num_bins: int = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Mutual information between two image batches (soft histogram method).

    MI(X, Y) = H(X) + H(Y) - H(X, Y)

    Args:
        x:        (B, C, H, W)
        y:        (B, C, H, W)
        num_bins: Number of histogram bins per dimension.

    Returns:
        (B,) MI values (higher = more shared information).
    """
    B = x.shape[0]

    def _to_flat_norm(t: torch.Tensor) -> torch.Tensor:
        f = t.reshape(B, -1)
        lo, hi = f.min(1, keepdim=True).values, f.max(1, keepdim=True).values
        return (f - lo) / (hi - lo + eps)

    xf = _to_flat_norm(x)   # (B, N)
    yf = _to_flat_norm(y)

    centers = torch.linspace(0, 1, num_bins, device=x.device)
    sigma = 1.0 / num_bins

    def _hist(v: torch.Tensor) -> torch.Tensor:
        d = (v.unsqueeze(-1) - centers.view(1, 1, -1)) / sigma
        h = torch.exp(-0.5 * d ** 2).sum(1)      # (B, K)
        return h / (h.sum(1, keepdim=True) + eps)

    px = _hist(xf)   # (B, K)
    py = _hist(yf)

    # Joint histogram (B, K, K)
    dx = (xf.unsqueeze(-1) - centers.view(1, 1, -1)) / sigma
    dy = (yf.unsqueeze(-1) - centers.view(1, 1, -1)) / sigma
    wx = torch.exp(-0.5 * dx ** 2)   # (B, N, K)
    wy = torch.exp(-0.5 * dy ** 2)   # (B, N, K)
    pxy = torch.bmm(wx.transpose(1, 2), wy) / (xf.shape[1] + eps)  # (B, K, K)
    pxy = pxy / (pxy.sum(dim=(1, 2), keepdim=True) + eps)

    def _entropy(p: torch.Tensor) -> torch.Tensor:
        return -(p * (p + eps).log()).sum(dim=list(range(1, p.dim())))

    return _entropy(px) + _entropy(py) - _entropy(pxy)   # (B,)


def average_gradient(x: torch.Tensor) -> torch.Tensor:
    """
    Average gradient (Sobel-based sharpness measure).

    AG = mean(sqrt(Gx² + Gy²))

    Args:
        x: (B, C, H, W) — grayscale or RGB (averaged over channels)

    Returns:
        (B,) gradient magnitude means.
    """
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)   # → (B, 1, H, W)

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-2, -1)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)

    mag = (gx ** 2 + gy ** 2 + 1e-12).sqrt()   # (B, 1, H, W)
    return mag.mean(dim=[1, 2, 3])              # (B,)


def spatial_frequency(x: torch.Tensor) -> torch.Tensor:
    """
    Spatial frequency: combined row + column frequency energy.

    SF = sqrt(RF² + CF²)  where
        RF = sqrt(mean((x[i,j] - x[i,j-1])²))
        CF = sqrt(mean((x[i,j] - x[i-1,j])²))

    Args:
        x: (B, C, H, W)

    Returns:
        (B,) spatial frequency values.
    """
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)

    rf_sq = ((x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2).mean(dim=[1, 2, 3])
    cf_sq = ((x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2).mean(dim=[1, 2, 3])
    return (rf_sq + cf_sq + 1e-12).sqrt()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) — differentiable PyTorch implementation.

    Args:
        pred:        (B, C, H, W) predicted image
        target:      (B, C, H, W) reference image
        window_size: Gaussian window size (default 11)

    Returns:
        (B,) SSIM values in [-1, 1] (1.0 = identical)
    """
    B, C, H, W = pred.shape
    win = _gaussian_window(window_size, pred.device, pred.dtype)
    # Expand kernel for multi-channel
    win = win.expand(C, 1, window_size, window_size)

    pad = window_size // 2

    def _filter(x):
        return F.conv2d(x, win, padding=pad, groups=C)

    mu1 = _filter(pred)
    mu2 = _filter(target)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _filter(pred   ** 2) - mu1_sq
    sigma2_sq = _filter(target ** 2) - mu2_sq
    sigma12   = _filter(pred * target) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = (
        (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps
    )

    return ssim_map.mean(dim=[1, 2, 3])   # (B,)


def compute_all(
    fused: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute all five fusion quality metrics.

    Args:
        fused:     (B, C, H, W) — generator output
        reference: (B, C, H, W) — RGB reference image

    Returns:
        dict with keys: EN, MI, AG, SF, SSIM  — each (B,)
    """
    return {
        "EN":   entropy_score(fused),
        "MI":   mutual_information(fused, reference),
        "AG":   average_gradient(fused),
        "SF":   spatial_frequency(fused),
        "SSIM": ssim(fused, reference),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian_window(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Create a (1, 1, size, size) normalised 2D Gaussian kernel."""
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d = g1d / g1d.sum()
    g2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)
    return g2d.unsqueeze(0).unsqueeze(0)   # (1, 1, size, size)

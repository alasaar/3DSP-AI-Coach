"""
Heatmap target generation and soft-argmax decoding.
"""

import torch

from configs.default import HEATMAP_SIGMA


def make_targets(
    labels: torch.Tensor,
    H: int,
    W: int,
    sigma: float = HEATMAP_SIGMA,
):
    """
    Generate Gaussian heatmap targets from normalised keypoint coords.

    Parameters
    ----------
    labels : (B, K, 3)  – [x_norm, y_norm, visibility]
    H, W   : spatial size of the model's heatmap output

    Returns
    -------
    targets : (B, K, H, W)
    mask    : (B, K, 1, 1)
    """
    xy  = labels[..., :2]
    vis = labels[..., 2]
    B, K = xy.shape[:2]

    ys = torch.arange(H, device=xy.device, dtype=torch.float32).view(1, 1, H, 1)
    xs = torch.arange(W, device=xy.device, dtype=torch.float32).view(1, 1, 1, W)

    cx = xy[..., 0].view(B, K, 1, 1) * (W - 1)
    cy = xy[..., 1].view(B, K, 1, 1) * (H - 1)

    g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    g = g / (g.amax(dim=(2, 3), keepdim=True) + 1e-6)

    mask = (vis > 0).float().view(B, K, 1, 1)
    return g, mask


def soft_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft-argmax decoding.

    Parameters
    ----------
    heatmaps : (B, K, Hh, Ww)

    Returns
    -------
    coords : (B, K, 2)  – normalised [x, y] in [0, 1]
    """
    B, K, Hh, Ww = heatmaps.shape
    prob = torch.softmax(heatmaps.view(B, K, -1) * 100.0, dim=-1)
    prob = prob.view(B, K, Hh, Ww)

    xr = torch.arange(Ww, device=heatmaps.device, dtype=heatmaps.dtype)
    yr = torch.arange(Hh, device=heatmaps.device, dtype=heatmaps.dtype)

    x = (prob.sum(2) * xr).sum(-1) / (Ww - 1)
    y = (prob.sum(3) * yr).sum(-1) / (Hh - 1)

    return torch.stack([x, y], -1)
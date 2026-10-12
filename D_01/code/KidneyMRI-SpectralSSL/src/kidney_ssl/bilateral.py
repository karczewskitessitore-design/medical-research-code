from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


BACKGROUND = 0
LEFT_KIDNEY = 1
RIGHT_KIDNEY = 2


@dataclass(frozen=True)
class BilateralAudit:
    components: int
    dominant_component_fraction: float
    left_fraction: float
    right_fraction: float
    crosses_midline: bool


def split_binary_kidneys(mask: np.ndarray, lr_axis: int = 0) -> np.ndarray:
    """Convert a binary RAS kidney mask to background/left/right labels.

    The first canonical RAS axis is the anatomical left/right axis.  The
    public dataset has two separated kidneys on opposite sides of the volume;
    using the mid-plane is deterministic and remains valid for soft-volume
    calculations where connected components are not differentiable.
    """

    binary = np.asarray(mask).astype(bool)
    labels = np.zeros(binary.shape, dtype=np.uint8)
    midpoint = binary.shape[lr_axis] // 2
    lower = [slice(None)] * binary.ndim
    upper = [slice(None)] * binary.ndim
    lower[lr_axis] = slice(0, midpoint)
    upper[lr_axis] = slice(midpoint, None)
    # In canonical RAS+ coordinates x increases toward anatomical Right.
    # Therefore the lower half is the left kidney and the upper half is the
    # right kidney. This matches the external adapters' class semantics.
    labels[tuple(lower)] = binary[tuple(lower)].astype(np.uint8) * LEFT_KIDNEY
    labels[tuple(upper)] = binary[tuple(upper)].astype(np.uint8) * RIGHT_KIDNEY
    return labels


def audit_bilateral_mask(mask: np.ndarray, lr_axis: int = 0) -> BilateralAudit:
    binary = np.asarray(mask).astype(bool)
    if not binary.any():
        return BilateralAudit(0, 0.0, 0.0, 0.0, False)
    labeled, components = ndimage.label(binary)
    counts = np.bincount(labeled.reshape(-1))[1:]
    dominant = float(np.sort(counts)[-2:].sum() / binary.sum()) if counts.size >= 2 else 1.0
    bilateral = split_binary_kidneys(binary, lr_axis)
    left = int((bilateral == LEFT_KIDNEY).sum())
    right = int((bilateral == RIGHT_KIDNEY).sum())
    total = left + right
    midpoint = binary.shape[lr_axis] // 2
    band = [slice(None)] * binary.ndim
    band[lr_axis] = slice(max(0, midpoint - 1), min(binary.shape[lr_axis], midpoint + 1))
    return BilateralAudit(
        components=int(components),
        dominant_component_fraction=dominant,
        left_fraction=left / total,
        right_fraction=right / total,
        crosses_midline=bool(binary[tuple(band)].any()),
    )


def bilateral_one_hot(labels: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError("Bilateral labels must have shape [B,H,W] or [B,1,H,W]")
    if labels.min() < 0 or labels.max() > RIGHT_KIDNEY:
        raise ValueError("Bilateral labels must contain only 0, 1, and 2")
    return F.one_hot(labels.long(), num_classes=3).permute(0, 3, 1, 2).to(dtype=dtype)


def kidney_union_probability(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim < 3 or probabilities.shape[1] != 3:
        raise ValueError("Expected three-class probabilities [B,3,...]")
    return probabilities[:, LEFT_KIDNEY : RIGHT_KIDNEY + 1].sum(dim=1, keepdim=True)


def swap_bilateral_labels(labels: torch.Tensor) -> torch.Tensor:
    swapped = labels.clone()
    swapped[labels == LEFT_KIDNEY] = RIGHT_KIDNEY
    swapped[labels == RIGHT_KIDNEY] = LEFT_KIDNEY
    return swapped


def bilateral_soft_volumes_ml(
    probabilities: torch.Tensor,
    voxel_volume_ml: float | torch.Tensor,
) -> torch.Tensor:
    """Return [left, right, total] soft volumes for one or more volumes."""

    if probabilities.ndim < 3:
        raise ValueError("Expected [batch, class, ...spatial] probabilities")
    if probabilities.shape[1] != 3:
        raise ValueError("Expected probabilities with background/left/right channels")
    # Selecting a class removes the class axis. Flattening every remaining
    # spatial axis is robust for both 2D batches [B,C,H,W] and complete
    # volumes [B,C,H,W,D]. The former implementation reused pre-selection
    # dimension indices and therefore addressed a non-existent dimension for
    # 3D volumes when DZ-BiMorph first activated.
    left = probabilities[:, LEFT_KIDNEY].flatten(start_dim=1).sum(dim=1)
    right = probabilities[:, RIGHT_KIDNEY].flatten(start_dim=1).sum(dim=1)
    scale = torch.as_tensor(voxel_volume_ml, device=probabilities.device, dtype=probabilities.dtype)
    if scale.ndim:
        scale = scale.reshape(-1)
        if scale.numel() not in {1, probabilities.shape[0]}:
            raise ValueError("voxel_volume_ml must be scalar or have one value per batch item")
    left = left * scale
    right = right * scale
    return torch.stack((left, right, left + right), dim=-1)

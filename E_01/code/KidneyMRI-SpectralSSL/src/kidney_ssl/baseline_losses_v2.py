from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F

from kidney_ssl.bilateral import kidney_union_probability
from kidney_ssl.v2_losses import jensen_shannon_disagreement


def probability_mse(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    student = torch.softmax(student_logits.float(), dim=1)
    teacher = torch.softmax(teacher_logits.detach().float(), dim=1)
    return F.mse_loss(student, teacher)


def cross_head_mutual_loss(head_a: torch.Tensor, head_b: torch.Tensor) -> torch.Tensor:
    probability_a = torch.softmax(head_a.float(), dim=1)
    probability_b = torch.softmax(head_b.float(), dim=1)
    return jensen_shannon_disagreement(probability_a, probability_b).mean()


@lru_cache(maxsize=32)
def _dct_basis_cpu(size: int) -> torch.Tensor:
    coordinate = torch.arange(size, dtype=torch.float32)
    frequency = torch.arange(size, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos(math.pi / size * (coordinate + 0.5) * frequency)
    basis[0] *= math.sqrt(1.0 / size)
    basis[1:] *= math.sqrt(2.0 / size)
    return basis


def dct2(value: torch.Tensor) -> torch.Tensor:
    height, width = value.shape[-2:]
    basis_h = _dct_basis_cpu(height).to(value.device, value.dtype)
    basis_w = _dct_basis_cpu(width).to(value.device, value.dtype)
    return torch.einsum("ih,bchw,jw->bcij", basis_h, value, basis_w)


def _region_similarity(value: torch.Tensor, grid: int) -> torch.Tensor:
    pooled = F.adaptive_avg_pool2d(value, (grid, grid)).flatten(2)
    normalized = F.normalize(pooled, dim=1)
    return torch.einsum("bci,bcj->bij", normalized, normalized)


def frcnet_output_consistency(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    frequency_weight: float = 1.0,
    region_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Clean-room output-level FDC/MRSC approximation.

    The MICCAI paper defines feature-level modules; the public page did not
    provide source code. This implementation is explicitly logged as a
    protocol deviation and must not be called an exact author-code replica.
    """

    student = torch.softmax(student_logits.float(), dim=1)
    teacher = torch.softmax(teacher_logits.detach().float(), dim=1)
    student_foreground = kidney_union_probability(student)
    teacher_foreground = kidney_union_probability(teacher)
    frequency = F.l1_loss(dct2(student_foreground), dct2(teacher_foreground))
    region_losses = [
        F.mse_loss(_region_similarity(student, grid), _region_similarity(teacher, grid))
        for grid in (2, 4, 8)
    ]
    region = torch.stack(region_losses).mean()
    return {
        "total": float(frequency_weight) * frequency + float(region_weight) * region,
        "frequency": frequency,
        "region": region,
    }


def semi_urf_uncertainty_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    progress: float,
    reliable_quantile_start: float = 0.50,
    reliable_quantile_end: float = 0.80,
) -> dict[str, torch.Tensor]:
    """Progressively uses both reliable interiors and uncertain boundaries.

    This follows the UDAT/uncertain-region principle from the paper abstract;
    BUCE and FEFF are represented by uncertainty-weighted probability and Haar
    boundary terms in this clean-room benchmark adapter. The deviation is
    persisted in the benchmark metadata.
    """

    student = torch.softmax(student_logits.float(), dim=1)
    teacher = torch.softmax(teacher_logits.detach().float(), dim=1)
    entropy = -(teacher.clamp_min(1e-6) * teacher.clamp_min(1e-6).log()).sum(
        dim=1, keepdim=True
    ) / math.log(teacher.shape[1])
    target_quantile = reliable_quantile_start + progress * (
        reliable_quantile_end - reliable_quantile_start
    )
    threshold = torch.quantile(entropy.detach().flatten(1), target_quantile, dim=1).view(-1, 1, 1, 1)
    reliable = (entropy <= threshold).float()
    uncertain = 1.0 - reliable
    error = (student - teacher).square().sum(dim=1, keepdim=True)
    reliable_loss = (error * reliable).sum() / reliable.sum().clamp_min(1.0)
    # Haar-like high-frequency response preserves supervision on difficult
    # boundary pixels instead of dropping them.
    kernel_x = student.new_tensor([[1.0, -1.0], [1.0, -1.0]]).view(1, 1, 2, 2) / 2.0
    kernel_y = student.new_tensor([[1.0, 1.0], [-1.0, -1.0]]).view(1, 1, 2, 2) / 2.0
    student_union = kidney_union_probability(student)
    teacher_union = kidney_union_probability(teacher)
    student_high = torch.cat(
        (F.conv2d(student_union, kernel_x), F.conv2d(student_union, kernel_y)), dim=1
    )
    teacher_high = torch.cat(
        (F.conv2d(teacher_union, kernel_x), F.conv2d(teacher_union, kernel_y)), dim=1
    )
    uncertain_down = F.interpolate(uncertain, size=student_high.shape[-2:], mode="nearest")
    boundary_error = (student_high - teacher_high).abs().mean(dim=1, keepdim=True)
    uncertain_loss = (boundary_error * uncertain_down).sum() / uncertain_down.sum().clamp_min(1.0)
    return {
        "total": reliable_loss + 0.2 * uncertain_loss,
        "reliable": reliable_loss,
        "uncertain_boundary": uncertain_loss,
        "entropy": entropy.mean(),
        "threshold": threshold.mean(),
    }


def randdg_frequency_randomize(
    image: torch.Tensor,
    generator: torch.Generator | None = None,
    low_frequency_fraction: float = 0.10,
) -> torch.Tensor:
    """Uniformly perturb low-frequency amplitude while preserving phase."""

    value = image.float()
    spectrum = torch.fft.fftshift(torch.fft.fft2(value, norm="ortho"), dim=(-2, -1))
    amplitude = spectrum.abs()
    phase = torch.angle(spectrum)
    height, width = value.shape[-2:]
    half_h = max(1, int(round(height * low_frequency_fraction / 2)))
    half_w = max(1, int(round(width * low_frequency_fraction / 2)))
    center_h, center_w = height // 2, width // 2
    slices = (..., slice(center_h - half_h, center_h + half_h), slice(center_w - half_w, center_w + half_w))
    randomized = amplitude.clone()
    region = amplitude[slices]
    low = region.amin(dim=(-2, -1), keepdim=True)
    high = region.amax(dim=(-2, -1), keepdim=True)
    uniform = torch.rand(region.shape, device=value.device, generator=generator)
    randomized[slices] = low + uniform * (high - low)
    perturbed = randomized * torch.exp(1j * phase)
    return torch.fft.ifft2(torch.fft.ifftshift(perturbed, dim=(-2, -1)), norm="ortho").real


from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kidney_ssl.bilateral import bilateral_one_hot, kidney_union_probability
from kidney_ssl.spectral import _cpu_band_bank


def multiclass_dice_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    with torch.amp.autocast(logits.device.type, enabled=False):
        value = logits.float()
        target = bilateral_one_hot(labels, dtype=torch.float32)
        probability = torch.softmax(value, dim=1)
        ce = F.cross_entropy(value, labels.long())
        dimensions = (0, 2, 3)
        intersection = (probability[:, 1:] * target[:, 1:]).sum(dim=dimensions)
        denominator = probability[:, 1:].sum(dim=dimensions) + target[:, 1:].sum(dim=dimensions)
        dice = 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()
        union_probability = kidney_union_probability(probability)
        union_target = target[:, 1:].sum(dim=1, keepdim=True)
        union_intersection = (union_probability * union_target).sum()
        union_dice = 1.0 - (2.0 * union_intersection + epsilon) / (
            union_probability.sum() + union_target.sum() + epsilon
        )
        return ce + 0.5 * dice + 0.5 * union_dice


def multiclass_dice(
    logits: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Foreground class Dice for a bilateral softmax segmentation.

    This is the three-class clean-room counterpart used for the Daniel U-Net
    baseline.  Left and right kidneys contribute equally; background is not
    included in the score.
    """
    with torch.amp.autocast(logits.device.type, enabled=False):
        probability = torch.softmax(logits.float(), dim=1)[:, 1:]
        target = bilateral_one_hot(labels, dtype=torch.float32)[:, 1:]
        dimensions = (0, 2, 3)
        intersection = (probability * target).sum(dim=dimensions)
        denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
        return 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def multiclass_dice_plus(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Multiclass Dice++ with powered false-positive/false-negative terms."""
    with torch.amp.autocast(logits.device.type, enabled=False):
        probability = torch.softmax(logits.float(), dim=1)[:, 1:]
        target = bilateral_one_hot(labels, dtype=torch.float32)[:, 1:]
        dimensions = (0, 2, 3)
        true_positive = (probability * target).sum(dim=dimensions)
        false_positive = (probability * (1.0 - target)).pow(gamma).sum(dim=dimensions)
        false_negative = ((1.0 - probability) * target).pow(gamma).sum(dim=dimensions)
        score = (2.0 * true_positive + epsilon) / (
            2.0 * true_positive + false_positive + false_negative + epsilon
        )
        return 1.0 - score.mean()


def multiclass_supervised_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    name: str = "dice_ce",
    dice_plus_gamma: float = 2.0,
) -> torch.Tensor:
    """Dispatch the preregistered bilateral supervised objective."""
    if name == "dice_ce":
        return multiclass_dice_ce(logits, labels)
    if name == "dice":
        return multiclass_dice(logits, labels)
    if name == "dice_plus":
        return multiclass_dice_plus(logits, labels, dice_plus_gamma)
    raise ValueError("Unknown multiclass supervised loss: " + repr(name))


def jensen_shannon_disagreement(
    probability_a: torch.Tensor,
    probability_b: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    with torch.amp.autocast(probability_a.device.type, enabled=False):
        a = probability_a.float().clamp_min(epsilon)
        b = probability_b.float().clamp_min(epsilon)
        middle = 0.5 * (a + b)
        js = 0.5 * (a * (a.log() - middle.log())).sum(dim=1, keepdim=True)
        js = js + 0.5 * (b * (b.log() - middle.log())).sum(dim=1, keepdim=True)
        return (js / math.log(2.0)).clamp(0.0, 1.0)


@dataclass(frozen=True)
class AdaptiveBandResult:
    loss: torch.Tensor
    angle_probabilities: torch.Tensor
    sampled_angles: torch.Tensor
    inclusion_probability: torch.Tensor


class UncertaintyAdaptiveBandLoss(torch.nn.Module):
    def __init__(
        self,
        angles: int = 180,
        width_fraction: float = 0.25,
        uniform_mixture: float = 0.20,
        temperature: float = 0.50,
    ) -> None:
        super().__init__()
        if not 0.0 < uniform_mixture <= 1.0:
            raise ValueError("uniform_mixture must be in (0, 1]")
        self.angles = int(angles)
        self.width_fraction = float(width_fraction)
        self.uniform_mixture = float(uniform_mixture)
        self.temperature = float(temperature)

    def _sampling_distribution(
        self,
        probability_a: torch.Tensor | None,
        probability_b: torch.Tensor | None,
        bank: torch.Tensor,
    ) -> torch.Tensor:
        batch = probability_a.shape[0] if probability_a is not None else 1
        if probability_a is None or probability_b is None:
            return bank.new_full((batch, self.angles), 1.0 / self.angles, dtype=torch.float32)
        with torch.amp.autocast(probability_a.device.type, enabled=False):
            difference = torch.fft.fftshift(
                torch.fft.fft2((probability_a.float() - probability_b.float()), norm="ortho"),
                dim=(-2, -1),
            ).abs().mean(dim=1)
            mask = bank.float()
            scores = torch.einsum("bhw,ahw->ba", difference, mask) / mask.sum(
                dim=(-2, -1)
            ).clamp_min(1.0)
            adaptive = torch.softmax(scores / max(self.temperature, 1e-6), dim=1)
            uniform = torch.full_like(adaptive, 1.0 / self.angles)
            return self.uniform_mixture * uniform + (1.0 - self.uniform_mixture) * adaptive

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        view_probability_a: torch.Tensor | None = None,
        view_probability_b: torch.Tensor | None = None,
        sampled_angles: torch.Tensor | None = None,
    ) -> AdaptiveBandResult:
        batch, _, height, width = logits.shape
        bank_cpu, _ = _cpu_band_bank(height, width, self.angles, self.width_fraction)
        bank = bank_cpu.to(logits.device)
        probabilities = torch.softmax(logits.float(), dim=1)
        angle_probability = self._sampling_distribution(view_probability_a, view_probability_b, bank)
        if angle_probability.shape[0] == 1 and batch > 1:
            angle_probability = angle_probability.expand(batch, -1)
        if sampled_angles is None:
            sampled_angles = torch.multinomial(angle_probability, 1).squeeze(1)
        selected = bank.index_select(0, sampled_angles).unsqueeze(1)
        inclusion = torch.einsum("ba,ahw->bhw", angle_probability, bank.float()).clamp_min(
            self.uniform_mixture / self.angles
        )
        target = bilateral_one_hot(labels, dtype=torch.float32)
        pred_channels = torch.cat((probabilities[:, 1:3], kidney_union_probability(probabilities)), dim=1)
        target_channels = torch.cat((target[:, 1:3], target[:, 1:3].sum(dim=1, keepdim=True)), dim=1)
        with torch.amp.autocast(logits.device.type, enabled=False):
            pred_fft = torch.fft.fftshift(
                torch.fft.fft2(pred_channels.float(), norm="ortho"), dim=(-2, -1)
            )
            target_fft = torch.fft.fftshift(
                torch.fft.fft2(target_channels.float(), norm="ortho"), dim=(-2, -1)
            )
            weighted = selected * (pred_fft - target_fft).abs() / inclusion[:, None]
            loss = weighted.sum(dim=(-2, -1)).mean() / (height * width)
        return AdaptiveBandResult(loss, angle_probability, sampled_angles, inclusion)


def _boundary_band(kidney_probability: torch.Tensor, width: int = 5) -> torch.Tensor:
    hard = (kidney_probability >= 0.5).float()
    dilation = F.max_pool2d(hard, width, stride=1, padding=width // 2)
    erosion = -F.max_pool2d(-hard, width, stride=1, padding=width // 2)
    return (dilation - erosion) > 0


def _sobel(value: torch.Tensor) -> torch.Tensor:
    kernel_x = value.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-2, -1)
    x = F.conv2d(value, kernel_x, padding=1)
    y = F.conv2d(value, kernel_y, padding=1)
    return torch.sqrt(x.square() + y.square() + 1e-8)


def _region_weighted_mean(
    value: torch.Tensor,
    reliability: torch.Tensor,
    regions: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    outputs = []
    for region in regions:
        weight = reliability * region.float()
        denominator = weight.sum()
        if bool(denominator > 0):
            outputs.append((value * weight).sum() / denominator)
    return torch.stack(outputs).mean() if outputs else value.new_zeros(())


def spectral_disagreement_teacher_loss(
    student_logits_a: torch.Tensor,
    student_logits_b: torch.Tensor,
    teacher_logits: torch.Tensor,
    boundary_weight: float = 0.2,
    boundary_width: int = 5,
) -> dict[str, torch.Tensor]:
    with torch.amp.autocast(student_logits_a.device.type, enabled=False):
        student_a = torch.softmax(student_logits_a.float(), dim=1)
        student_b = torch.softmax(student_logits_b.float(), dim=1)
        teacher = torch.softmax(teacher_logits.detach().float(), dim=1)
        disagreement = jensen_shannon_disagreement(student_a, student_b)
        entropy = -(teacher.clamp_min(1e-6) * teacher.clamp_min(1e-6).log()).sum(
            dim=1, keepdim=True
        ) / math.log(teacher.shape[1])
        reliability = ((1.0 - disagreement) * (1.0 - entropy)).clamp(0.0, 1.0)
        teacher_kidney = kidney_union_probability(teacher)
        boundary = _boundary_band(teacher_kidney, boundary_width)
        foreground = (teacher_kidney >= 0.5) & ~boundary
        background = (teacher_kidney < 0.5) & ~boundary
        mean_student = 0.5 * (student_a + student_b)
        probability_error = (mean_student - teacher).square().sum(dim=1, keepdim=True)
        probability_loss = _region_weighted_mean(
            probability_error, reliability, (foreground, background, boundary)
        )
        student_edge = _sobel(kidney_union_probability(mean_student))
        teacher_edge = _sobel(teacher_kidney)
        boundary_reliability = (0.1 + 0.9 * reliability) * boundary.float()
        boundary_loss = ((student_edge - teacher_edge).abs() * boundary_reliability).sum() / (
            boundary_reliability.sum().clamp_min(1.0)
        )
        total = probability_loss + float(boundary_weight) * boundary_loss
        return {
            "total": total,
            "probability": probability_loss,
            "boundary": boundary_loss,
            "disagreement": disagreement.mean(),
            "teacher_entropy": entropy.mean(),
        }


def dead_zone_bilateral_morphometry_loss(
    student_volumes_ml: torch.Tensor,
    teacher_volumes_ml: torch.Tensor | None = None,
    thresholds: tuple[float, float, float] = (0.05, 0.05, 0.04),
    anchor_weight: float = 0.1,
    epsilon: float = 1e-3,
) -> dict[str, torch.Tensor]:
    if student_volumes_ml.ndim != 2 or student_volumes_ml.shape[1] != 3:
        raise ValueError("Expected acquisition-by-[left,right,total] volume matrix")
    value = student_volumes_ml.float()
    mean = value.mean(dim=0).clamp_min(epsilon)
    cv = value.std(dim=0, unbiased=True) / mean
    limit = value.new_tensor(thresholds)
    repeatability = F.relu(cv - limit).square().mean()
    anchor = value.new_zeros(())
    if teacher_volumes_ml is not None:
        teacher_mean = teacher_volumes_ml.detach().float().mean(dim=0).clamp_min(epsilon)
        anchor = F.smooth_l1_loss(torch.log(mean), torch.log(teacher_mean))
    return {
        "total": repeatability + float(anchor_weight) * anchor,
        "repeatability": repeatability,
        "anchor": anchor,
        "cv_left": cv[0],
        "cv_right": cv[1],
        "cv_total": cv[2],
    }


def one_sided_pcgrad(
    main_gradients: list[torch.Tensor | None],
    auxiliary_gradients: list[torch.Tensor | None],
    epsilon: float = 1e-12,
) -> list[torch.Tensor | None]:
    pairs = [
        (main, auxiliary)
        for main, auxiliary in zip(main_gradients, auxiliary_gradients)
        if main is not None and auxiliary is not None
    ]
    if not pairs:
        return auxiliary_gradients
    dot = sum((main * auxiliary).sum() for main, auxiliary in pairs)
    norm = sum(main.square().sum() for main, _ in pairs).clamp_min(epsilon)
    if bool(dot >= 0):
        return auxiliary_gradients
    scale = dot / norm
    return [
        None if auxiliary is None else auxiliary - scale * main if main is not None else auxiliary
        for main, auxiliary in zip(main_gradients, auxiliary_gradients)
    ]

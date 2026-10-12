from __future__ import annotations

import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss

from kidney_ssl.model import logits_heads


def supervised_dice_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    return loss(logits, target)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dimensions = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    return (1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def dice_plus_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Dice++ from Yeung et al., used by the Damiano et al. baseline."""
    probability = torch.sigmoid(logits)
    dimensions = tuple(range(1, probability.ndim))
    true_positive = (probability * target).sum(dim=dimensions)
    false_positive = (probability * (1.0 - target)).pow(gamma).sum(dim=dimensions)
    false_negative = ((1.0 - probability) * target).pow(gamma).sum(dim=dimensions)
    score = (2.0 * true_positive + epsilon) / (
        2.0 * true_positive + false_positive + false_negative + epsilon
    )
    return (1.0 - score).mean()


def supervised_loss(
    output: torch.Tensor | tuple[torch.Tensor, ...],
    target: torch.Tensor,
    name: str = "dice_ce",
    gamma: float = 2.0,
) -> torch.Tensor:
    functions = {
        "dice_ce": supervised_dice_ce,
        "dice": dice_loss,
        "dice_plus": lambda logits, truth: dice_plus_loss(logits, truth, gamma),
        "bce": F.binary_cross_entropy_with_logits,
    }
    try:
        function = functions[name]
    except KeyError as error:
        raise ValueError(f"Unknown supervised loss {name!r}; available: {tuple(functions)}") from error
    return torch.stack([function(head, target) for head in logits_heads(output)]).mean()


def teacher_consistency_loss(
    student_logits: tuple[torch.Tensor, ...],
    teacher_logits: torch.Tensor,
    threshold: float = 0.8,
) -> torch.Tensor:
    teacher_probability = torch.sigmoid(teacher_logits).detach()
    confidence = torch.maximum(teacher_probability, 1.0 - teacher_probability)
    valid = confidence >= threshold
    if not bool(valid.any()):
        return teacher_probability.new_zeros(())
    losses = []
    for logits in student_logits:
        probability = torch.sigmoid(logits)
        losses.append(F.mse_loss(probability[valid], teacher_probability[valid]))
    return torch.stack(losses).mean()


def cross_head_consistency_loss(
    student_logits: tuple[torch.Tensor, ...],
    teacher_logits: tuple[torch.Tensor, ...],
    threshold: float = 0.8,
) -> torch.Tensor:
    if len(student_logits) != 2 or len(teacher_logits) != 2:
        raise ValueError("Cross-head consistency requires exactly two student and teacher heads")
    losses = []
    for student, teacher in ((student_logits[0], teacher_logits[1]), (student_logits[1], teacher_logits[0])):
        probability = torch.sigmoid(teacher).detach()
        confidence = torch.maximum(probability, 1.0 - probability)
        valid = confidence >= threshold
        if bool(valid.any()):
            losses.append(F.binary_cross_entropy_with_logits(student[valid], (probability[valid] >= 0.5).float()))
    if not losses:
        return student_logits[0].new_zeros(())
    return torch.stack(losses).mean()


def soft_tkv_ml(probability_volume: torch.Tensor, spacing_mm: tuple[float, float, float]) -> torch.Tensor:
    voxel_ml = float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2]) / 1000.0
    return probability_volume.sum() * voxel_ml


def repeat_tkv_loss(volume_a_ml: torch.Tensor, volume_b_ml: torch.Tensor, epsilon_ml: float = 0.001) -> torch.Tensor:
    difference = torch.log(volume_a_ml + epsilon_ml) - torch.log(volume_b_ml + epsilon_ml)
    return F.smooth_l1_loss(difference, torch.zeros_like(difference))

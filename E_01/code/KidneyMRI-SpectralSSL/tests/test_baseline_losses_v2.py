from __future__ import annotations

import torch

from kidney_ssl.baseline_losses_v2 import (
    cross_head_mutual_loss,
    dct2,
    frcnet_output_consistency,
    randdg_frequency_randomize,
    semi_urf_uncertainty_loss,
)


def test_dct_and_frcnet_consistency_are_finite_and_zero_for_equal_logits() -> None:
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    assert dct2(logits).shape == logits.shape
    result = frcnet_output_consistency(logits, logits.detach())
    assert result["total"].item() < 1e-6
    result["total"].backward()
    assert torch.isfinite(logits.grad).all()


def test_semi_urf_uses_uncertain_boundary_without_nan() -> None:
    student = torch.randn(2, 3, 16, 16, requires_grad=True)
    teacher = torch.randn(2, 3, 16, 16)
    result = semi_urf_uncertainty_loss(student, teacher, 0.5)
    assert torch.isfinite(result["total"])
    result["total"].backward()
    assert torch.isfinite(student.grad).all()


def test_randdg_shape_real_and_cross_head_identity() -> None:
    image = torch.randn(2, 3, 17, 19)
    result = randdg_frequency_randomize(image, torch.Generator().manual_seed(4))
    assert result.shape == image.shape
    assert torch.isfinite(result).all()
    logits = torch.randn(2, 3, 8, 8)
    assert cross_head_mutual_loss(logits, logits).item() < 1e-6


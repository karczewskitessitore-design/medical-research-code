from __future__ import annotations

import pytest
import torch

from kidney_ssl.bilateral import bilateral_one_hot, kidney_union_probability
from kidney_ssl.spectral import _cpu_band_bank

from kidney_ssl.v2_losses import (
    UncertaintyAdaptiveBandLoss,
    dead_zone_bilateral_morphometry_loss,
    jensen_shannon_disagreement,
    multiclass_dice,
    multiclass_dice_ce,
    multiclass_dice_plus,
    multiclass_supervised_loss,
    one_sided_pcgrad,
    spectral_disagreement_teacher_loss,
)


def _example() -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.zeros((2, 32, 32), dtype=torch.long)
    labels[:, 4:12, 8:24] = 1
    labels[:, 20:28, 8:24] = 2
    logits = torch.randn((2, 3, 32, 32), requires_grad=True)
    return logits, labels


def test_multiclass_loss_has_finite_gradient() -> None:
    logits, labels = _example()
    loss = multiclass_dice_ce(logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("name", ["dice_ce", "dice", "dice_plus"])
def test_multiclass_supervised_registry_has_finite_gradient(name: str) -> None:
    logits, labels = _example()
    loss = multiclass_supervised_loss(logits, labels, name)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_dice_and_dice_plus_reward_a_near_perfect_prediction() -> None:
    _, labels = _example()
    logits = torch.full((2, 3, 32, 32), -12.0)
    logits.scatter_(1, labels[:, None], 12.0)
    assert multiclass_dice(logits, labels) < 1e-4
    assert multiclass_dice_plus(logits, labels) < 1e-4


def test_multiclass_supervised_registry_rejects_unknown_loss() -> None:
    logits, labels = _example()
    with pytest.raises(ValueError, match="Unknown multiclass supervised loss"):
        multiclass_supervised_loss(logits, labels, "not-a-loss")


def test_disagreement_zero_for_identical_probabilities() -> None:
    probability = torch.softmax(torch.randn((2, 3, 16, 16)), dim=1)
    disagreement = jensen_shannon_disagreement(probability, probability)
    assert torch.allclose(disagreement, torch.zeros_like(disagreement), atol=1e-6)


def test_adaptive_band_probabilities_and_gradient() -> None:
    logits, labels = _example()
    probability_a = torch.softmax(torch.randn_like(logits), dim=1)
    probability_b = torch.softmax(torch.randn_like(logits), dim=1)
    loss = UncertaintyAdaptiveBandLoss(angles=18)(
        logits, labels, probability_a, probability_b
    )
    assert torch.allclose(loss.angle_probabilities.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.all(loss.inclusion_probability > 0)
    loss.loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_adaptive_band_horvitz_thompson_monte_carlo_within_three_percent() -> None:
    torch.manual_seed(19)
    logits, labels = _example()
    probability_a = torch.softmax(torch.randn_like(logits), dim=1)
    probability_b = torch.softmax(torch.randn_like(logits), dim=1)
    module = UncertaintyAdaptiveBandLoss(angles=18)
    probe = module(
        logits,
        labels,
        probability_a,
        probability_b,
        sampled_angles=torch.zeros(2, dtype=torch.long),
    )
    probabilities = torch.softmax(logits.detach(), dim=1)
    target = bilateral_one_hot(labels)
    pred_channels = torch.cat((probabilities[:, 1:3], kidney_union_probability(probabilities)), dim=1)
    target_channels = torch.cat((target[:, 1:3], target[:, 1:3].sum(dim=1, keepdim=True)), dim=1)
    difference = (
        torch.fft.fftshift(
            torch.fft.fft2(pred_channels, norm="ortho"), dim=(-2, -1)
        )
        - torch.fft.fftshift(
            torch.fft.fft2(target_channels, norm="ortho"), dim=(-2, -1)
        )
    ).abs()
    full = difference.sum(dim=(-2, -1)).mean() / (32 * 32)
    bank, _ = _cpu_band_bank(32, 32, 18, 0.25)
    generator = torch.Generator().manual_seed(31)
    estimates = []
    for _ in range(3000):
        sampled = torch.multinomial(probe.angle_probabilities, 1, generator=generator).squeeze(1)
        selected = bank.index_select(0, sampled).unsqueeze(1)
        estimate = (selected * difference / probe.inclusion_probability[:, None]).sum(
            dim=(-2, -1)
        ).mean() / (32 * 32)
        estimates.append(estimate)
    monte_carlo = torch.stack(estimates).mean()
    relative_error = abs(float(monte_carlo - full)) / max(abs(float(full)), 1e-8)
    assert relative_error < 0.03


def test_csd_loss_regions_and_gradient() -> None:
    a, _ = _example()
    b = torch.randn_like(a, requires_grad=True)
    teacher = torch.randn_like(a)
    values = spectral_disagreement_teacher_loss(a, b, teacher)
    values["total"].backward()
    assert torch.isfinite(values["total"])
    assert a.grad is not None and b.grad is not None


def test_dead_zone_morphometry_loss() -> None:
    stable = torch.tensor(
        [[100.0, 110.0, 210.0], [101.0, 109.0, 210.0], [99.0, 111.0, 210.0]]
    )
    unstable = stable.clone()
    unstable[2] = torch.tensor([70.0, 150.0, 220.0])
    assert dead_zone_bilateral_morphometry_loss(stable)["repeatability"] == 0
    assert dead_zone_bilateral_morphometry_loss(unstable)["repeatability"] > 0
    reversed_loss = dead_zone_bilateral_morphometry_loss(torch.flip(unstable, dims=(0,)))[
        "repeatability"
    ]
    assert torch.allclose(
        dead_zone_bilateral_morphometry_loss(unstable)["repeatability"], reversed_loss
    )


def test_one_sided_pcgrad_removes_conflict() -> None:
    main = [torch.tensor([1.0, 0.0])]
    auxiliary = [torch.tensor([-1.0, 1.0])]
    projected = one_sided_pcgrad(main, auxiliary)
    assert projected[0] is not None
    assert torch.dot(main[0], projected[0]) >= -1e-7
    aligned = [torch.tensor([1.0, 1.0])]
    assert torch.equal(one_sided_pcgrad(main, aligned)[0], aligned[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_extreme_logits_have_finite_cuda_losses(dtype: torch.dtype) -> None:
    logits = torch.empty((2, 3, 16, 16), device="cuda", dtype=dtype).uniform_(-10000, 10000)
    logits.requires_grad_(True)
    labels = torch.randint(0, 3, (2, 16, 16), device="cuda")
    probabilities = torch.softmax(logits.float(), dim=1)
    loss = multiclass_dice_ce(logits, labels)
    csd = spectral_disagreement_teacher_loss(logits, -logits, logits.detach())
    band = UncertaintyAdaptiveBandLoss(18, 0.25, 0.2, 0.5)(
        logits, labels, probabilities, probabilities
    ).loss
    total = loss + csd["total"] + band
    assert torch.isfinite(total)
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

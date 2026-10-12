from __future__ import annotations

import torch

from kidney_ssl.spectral import (
    RandomBandLoss,
    complementary_fourier_masks,
    fourier_view,
    is_hermitian_mask,
)


def test_random_band_enumerated_expectation_matches_full_loss():
    generator = torch.Generator().manual_seed(7)
    prediction = torch.rand((1, 1, 32, 32), generator=generator, requires_grad=True)
    target = (torch.rand((1, 1, 32, 32), generator=generator) > 0.5).float()
    loss = RandomBandLoss(angles=180, width_fraction=0.25)
    estimates = [loss(prediction, target, torch.tensor([angle])) for angle in range(180)]
    enumerated = torch.stack(estimates).mean()
    full = loss.full_spectrum_loss(prediction, target)
    relative = torch.abs(enumerated - full) / full.clamp_min(1e-8)
    assert relative.item() < 0.03
    enumerated.backward()
    assert torch.isfinite(prediction.grad).all()


def test_complementary_masks_are_hermitian_shared_center_and_disjoint_elsewhere():
    theta, lam, center = complementary_fourier_masks(
        4, 64, 64, center_fraction=0.08, base_keep_rate=0.70, radial_power=4
    )
    assert is_hermitian_mask(theta)
    assert is_hermitian_mask(lam)
    assert theta[:, 0, center].all()
    assert lam[:, 0, center].all()
    assert not (theta[:, 0, ~center] & lam[:, 0, ~center]).any()
    union_keep_rate = (theta[:, 0, ~center] | lam[:, 0, ~center]).float().mean().item()
    assert abs(union_keep_rate - 0.70) < 0.03
    image = torch.randn(4, 3, 64, 64)
    view = fourier_view(image, theta)
    assert view.shape == image.shape
    assert not torch.is_complex(view)
    assert torch.isfinite(view).all()

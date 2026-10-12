from __future__ import annotations

import numpy as np
import torch

from kidney_ssl.augmentation import AugmentationConfig, augment_bilateral_batch
from kidney_ssl.bilateral import (
    LEFT_KIDNEY,
    RIGHT_KIDNEY,
    audit_bilateral_mask,
    bilateral_one_hot,
    bilateral_soft_volumes_ml,
    split_binary_kidneys,
)


def test_split_binary_kidneys_and_audit() -> None:
    mask = np.zeros((32, 20, 5), dtype=np.uint8)
    mask[3:10, 6:14, 1:4] = 1
    mask[22:29, 6:14, 1:4] = 1
    labels = split_binary_kidneys(mask)
    assert set(np.unique(labels)) == {0, LEFT_KIDNEY, RIGHT_KIDNEY}
    audit = audit_bilateral_mask(mask)
    assert audit.components == 2
    assert audit.dominant_component_fraction == 1.0
    assert not audit.crosses_midline
    assert 0.49 < audit.left_fraction < 0.51
    assert np.all(labels[3:10, 6:14, 1:4] == LEFT_KIDNEY)
    assert np.all(labels[22:29, 6:14, 1:4] == RIGHT_KIDNEY)


def test_bilateral_one_hot() -> None:
    labels = torch.tensor([[[0, 1], [2, 0]]])
    result = bilateral_one_hot(labels)
    assert result.shape == (1, 3, 2, 2)
    assert torch.allclose(result.sum(dim=1), torch.ones((1, 2, 2)))


def test_bilateral_soft_volumes_support_2d_and_3d_spatial_tensors() -> None:
    two_dimensional = torch.zeros((2, 3, 4, 5), requires_grad=True)
    two_dimensional.data[:, LEFT_KIDNEY] = 0.25
    two_dimensional.data[:, RIGHT_KIDNEY] = 0.50
    volume_2d = bilateral_soft_volumes_ml(two_dimensional, 0.1)
    assert volume_2d.shape == (2, 3)
    assert torch.allclose(volume_2d[0], torch.tensor([0.5, 1.0, 1.5]))

    three_dimensional = torch.zeros((2, 3, 4, 5, 6), requires_grad=True)
    three_dimensional.data[:, LEFT_KIDNEY] = 0.25
    three_dimensional.data[:, RIGHT_KIDNEY] = 0.50
    volume_3d = bilateral_soft_volumes_ml(
        three_dimensional, torch.tensor([0.1, 0.2])
    )
    assert volume_3d.shape == (2, 3)
    assert torch.allclose(volume_3d[0], torch.tensor([3.0, 6.0, 9.0]))
    assert torch.allclose(volume_3d[1], torch.tensor([6.0, 12.0, 18.0]))
    volume_3d.sum().backward()
    assert three_dimensional.grad is not None
    assert torch.isfinite(three_dimensional.grad).all()


def test_bilateral_soft_volumes_reject_invalid_batch_scaling() -> None:
    with torch.no_grad():
        probabilities = torch.zeros((2, 3, 4, 5, 6))
        with np.testing.assert_raises(ValueError):
            bilateral_soft_volumes_ml(probabilities, torch.ones(3))


def test_lr_flip_swaps_kidney_identity() -> None:
    image = torch.zeros((1, 3, 8, 8))
    labels = torch.zeros((1, 8, 8), dtype=torch.long)
    labels[:, 1:3, 2:6] = LEFT_KIDNEY
    config = AugmentationConfig(
        lr_flip_probability=1.0,
        rotation_degrees=0.0,
        scale_min=1.0,
        scale_max=1.0,
        translation_fraction=0.0,
        gamma_min=1.0,
        gamma_max=1.0,
        contrast_min=1.0,
        contrast_max=1.0,
        bias_strength=0.0,
        noise_std=0.0,
    )
    _, transformed = augment_bilateral_batch(image, labels, config)
    assert transformed is not None
    assert torch.all(transformed[:, 5:7, 2:6] == RIGHT_KIDNEY)
    assert not torch.any(transformed == LEFT_KIDNEY)

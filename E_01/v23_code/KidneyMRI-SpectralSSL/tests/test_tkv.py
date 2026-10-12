from __future__ import annotations

import torch

from kidney_ssl.losses import repeat_tkv_loss, soft_tkv_ml


def test_tkv_physical_units_and_loss_symmetry():
    probability = torch.ones(2, 3, 4)
    volume = soft_tkv_ml(probability, (2.0, 2.0, 5.0))
    assert torch.isclose(volume, torch.tensor(0.48))
    same = repeat_tkv_loss(volume, volume)
    perturbed_ab = repeat_tkv_loss(volume, volume * 1.2)
    perturbed_ba = repeat_tkv_loss(volume * 1.2, volume)
    assert same.item() == 0.0
    assert perturbed_ab.item() > 0.0
    assert torch.allclose(perturbed_ab, perturbed_ba)


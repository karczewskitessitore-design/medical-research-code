from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from kidney_ssl.evaluation_v2 import _predict, _restore_model_volume


class _DropoutProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout2d(0.5)
        self.output = torch.nn.Conv2d(3, 3, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(value))


def test_restore_probability_keeps_classes_last_for_each_slice_axis() -> None:
    probability = torch.zeros(3, 4, 5, 6)
    probability[1] = 1.0
    for axis, canonical_shape in ((0, (6, 4, 5)), (1, (4, 6, 5)), (2, (4, 5, 6))):
        volume = SimpleNamespace(model_original_shape=(4, 5, 6), ras_slice_axis=axis)
        restored = _restore_model_volume(probability, volume, "bilinear")
        assert restored.shape == (*canonical_shape, 3)
        assert np.allclose(restored[..., 1], 1.0)


def test_restore_scalar_volume_for_each_slice_axis() -> None:
    scalar = torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(4, 5, 6)
    for axis, canonical_shape in ((0, (6, 4, 5)), (1, (4, 6, 5)), (2, (4, 5, 6))):
        volume = SimpleNamespace(model_original_shape=(4, 5, 6), ras_slice_axis=axis)
        restored = _restore_model_volume(scalar, volume, "bilinear")
        assert restored.shape == canonical_shape


def test_mcd_and_intensity_tta_prediction_modes_are_finite() -> None:
    torch.manual_seed(4)
    model = _DropoutProbe().eval()
    slices = torch.randn(4, 3, 12, 10)
    for mode in ("deterministic", "mcd", "tta"):
        probability, uncertainty = _predict(
            model,
            slices,
            torch.device("cpu"),
            2,
            "fp32",
            {"enabled": False},
            mode,
            4,
        )
        assert probability.shape == (3, 12, 10, 4)
        assert uncertainty.shape == (12, 10, 4)
        assert torch.isfinite(probability).all()
        assert torch.isfinite(uncertainty).all()
        assert (uncertainty >= 0).all()

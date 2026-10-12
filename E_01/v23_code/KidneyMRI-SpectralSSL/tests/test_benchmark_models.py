from __future__ import annotations

from pathlib import Path

import pytest
import torch

from kidney_ssl.losses import supervised_loss
from kidney_ssl.model import build_model, create_teacher, logits_heads, primary_logits
from kidney_ssl.training import _cmmt_consistency, _tcsm_consistency


NATIVE_MODELS = (
    "monai_unet",
    "daniel_unet",
    "damiano_unet",
    "unetpp",
    "tcsm_unet",
    "cmmt_net",
)


def _config(name: str) -> dict:
    return {
        "name": name,
        "channels": [4, 8, 16],
        "residual_units": 1,
        "in_channels": 3,
        "out_channels": 1,
        "options": {
            "dropout": 0.1 if name in {"damiano_unet", "cmmt_net"} else 0.0,
            "deep_supervision": True,
        },
    }


@pytest.mark.parametrize("name", NATIVE_MODELS)
def test_registered_model_shape_gradient_and_checkpoint(name: str, tmp_path: Path):
    config = _config(name)
    model = build_model(config)
    image = torch.randn(2, 3, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    output = model(image)
    assert primary_logits(output).shape == target.shape
    expected_heads = 2 if name == "cmmt_net" else (2 if name == "unetpp" else 1)
    assert len(logits_heads(output)) == expected_heads
    loss = supervised_loss(output, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    checkpoint = tmp_path / f"{name}.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = build_model(config)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("name", NATIVE_MODELS)
def test_registered_model_cuda_amp(name: str):
    model = build_model(_config(name)).cuda()
    image = torch.randn(1, 3, 32, 32, device="cuda")
    target = torch.rand(1, 1, 32, 32, device="cuda")
    with torch.amp.autocast("cuda"):
        loss = supervised_loss(model(image), target)
    loss.backward()
    assert torch.isfinite(loss)


def test_semisupervised_baseline_losses_are_finite_and_differentiable():
    image = torch.randn(2, 3, 32, 32)
    tcsm = build_model(_config("tcsm_unet"))
    tcsm_loss = _tcsm_consistency(tcsm, create_teacher(tcsm), image, 0.5)
    tcsm_loss.backward()
    assert torch.isfinite(tcsm_loss)

    cmmt = build_model(_config("cmmt_net"))
    cmmt_loss = _cmmt_consistency(
        cmmt,
        create_teacher(cmmt),
        image,
        threshold=0.5,
        noise_std=0.1,
        cutmix_probability=1.0,
        vat_xi=0.001,
        vat_epsilon=1.0,
        vat_weight=0.1,
    )
    cmmt_loss.backward()
    assert torch.isfinite(cmmt_loss)

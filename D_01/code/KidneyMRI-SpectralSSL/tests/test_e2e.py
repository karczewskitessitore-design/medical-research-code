from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from kidney_ssl.evaluation import evaluate
from kidney_ssl.model import build_model
from kidney_ssl.training import train


def _write_case(root: Path, subject: str, acquisition: str):
    image_path = root / f"{subject}_T2W_{acquisition}.nii.gz"
    mask_path = root / f"{subject}_T2W_{acquisition}_mask.nii.gz"
    rng = np.random.default_rng(abs(hash(subject + acquisition)) % (2**32))
    image = rng.normal(size=(16, 16, 3)).astype(np.float32)
    mask = np.zeros((16, 16, 3), dtype=np.uint8)
    mask[4:12, 5:11] = 1
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    return {
        "subject_id": subject,
        "cohort": subject.split("_")[0],
        "acquisition_id": acquisition,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "shape": "16x16x3",
        "spacing": "1.0x1.0x1.0",
        "is_repeat": False,
    }


def test_one_cpu_epoch_checkpoint_prediction_and_metrics(tmp_path: Path):
    data = tmp_path / "data"
    split = data / "splits"
    split.mkdir(parents=True)
    rows = [_write_case(data, "HC_01", "1"), _write_case(data, "CKD_01", "1")]
    manifest = pd.DataFrame(rows)
    manifest["test_fold"] = [1, 0]
    manifest.to_csv(split / "manifest_with_folds.csv", index=False)
    pd.DataFrame(
        [{"subject_id": "HC_01", "labeled_fraction": 0.25}]
    ).to_csv(split / "labeled_fold0.csv", index=False)
    config = {
        "data": {"root": str(data), "manifest": "unused", "folds": str(split), "image_size": [16, 16], "clip_percentiles": [0.5, 99.5]},
        "split": {"labeled_fraction": 0.25, "fold": 0, "seed": 7},
        "model": {"channels": [4, 8], "residual_units": 1, "in_channels": 3, "out_channels": 1},
        "spectral_band": {"enabled": True, "angles": 18, "width_fraction": 0.25, "weight": 0.05},
        "mean_teacher": {"enabled": False, "center_fraction": 0.08, "base_keep_rate": 0.7, "radial_power": 4, "confidence_threshold": 0.8, "ema_decay": 0.99, "ramp_epochs": 1, "weight": 1.0},
        "repeat_tkv": {"enabled": False, "weight": 0.02, "epsilon_ml": 0.001, "ramp_epochs": 1, "steps_per_epoch": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.00001, "epochs": 1, "batch_size": 2, "gradient_clip": 1.0},
        "runtime": {"output_root": str(tmp_path / "experiments"), "run_name": "E2E", "amp": False, "num_workers": 0, "heartbeat_seconds": 1, "stall_minutes": 20, "timeout_hours": 1, "device": "cpu", "cache_volumes": True},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = train(config_path, fold=0, seed=7)
    assert (run_dir / "last_student.pt").exists()
    assert (run_dir / "last_teacher.pt").exists()
    summary = evaluate(run_dir)
    assert summary.exists()
    assert (run_dir / "metrics_acquisition.csv").exists()
    assert list((run_dir / "predictions").glob("*.nii.gz"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_amp_forward_backward():
    model = build_model({"channels": [4, 8], "residual_units": 1, "in_channels": 3, "out_channels": 1}).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    image = torch.randn(2, 3, 32, 32, device="cuda")
    target = torch.rand(2, 1, 32, 32, device="cuda")
    scaler = torch.amp.GradScaler("cuda")
    with torch.amp.autocast("cuda"):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(image), target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    assert torch.isfinite(loss)


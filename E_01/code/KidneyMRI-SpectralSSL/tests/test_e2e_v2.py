from __future__ import annotations

from pathlib import Path
import json

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from kidney_ssl.evaluation_v2 import evaluate_v2
from kidney_ssl.training_v2 import train_v2


def _case(root: Path, subject: str, fold: int, acquisition: str = "A1") -> dict:
    image_path = root / f"{subject}_{acquisition}.nii.gz"
    mask_path = root / f"{subject}_{acquisition}_mask.nii.gz"
    image = np.random.default_rng(fold).normal(size=(16, 16, 3)).astype(np.float32)
    mask = np.zeros((16, 16, 3), dtype=np.uint8)
    mask[2:6, 5:10, :] = 1
    mask[10:14, 5:10, :] = 1
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    return {
        "subject_id": subject,
        "cohort": "HC" if subject.startswith("HC") else "CKD",
        "acquisition_id": acquisition,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "shape": "16x16x3",
        "spacing": "1x1x1",
        "is_repeat": False,
        "test_fold": fold,
    }


def test_v2_cpu_epoch_is_resumable_and_evaluates_bilateral_metrics(tmp_path: Path) -> None:
    data = tmp_path / "data"
    split = data / "splits"
    split.mkdir(parents=True)
    training_cases = [_case(data, "HC_01", 1, f"A{index}") for index in range(1, 6)]
    pd.DataFrame([*training_cases, _case(data, "CKD_01", 0)]).to_csv(
        split / "manifest_with_folds.csv", index=False
    )
    pd.DataFrame([{"subject_id": "HC_01", "labeled_fraction": 0.25}]).to_csv(
        split / "labeled_fold0.csv", index=False
    )
    config = {
        "schema_version": 2,
        "data": {
            "root": str(data),
            "manifest": "unused",
            "folds": str(split),
            "image_size": [16, 16],
            "clip_percentiles": [0.5, 99.5],
        },
        "split": {"labeled_fraction": 0.25, "fold": 0, "seed": 7},
        "model": {
            "name": "monai_unet",
            "trainer": "native_v2",
            "channels": [4, 8],
            "residual_units": 1,
            "in_channels": 3,
            "out_channels": 3,
        },
        "bilateral_labels": {"enabled": True},
        "augmentation": {
            "enabled": True,
            "lr_flip_probability": 0.5,
            "rotation_degrees": 5,
            "scale_min": 0.95,
            "scale_max": 1.05,
            "translation_fraction": 0.05,
            "gamma_min": 0.9,
            "gamma_max": 1.1,
            "contrast_min": 0.9,
            "contrast_max": 1.1,
            "bias_strength": 0.05,
            "noise_std": 0.01,
        },
        "adaptive_band": {
            "enabled": True,
            "angles": 18,
            "width_fraction": 0.25,
            "uniform_mixture": 0.2,
            "temperature": 0.5,
            "weight": 0.05,
        },
        "spectral_disagreement": {
            "enabled": True,
            "center_fraction": 0.08,
            "base_keep_rate": 0.7,
            "radial_power": 4,
            "boundary_width": 3,
            "boundary_weight": 0.2,
            "weight": 1.0,
            "ema_decay": 0.99,
            "ramp_epochs": 1,
        },
        "repeat_morphometry": {
            "enabled": True,
            "start_epoch": 1,
            "cv_thresholds": [0.05, 0.05, 0.04],
            "anchor_weight": 0.1,
            "weight": 0.01,
        },
        "baseline_ssl": {"enabled": False},
        "external_validation": {"manifests": {}},
        "spectral_band": {"enabled": False},
        "mean_teacher": {"enabled": False},
        "repeat_tkv": {"enabled": False},
        "optimizer": {
            "learning_rate": 3e-4,
            "minimum_learning_rate": 1e-6,
            "weight_decay": 1e-5,
            "warmup_epochs": 1,
            "epochs": 1,
            "batch_size": 2,
            "gradient_clip": 1.0,
        },
        "runtime": {
            "output_root": str(tmp_path / "experiments"),
            "run_name": "V2_E2E",
            "precision": "fp32",
            "num_workers": 0,
            "heartbeat_seconds": 1,
            "device": "cpu",
            "cache_volumes": True,
            "resume": False,
        },
    }
    config_path = tmp_path / "v2.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run = train_v2(config_path, 0, 7)
    assert (run / "last_checkpoint.pt").is_file()
    assert (run / "last_teacher.pt").is_file()
    first_epoch = json.loads((run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first_epoch["repeat_update_applied"] is True
    assert first_epoch["repeat_subject"] == "HC_01"
    assert evaluate_v2(run).is_file()
    metrics = pd.read_csv(run / "metrics_subject.csv")
    assert {"dice", "dice_left", "dice_right", "surface_dice_2mm"}.issubset(metrics)
    config["optimizer"]["epochs"] = 2
    config["runtime"]["resume"] = True
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    resumed = train_v2(config_path, 0, 7)
    assert resumed == run
    assert len((run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2

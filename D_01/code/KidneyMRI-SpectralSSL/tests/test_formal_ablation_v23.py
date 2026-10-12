from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import yaml

from kidney_ssl.config import load_config
from kidney_ssl.evaluation_v2 import evaluate_v2
from kidney_ssl.training_v2 import train_v2


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_v22_defaults_remain_bilateral_and_regional() -> None:
    config = load_config(ROOT / "configs" / "v2" / "CSD_v22_5slice.yaml")
    assert config["model"]["out_channels"] == 3
    assert config["model"].get("options", {}).get(
        "output_space", "bilateral_3class"
    ) == "bilateral_3class"
    assert config["spectral_disagreement"].get(
        "loss_mode", "regional_reliability"
    ) == "regional_reliability"


def test_v23_component_configs_are_decision_complete() -> None:
    binary = load_config(ROOT / "configs" / "v2" / "BinarySup_5slice.yaml")
    mean_teacher = load_config(ROOT / "configs" / "v2" / "MeanTeacher_5slice.yaml")
    m1_only = load_config(ROOT / "configs" / "v2" / "CSD_Global_5slice.yaml")

    assert binary["model"]["out_channels"] == 2
    assert binary["model"]["options"]["output_space"] == "binary_union"
    assert binary["spectral_disagreement"]["enabled"] is False

    assert mean_teacher["model"]["out_channels"] == 3
    assert mean_teacher["baseline_ssl"] == {
        "enabled": True,
        "mode": "mean_teacher",
        "weight": 1.0,
        "ramp_epochs": 30,
    }
    assert mean_teacher["spectral_disagreement"]["enabled"] is False

    assert m1_only["spectral_disagreement"]["enabled"] is True
    assert m1_only["spectral_disagreement"]["loss_mode"] == "global_uniform"
    assert m1_only["repeat_morphometry"]["enabled"] is False


def test_v23_formal_suite_locks_five_groups_and_four_contrasts() -> None:
    suite_path = ROOT / "configs" / "autodl" / "q2_v23_formal_ablation.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    methods = {item["id"]: item for item in suite["methods"]}
    assert len(methods) == 5
    assert set(methods) == {
        "V23_BinarySup_5slice",
        "V22_B0plus_5slice",
        "V23_MeanTeacher_5slice",
        "V23_M1_CSD_Global_5slice",
        "KidneyBand_ReliableSSL_v2_2_CSD",
    }
    runnable = {key for key, value in methods.items() if "internal" in value["phases"]}
    assert runnable == {
        "V23_BinarySup_5slice",
        "V23_MeanTeacher_5slice",
        "V23_M1_CSD_Global_5slice",
    }
    phase = suite["phases"]["internal"]
    assert phase["folds"] == [0, 1, 2, 3, 4]
    assert phase["seeds"] == [20260811, 20260812, 20260813]
    contrasts = suite["formal_ablation"]["contrasts"]
    assert [item["id"] for item in contrasts] == ["M3", "MT_control", "M1", "M2"]
    assert suite["prior_results"]["sha256"] == (
        "369f6788b5493c5adb4dd76cc8c657e683b65144ac3cd939f7ba9832fa2fec96"
    )


def _tiny_case(root: Path, subject: str, fold: int) -> dict[str, object]:
    image_path = root / f"{subject}.nii.gz"
    mask_path = root / f"{subject}_mask.nii.gz"
    image = np.random.default_rng(sum(map(ord, subject))).normal(size=(16, 16, 3)).astype(np.float32)
    mask = np.zeros((16, 16, 3), dtype=np.uint8)
    mask[2:6, 5:10, :] = 1
    mask[10:14, 5:10, :] = 1
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    return {
        "subject_id": subject,
        "cohort": "HC" if subject.startswith("HC") else "CKD",
        "acquisition_id": "1",
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "shape": "16x16x3",
        "spacing": "1x1x1",
        "is_repeat": False,
        "test_fold": fold,
    }


def _tiny_config(tmp_path: Path, variant: str) -> Path:
    data = tmp_path / "data"
    split = data / "splits"
    split.mkdir(parents=True)
    pd.DataFrame(
        [
            _tiny_case(data, "HC_01", 1),
            _tiny_case(data, "CKD_02", 1),
            _tiny_case(data, "CKD_01", 0),
        ]
    ).to_csv(split / "manifest_with_folds.csv", index=False)
    pd.DataFrame([{"subject_id": "HC_01", "labeled_fraction": 0.25}]).to_csv(
        split / "labeled_fold0.csv", index=False
    )
    binary = variant == "binary"
    mean_teacher = variant == "mean_teacher"
    m1_only = variant == "m1_only"
    config = {
        "schema_version": 2,
        "data": {
            "root": str(data), "manifest": "unused", "folds": str(split),
            "image_size": [16, 16], "clip_percentiles": [0.5, 99.5],
        },
        "split": {"labeled_fraction": 0.25, "fold": 0, "seed": 7},
        "model": {
            "name": "monai_unet", "trainer": "native_v2", "channels": [4, 8],
            "residual_units": 1, "in_channels": 3, "out_channels": 2 if binary else 3,
            "options": {"loss": "dice_ce", "output_space": "binary_union" if binary else "bilateral_3class"},
        },
        "bilateral_labels": {"enabled": True},
        "augmentation": {
            "enabled": True, "lr_flip_probability": 0.0, "rotation_degrees": 2.0,
            "scale_min": 0.98, "scale_max": 1.02, "translation_fraction": 0.01,
            "gamma_min": 0.95, "gamma_max": 1.05, "contrast_min": 0.95,
            "contrast_max": 1.05, "bias_strength": 0.01, "noise_std": 0.01,
        },
        "adaptive_band": {"enabled": False, "angles": 18, "width_fraction": 0.25, "uniform_mixture": 0.2, "temperature": 0.5, "weight": 0.05},
        "spectral_disagreement": {
            "enabled": m1_only, "loss_mode": "global_uniform" if m1_only else "regional_reliability",
            "center_fraction": 0.08, "base_keep_rate": 0.7, "radial_power": 4,
            "boundary_width": 3, "boundary_weight": 0.2, "weight": 1.0,
            "ema_decay": 0.99, "ramp_epochs": 1,
        },
        "repeat_morphometry": {"enabled": False, "start_epoch": 30, "cv_thresholds": [0.05, 0.05, 0.04], "anchor_weight": 0.1, "weight": 0.01},
        "baseline_ssl": {"enabled": mean_teacher, "mode": "mean_teacher" if mean_teacher else "none", "weight": 1.0, "ramp_epochs": 1},
        "external_validation": {"manifests": {}},
        "spectral_band": {"enabled": False}, "mean_teacher": {"enabled": False}, "repeat_tkv": {"enabled": False},
        "optimizer": {
            "learning_rate": 3e-4, "minimum_learning_rate": 1e-6, "weight_decay": 1e-5,
            "warmup_epochs": 1, "epochs": 1, "batch_size": 2, "gradient_clip": 1.0,
        },
        "runtime": {
            "output_root": str(tmp_path / "experiments"), "run_name": f"V23_{variant}",
            "precision": "fp32", "num_workers": 0, "heartbeat_seconds": 1,
            "device": "cpu", "cache_volumes": True, "resume": False,
            "max_steps_per_epoch": 2,
        },
    }
    path = tmp_path / f"{variant}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.mark.parametrize("variant", ["binary", "mean_teacher", "m1_only"])
def test_v23_new_variant_two_step_cpu_smoke(tmp_path: Path, variant: str) -> None:
    config_path = _tiny_config(tmp_path, variant)
    run = train_v2(config_path, fold=0, seed=7)
    assert (run / "last_teacher.pt").is_file()
    metrics = json.loads((run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert np.isfinite(metrics["loss_total"])
    if variant == "mean_teacher":
        assert metrics["loss_baseline_ssl"] >= 0
    if variant == "m1_only":
        assert metrics["loss_consistency"] >= 0
        assert metrics["loss_boundary"] == 0
    evaluate_v2(run)
    acquisition = pd.read_csv(run / "metrics_acquisition.csv")
    expected_space = "binary_union" if variant == "binary" else "bilateral_3class"
    assert acquisition.output_space.unique().tolist() == [expected_space]
    assert np.isfinite(acquisition[["dice", "hd95_mm", "pred_tkv_ml"]].to_numpy()).all()

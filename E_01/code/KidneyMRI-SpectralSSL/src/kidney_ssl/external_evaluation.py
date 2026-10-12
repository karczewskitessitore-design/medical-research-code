from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from kidney_ssl.bilateral import LEFT_KIDNEY, RIGHT_KIDNEY
from kidney_ssl.config import load_config
from kidney_ssl.data.dataset import VolumeStore, volume_batch
from kidney_ssl.evaluation_v2 import _predict, _restore_model_volume
from kidney_ssl.metrics import dice_score, surface_distances, volume_ml
from kidney_ssl.metrics_v2 import calibration_metrics, error_detection_metrics, surface_dice
from kidney_ssl.model import build_model
from kidney_ssl.utils import atomic_json, choose_device


def _discover_runs(run_dir: Path, seed: int | None = None) -> list[Path]:
    if (run_dir / "last_teacher.pt").is_file():
        return [run_dir]
    runs = sorted(
        path.parent
        for path in run_dir.glob("fold*_seed*/last_teacher.pt")
        if (path.parent / "resolved_config.yaml").is_file()
    )
    if not runs:
        runs = sorted(
            path.parent
            for path in run_dir.glob("**/fold*_seed*/last_teacher.pt")
            if (path.parent / "resolved_config.yaml").is_file()
        )
    if seed is not None:
        runs = [path for path in runs if path.name.endswith(f"_seed{seed}")]
    if not runs:
        raise FileNotFoundError(f"No schema-v2 Teacher checkpoints below {run_dir}")
    return runs


def _manifest_from_config(config: dict[str, Any], dataset: str) -> Path:
    external = config.get("external_validation", {})
    manifests = external.get("manifests", {})
    aliases = {
        "chaos_t2_spir": "chaos",
        "amos22_mri": "amos",
        "chaos": "chaos",
        "amos": "amos",
    }
    key = aliases.get(dataset.lower())
    if key is None:
        raise ValueError("dataset must be chaos, chaos_t2_spir, amos, or amos22_mri")
    candidate = manifests.get(key)
    if not candidate:
        raise ValueError(f"external_validation.manifests.{key} is not configured")
    path = Path(str(candidate)).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"External manifest does not exist: {path}. Run kidney-seg prepare-external first."
        )
    return path


@torch.no_grad()
def evaluate_external(
    run_dir: str | Path,
    dataset: str,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    run_dir = Path(run_dir).resolve()
    runs = _discover_runs(run_dir, seed)
    configs = [load_config(path / "resolved_config.yaml") for path in runs]
    if any(int(config.get("schema_version", 1)) != 2 for config in configs):
        raise ValueError("External evaluation accepts schema-version 2 runs only")
    first = configs[0]
    manifest_path = _manifest_from_config(first, dataset)
    manifest = pd.read_csv(manifest_path)
    required = {"subject_id", "acquisition_id", "image_path", "mask_path"}
    if not required.issubset(manifest):
        raise ValueError(f"External manifest missing columns: {sorted(required - set(manifest))}")
    device = choose_device(str(first["runtime"]["device"]))
    precision = str(first["runtime"].get("precision", "bf16")) if device.type == "cuda" else "fp32"
    output = Path(output_dir).resolve() if output_dir else run_dir / "external" / dataset.lower()
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    store = VolumeStore(
        tuple(first["data"]["image_size"]),
        tuple(first["data"]["clip_percentiles"]),
        bool(first["runtime"]["cache_volumes"]),
    )
    rows = []
    for record in manifest.to_dict("records"):
        probability_sum = None
        disagreement_sum = None
        target_image = nib.as_closest_canonical(nib.load(str(record["mask_path"])))
        target_labels = np.asarray(target_image.dataobj).astype(np.uint8)
        for current_run, config in zip(runs, configs):
            checkpoint = torch.load(
                current_run / "last_teacher.pt", map_location="cpu", weights_only=True
            )
            model = build_model(config["model"]).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            slices, volume = volume_batch(
                store,
                record,
                load_mask=False,
                stack_depth=int(config["model"]["in_channels"]),
            )
            probability, disagreement = _predict(
                model,
                slices,
                device,
                int(config["optimizer"]["batch_size"]),
                precision,
                config["spectral_disagreement"],
            )
            restored_probability = _restore_model_volume(probability, volume, "bilinear")
            restored_disagreement = _restore_model_volume(disagreement, volume, "bilinear")
            if restored_probability.shape[:3] != target_labels.shape:
                raise ValueError(
                    f"Prediction/target shape mismatch for {record['subject_id']}: "
                    f"{restored_probability.shape[:3]} vs {target_labels.shape}"
                )
            probability_sum = (
                restored_probability
                if probability_sum is None
                else probability_sum + restored_probability
            )
            disagreement_sum = (
                restored_disagreement
                if disagreement_sum is None
                else disagreement_sum + restored_disagreement
            )
            del model
        probability_mean = probability_sum / len(runs)
        disagreement_mean = disagreement_sum / len(runs)
        prediction = probability_mean.argmax(axis=-1).astype(np.uint8)
        pred_union = prediction > 0
        target_union = target_labels > 0
        spacing = tuple(float(value) for value in target_image.header.get_zooms()[:3])
        pred_tkv = volume_ml(pred_union, spacing)
        true_tkv = volume_ml(target_union, spacing)
        hd95, asd = surface_distances(pred_union, target_union, spacing)
        kidney_probability = probability_mean[..., LEFT_KIDNEY] + probability_mean[..., RIGHT_KIDNEY]
        calibration = calibration_metrics(kidney_probability, target_union)
        errors = pred_union != target_union
        uncertainty = error_detection_metrics(disagreement_mean, errors)
        case_id = str(record["subject_id"])
        nib.save(
            nib.Nifti1Image(prediction, target_image.affine, target_image.header),
            prediction_dir / f"{case_id}_bilateral.nii.gz",
        )
        rows.append(
            {
                "subject_id": case_id,
                "cohort": str(record.get("dataset", dataset)),
                "is_repeat": False,
                "dice": dice_score(pred_union, target_union),
                "dice_left": dice_score(prediction == LEFT_KIDNEY, target_labels == LEFT_KIDNEY),
                "dice_right": dice_score(prediction == RIGHT_KIDNEY, target_labels == RIGHT_KIDNEY),
                "hd95_mm": hd95,
                "asd_mm": asd,
                "surface_dice_2mm": surface_dice(pred_union, target_union, spacing, 2.0),
                "pred_tkv_ml": pred_tkv,
                "true_tkv_ml": true_tkv,
                "tkv_relative_error": abs(pred_tkv - true_tkv) / max(true_tkv, 1e-8),
                **calibration,
                **uncertainty,
            }
        )
        print(f"EXTERNAL_HEARTBEAT dataset={dataset} case={case_id}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "metrics_subject.csv", index=False)
    numeric = table.select_dtypes(include=[np.number]).columns.tolist()
    summary = {
        "schema_version": 2,
        "dataset": dataset,
        "source_only": True,
        "target_training_or_tuning": False,
        "ensemble_runs": [str(path) for path in runs],
        "subjects": len(table),
        "overall": {column: float(table[column].mean()) for column in numeric},
    }
    atomic_json(summary, output / "evaluation_summary.json")
    return output / "evaluation_summary.json"

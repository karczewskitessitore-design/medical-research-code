from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from nibabel.processing import resample_from_to

from kidney_ssl.config import load_config
from kidney_ssl.data.dataset import VolumeStore, volume_batch
from kidney_ssl.metrics import dice_score, repeatability_summary, surface_distances, volume_ml
from kidney_ssl.model import build_model, primary_logits
from kidney_ssl.utils import atomic_json, choose_device


def _dihedral(image: torch.Tensor, turns: int, flip: bool) -> torch.Tensor:
    value = torch.rot90(image, turns, dims=(-2, -1))
    return torch.flip(value, dims=(-1,)) if flip else value


def _inverse_dihedral(image: torch.Tensor, turns: int, flip: bool) -> torch.Tensor:
    value = torch.flip(image, dims=(-1,)) if flip else image
    return torch.rot90(value, -turns, dims=(-2, -1))


def _enable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()


@torch.no_grad()
def _predict(
    model,
    slices,
    device,
    batch_size,
    amp,
    inference: str = "deterministic",
    samples: int = 20,
):
    outputs = []
    context = (lambda: torch.amp.autocast("cuda", enabled=amp)) if device.type == "cuda" else nullcontext
    if inference == "mcd":
        _enable_dropout(model)
    for start in range(0, len(slices), batch_size):
        image = slices[start : start + batch_size].to(device)
        with context():
            if inference == "deterministic":
                probability = torch.sigmoid(primary_logits(model(image)))
            elif inference == "mcd":
                probability = torch.stack(
                    [torch.sigmoid(primary_logits(model(image))) for _ in range(samples)], dim=0
                ).mean(dim=0)
            elif inference == "tta":
                transforms = [(turns, flip) for turns in range(4) for flip in (False, True)]
                transforms = transforms[: max(1, min(samples, len(transforms)))]
                predictions = []
                for turns, flip in transforms:
                    transformed = _dihedral(image, turns, flip)
                    predicted = torch.sigmoid(primary_logits(model(transformed)))
                    predictions.append(_inverse_dihedral(predicted, turns, flip))
                probability = torch.stack(predictions, dim=0).mean(dim=0)
            else:
                raise ValueError(f"Unknown inference mode {inference!r}")
            outputs.append(probability.cpu())
    model.eval()
    return torch.cat(outputs, dim=0)[:, 0].permute(1, 2, 0)


def _restore_to_original(probability: torch.Tensor, volume) -> nib.Nifti1Image:
    resized = F.interpolate(
        probability.permute(2, 0, 1).unsqueeze(1),
        size=volume.model_original_shape[:2],
        mode="bilinear",
        align_corners=False,
    )[:, 0].permute(1, 2, 0).numpy()
    canonical_array = np.moveaxis(resized, -1, volume.ras_slice_axis)
    if tuple(canonical_array.shape) != tuple(volume.canonical_shape):
        raise AssertionError(
            f"Restored canonical shape mismatch: {canonical_array.shape} != {volume.canonical_shape}"
        )
    canonical = nib.Nifti1Image(canonical_array.astype(np.float32), volume.canonical_affine)
    return resample_from_to(
        canonical,
        (volume.original_image.shape, volume.original_image.affine),
        order=1,
    )


def evaluate(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).resolve()
    config = load_config(run_dir / "resolved_config.yaml")
    fold = int(config["split"]["fold"])
    device = choose_device(str(config["runtime"]["device"]))
    amp = bool(config["runtime"]["amp"] and device.type == "cuda")
    checkpoint_role = str(config["runtime"].get("checkpoint", "teacher"))
    if checkpoint_role not in {"student", "teacher"}:
        raise ValueError("runtime.checkpoint must be 'student' or 'teacher'")
    checkpoint_path = run_dir / f"last_{checkpoint_role}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    options = config["model"].get("options", {})
    inference = str(options.get("inference", "deterministic"))
    inference_samples = int(options.get("inference_samples", 20))
    inference_modes = [str(value) for value in options.get("inference_modes", [inference])]
    if inference not in inference_modes:
        inference_modes.insert(0, inference)
    unknown_modes = set(inference_modes) - {"deterministic", "mcd", "tta"}
    if unknown_modes:
        raise ValueError(f"Unknown inference modes: {sorted(unknown_modes)}")
    stack_depth = int(config["model"]["in_channels"])

    manifest = pd.read_csv(Path(config["data"]["folds"]) / "manifest_with_folds.csv")
    test_records = manifest[manifest.test_fold == fold].reset_index(drop=True)
    store = VolumeStore(
        tuple(config["data"]["image_size"]),
        tuple(config["data"]["clip_percentiles"]),
        bool(config["runtime"]["cache_volumes"]),
    )
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    rows_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in inference_modes}
    inference_started = time.perf_counter()
    for record in test_records.to_dict("records"):
        slices, volume = volume_batch(store, record, load_mask=True, stack_depth=stack_depth)
        target_image = nib.load(str(record["mask_path"]))
        target = np.asarray(target_image.dataobj) > 0
        spacing = tuple(float(v) for v in target_image.header.get_zooms()[:3])
        true_tkv = volume_ml(target, spacing)
        for mode in inference_modes:
            probability = _predict(
                model,
                slices,
                device,
                int(config["optimizer"]["batch_size"]),
                amp,
                mode,
                inference_samples,
            )
            restored = _restore_to_original(probability, volume)
            mode_suffix = "" if mode == inference else f"_{mode}"
            output_path = prediction_dir / (
                f"{record['subject_id']}_{record['acquisition_id']}{mode_suffix}_prob.nii.gz"
            )
            nib.save(restored, output_path)
            pred = np.asarray(restored.dataobj) >= 0.5
            hd95, asd = surface_distances(pred, target, spacing)
            pred_tkv = volume_ml(pred, spacing)
            rows_by_mode[mode].append(
                {
                    "subject_id": str(record["subject_id"]),
                    "cohort": str(record["cohort"]),
                    "acquisition_id": str(record["acquisition_id"]),
                    "is_repeat": bool(record["is_repeat"]),
                    "dice": dice_score(pred, target),
                    "hd95_mm": hd95,
                    "asd_mm": asd,
                    "pred_tkv_ml": pred_tkv,
                    "true_tkv_ml": true_tkv,
                    "tkv_relative_error": abs(pred_tkv - true_tkv) / max(true_tkv, 1e-8),
                    "prediction_path": str(output_path),
                }
            )
            print(
                f"EVALUATION_HEARTBEAT subject={record['subject_id']} "
                f"acquisition={record['acquisition_id']} mode={mode}",
                flush=True,
            )
    acquisition_table = pd.DataFrame(rows_by_mode[inference])
    acquisition_table.to_csv(run_dir / "metrics_acquisition.csv", index=False)
    numeric = ["dice", "hd95_mm", "asd_mm", "pred_tkv_ml", "true_tkv_ml", "tkv_relative_error"]
    subject_table = acquisition_table.groupby(
        ["subject_id", "cohort", "is_repeat"], as_index=False
    )[numeric].mean()
    subject_table.to_csv(run_dir / "metrics_subject.csv", index=False)
    inference_variants: dict[str, Any] = {}
    for mode, mode_rows in rows_by_mode.items():
        mode_acquisition = pd.DataFrame(mode_rows)
        mode_subject = mode_acquisition.groupby(
            ["subject_id", "cohort", "is_repeat"], as_index=False
        )[numeric].mean()
        mode_acquisition.to_csv(run_dir / f"metrics_acquisition_{mode}.csv", index=False)
        mode_subject.to_csv(run_dir / f"metrics_subject_{mode}.csv", index=False)
        inference_variants[mode] = {
            "overall": {column: float(mode_subject[column].mean()) for column in numeric},
            "by_cohort": {
                cohort: {column: float(group[column].mean()) for column in numeric}
                for cohort, group in mode_subject.groupby("cohort")
            },
            "repeatability": repeatability_summary(mode_rows),
        }
    repeat = inference_variants[inference]["repeatability"]
    summary = {
        "checkpoint": str(checkpoint_path),
        "evaluation_unit": "subject",
        "subjects": int(subject_table.subject_id.nunique()),
        "acquisitions": int(len(acquisition_table)),
        "overall": {column: float(subject_table[column].mean()) for column in numeric},
        "by_cohort": {
            cohort: {column: float(group[column].mean()) for column in numeric}
            for cohort, group in subject_table.groupby("cohort")
        },
        "repeatability": repeat,
        "model_name": str(config["model"].get("name", "monai_unet")),
        "inference": inference,
        "inference_modes": inference_modes,
        "inference_variants": inference_variants,
        "inference_samples": inference_samples,
        "inference_seconds": time.perf_counter() - inference_started,
        "pilot_non_confirmatory": str(config["runtime"].get("phase", "pilot")) != "confirmatory",
    }
    atomic_json(summary, run_dir / "evaluation_summary.json")

    repeat_rows = acquisition_table[acquisition_table.is_repeat.astype(bool)]
    differences = []
    means = []
    for _, group in repeat_rows.groupby("subject_id"):
        values = group.pred_tkv_ml.to_numpy(float)
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                differences.append(values[i] - values[j])
                means.append((values[i] + values[j]) / 2.0)
    if differences:
        differences_np = np.asarray(differences)
        mean_diff = differences_np.mean()
        sd = differences_np.std(ddof=1)
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.scatter(means, differences, s=18, alpha=0.7)
        axis.axhline(mean_diff, color="black")
        axis.axhline(mean_diff + 1.96 * sd, color="red", linestyle="--")
        axis.axhline(mean_diff - 1.96 * sd, color="red", linestyle="--")
        axis.set_xlabel("Pair mean predicted TKV (mL)")
        axis.set_ylabel("Within-subject pair difference (mL)")
        axis.set_title("Scan-rescan Bland-Altman")
        figure.tight_layout()
        figure.savefig(run_dir / "bland_altman_repeat_tkv.png", dpi=180)
        plt.close(figure)
    return run_dir / "evaluation_summary.json"

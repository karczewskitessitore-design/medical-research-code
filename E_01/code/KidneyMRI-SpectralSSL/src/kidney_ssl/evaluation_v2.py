from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from kidney_ssl.bilateral import LEFT_KIDNEY, RIGHT_KIDNEY, split_binary_kidneys
from kidney_ssl.config import load_config
from kidney_ssl.data.dataset import VolumeStore, volume_batch
from kidney_ssl.metrics import dice_score, surface_distances, volume_ml
from kidney_ssl.metrics_v2 import (
    calibration_metrics,
    error_detection_metrics,
    repeated_measurement_summary,
    risk_coverage_curve,
    surface_dice,
)
from kidney_ssl.model import build_model, primary_logits
from kidney_ssl.spectral import complementary_fourier_masks, fourier_view
from kidney_ssl.utils import atomic_json, choose_device
from kidney_ssl.v2_losses import jensen_shannon_disagreement


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def _enable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()


@torch.no_grad()
def _predict(
    model: torch.nn.Module,
    slices: torch.Tensor,
    device: torch.device,
    batch_size: int,
    precision: str,
    spectral_options: dict,
    inference: str = "deterministic",
    samples: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inference not in {"deterministic", "mcd", "tta"}:
        raise ValueError(f"Unknown inference mode: {inference!r}")
    if inference == "mcd":
        _enable_dropout(model)
    probabilities = []
    disagreements = []
    for start in range(0, len(slices), batch_size):
        image = slices[start : start + batch_size].to(device)
        with _autocast(device, precision):
            if inference == "mcd":
                draws = torch.stack(
                    [
                        torch.softmax(primary_logits(model(image)).float(), dim=1)
                        for _ in range(max(1, samples))
                    ],
                    dim=0,
                )
                probability = draws.mean(dim=0)
                disagreement = draws.var(dim=0, unbiased=False).sum(dim=1, keepdim=True)
            elif inference == "tta":
                count = max(1, min(samples, 9))
                factors = torch.linspace(0.85, 1.15, count, device=image.device)
                mean = image.mean(dim=(-2, -1), keepdim=True)
                draws = torch.stack(
                    [
                        torch.softmax(
                            primary_logits(model(mean + factor * (image - mean))).float(), dim=1
                        )
                        for factor in factors
                    ],
                    dim=0,
                )
                probability = draws.mean(dim=0)
                disagreement = draws.var(dim=0, unbiased=False).sum(dim=1, keepdim=True)
            else:
                probability = torch.softmax(primary_logits(model(image)).float(), dim=1)
            if inference == "deterministic" and bool(spectral_options.get("enabled", False)):
                generator = torch.Generator(device=device).manual_seed(9301 + start)
                theta, lam, _ = complementary_fourier_masks(
                    len(image),
                    image.shape[-2],
                    image.shape[-1],
                    float(spectral_options["center_fraction"]),
                    float(spectral_options["base_keep_rate"]),
                    int(spectral_options["radial_power"]),
                    device,
                    generator,
                )
                probability_a = torch.softmax(
                    primary_logits(model(fourier_view(image, theta))).float(), dim=1
                )
                probability_b = torch.softmax(
                    primary_logits(model(fourier_view(image, lam))).float(), dim=1
                )
                disagreement = jensen_shannon_disagreement(probability_a, probability_b)
            elif inference == "deterministic":
                entropy = -(probability.clamp_min(1e-6) * probability.clamp_min(1e-6).log()).sum(
                    dim=1, keepdim=True
                ) / np.log(probability.shape[1])
                disagreement = entropy
        probabilities.append(probability.cpu())
        disagreements.append(disagreement.cpu())
    model.eval()
    probability_volume = torch.cat(probabilities).permute(1, 2, 3, 0)
    disagreement_volume = torch.cat(disagreements)[:, 0].permute(1, 2, 0)
    return probability_volume, disagreement_volume


def _restore_model_volume(value: torch.Tensor, volume, mode: str) -> np.ndarray:
    channels = value.shape[0] if value.ndim == 4 else None
    if channels is None:
        batch = value.permute(2, 0, 1).unsqueeze(1)
    else:
        batch = value.permute(3, 0, 1, 2)
    resized = F.interpolate(
        batch.float(),
        size=volume.model_original_shape[:2],
        mode=mode,
        align_corners=False if mode != "nearest" else None,
    )
    if channels is None:
        model_array = resized[:, 0].permute(1, 2, 0).numpy()
        canonical = np.moveaxis(model_array, 2, volume.ras_slice_axis)
    else:
        # [D, C, H, W] -> [H, W, D, C].  The class dimension stays last
        # while the acquired slice axis is restored among the three spatial
        # axes.  Keeping these operations explicit avoids confusing the depth
        # and class axes when the original acquisition is not axial.
        model_array = resized.permute(2, 3, 0, 1).numpy()
        canonical = np.moveaxis(model_array, 2, volume.ras_slice_axis)
    return canonical


def evaluate_v2(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir).resolve()
    config = load_config(run_dir / "resolved_config.yaml")
    if int(config.get("schema_version", 1)) != 2:
        raise ValueError("evaluate_v2 requires a schema_version 2 run")
    fold = int(config["split"]["fold"])
    device = choose_device(str(config["runtime"]["device"]))
    precision = str(config["runtime"].get("precision", "bf16" if device.type == "cuda" else "fp32"))
    if device.type != "cuda":
        precision = "fp32"
    checkpoint = torch.load(run_dir / "last_teacher.pt", map_location="cpu", weights_only=True)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    options = config["model"].get("options", {})
    configured_inference = str(options.get("inference", "deterministic"))
    if configured_inference == "mcd_tta_sensitivity":
        inference_modes = ["deterministic", "mcd", "tta"]
    else:
        inference_modes = [
            str(value) for value in options.get("inference_modes", [configured_inference])
        ]
        if "deterministic" not in inference_modes:
            inference_modes.insert(0, "deterministic")
    unknown_modes = set(inference_modes) - {"deterministic", "mcd", "tta"}
    if unknown_modes:
        raise ValueError(f"Unknown inference modes: {sorted(unknown_modes)}")
    inference_samples = int(options.get("inference_samples", 20))
    stack_depth = int(config["model"]["in_channels"])
    manifest = pd.read_csv(Path(config["data"]["folds"]) / "manifest_with_folds.csv")
    test_records = manifest[manifest.test_fold == fold].reset_index(drop=True)
    store = VolumeStore(
        tuple(config["data"]["image_size"]),
        tuple(config["data"]["clip_percentiles"]),
        bool(config["runtime"]["cache_volumes"]),
    )
    rows_by_mode: dict[str, list[dict[str, object]]] = {mode: [] for mode in inference_modes}
    curves_by_mode: dict[str, dict[str, list[dict[str, float]]]] = {
        mode: {} for mode in inference_modes
    }
    started = time.perf_counter()
    for record in test_records.to_dict("records"):
        slices, volume = volume_batch(store, record, load_mask=False, stack_depth=stack_depth)
        target_image = nib.as_closest_canonical(nib.load(str(record["mask_path"])))
        target_binary = np.asarray(target_image.dataobj) > 0
        target_labels = split_binary_kidneys(target_binary, lr_axis=0)
        target_union = target_labels > 0
        spacing = tuple(float(value) for value in target_image.header.get_zooms()[:3])
        true_tkv = volume_ml(target_union, spacing)
        case_id = f"{record['subject_id']}_{record['acquisition_id']}"
        for mode in inference_modes:
            probability, disagreement = _predict(
                model,
                slices,
                device,
                int(config["optimizer"]["batch_size"]),
                precision,
                config["spectral_disagreement"],
                mode,
                inference_samples,
            )
            probability_canonical = _restore_model_volume(probability, volume, "bilinear")
            disagreement_canonical = _restore_model_volume(disagreement, volume, "bilinear")
            predicted_labels = probability_canonical.argmax(axis=-1).astype(np.uint8)
            predicted_union = predicted_labels > 0
            pred_tkv = volume_ml(predicted_union, spacing)
            hd95, asd = surface_distances(predicted_union, target_union, spacing)
            kidney_probability = (
                probability_canonical[..., LEFT_KIDNEY]
                + probability_canonical[..., RIGHT_KIDNEY]
            )
            calibration = calibration_metrics(kidney_probability, target_union)
            error = predicted_union != target_union
            uncertainty = error_detection_metrics(disagreement_canonical, error)
            curves_by_mode[mode][case_id] = risk_coverage_curve(disagreement_canonical, error)
            prediction_dir = run_dir / ("predictions" if mode == "deterministic" else f"predictions_{mode}")
            prediction_dir.mkdir(exist_ok=True)
            nib.save(
                nib.Nifti1Image(predicted_labels, target_image.affine, target_image.header),
                prediction_dir / f"{case_id}_bilateral.nii.gz",
            )
            rows_by_mode[mode].append(
                {
                    "subject_id": str(record["subject_id"]),
                    "cohort": str(record["cohort"]),
                    "acquisition_id": str(record["acquisition_id"]),
                    "is_repeat": bool(record["is_repeat"]),
                    "dice": dice_score(predicted_union, target_union),
                    "dice_left": dice_score(
                        predicted_labels == LEFT_KIDNEY, target_labels == LEFT_KIDNEY
                    ),
                    "dice_right": dice_score(
                        predicted_labels == RIGHT_KIDNEY, target_labels == RIGHT_KIDNEY
                    ),
                    "hd95_mm": hd95,
                    "asd_mm": asd,
                    "surface_dice_2mm": surface_dice(
                        predicted_union, target_union, spacing, 2.0
                    ),
                    "pred_tkv_ml": pred_tkv,
                    "true_tkv_ml": true_tkv,
                    "tkv_relative_error": abs(pred_tkv - true_tkv) / max(true_tkv, 1e-8),
                    **calibration,
                    **uncertainty,
                }
            )
        print(f"EVALUATION_V2_HEARTBEAT case={case_id} modes={inference_modes}", flush=True)
    sensitivity = {"primary_mode": "deterministic", "modes": {}}
    elapsed = time.perf_counter() - started
    for mode, rows in rows_by_mode.items():
        acquisition = pd.DataFrame(rows)
        suffix = "" if mode == "deterministic" else f"_{mode}"
        acquisition.to_csv(run_dir / f"metrics_acquisition{suffix}.csv", index=False)
        numeric = acquisition.select_dtypes(include=[np.number]).columns.difference(
            ["is_repeat"]
        ).tolist()
        subjects = acquisition.groupby(
            ["subject_id", "cohort", "is_repeat"], as_index=False
        )[numeric].mean()
        subjects.to_csv(run_dir / f"metrics_subject{suffix}.csv", index=False)
        summary = {
            "schema_version": 2,
            "evaluation_unit": "subject",
            "inference_mode": mode,
            "primary_statistical_result": mode == "deterministic",
            "subjects": int(subjects.subject_id.nunique()),
            "acquisitions": int(len(acquisition)),
            "overall": {column: float(subjects[column].mean()) for column in numeric},
            "by_cohort": {
                cohort: {column: float(group[column].mean()) for column in numeric}
                for cohort, group in subjects.groupby("cohort")
            },
            "repeatability": repeated_measurement_summary(rows),
            "inference_seconds_all_modes": elapsed,
            "uncertainty_source": (
                "mcd_variance"
                if mode == "mcd"
                else "intensity_tta_variance"
                if mode == "tta"
                else "complementary_spectral_view_js"
                if bool(config["spectral_disagreement"]["enabled"])
                else "predictive_entropy"
            ),
        }
        atomic_json(summary, run_dir / f"evaluation_summary{suffix}.json")
        (run_dir / f"risk_coverage{suffix}.json").write_text(
            json.dumps(curves_by_mode[mode], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        sensitivity["modes"][mode] = summary["overall"]
    if len(inference_modes) > 1:
        atomic_json(sensitivity, run_dir / "inference_sensitivity.json")
    return run_dir / "evaluation_summary.json"

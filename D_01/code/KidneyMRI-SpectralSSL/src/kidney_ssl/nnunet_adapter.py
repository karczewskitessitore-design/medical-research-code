from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.processing import resample_from_to

from kidney_ssl.bilateral import LEFT_KIDNEY, RIGHT_KIDNEY, split_binary_kidneys
from kidney_ssl.config import dump_config, load_config
from kidney_ssl.metrics import dice_score, repeatability_summary, surface_distances, volume_ml
from kidney_ssl.metrics_v2 import surface_dice
from kidney_ssl.utils import atomic_json


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _case_id(subject_id: str, acquisition_id: str) -> str:
    safe_subject = "".join(character if character.isalnum() else "_" for character in subject_id)
    safe_acquisition = "".join(character if character.isalnum() else "_" for character in acquisition_id)
    return f"Kidney_{safe_subject}_{safe_acquisition}"


def prepare_nnunet_external_input(
    manifest: pd.DataFrame, destination: Path
) -> pd.DataFrame:
    """Create an nnU-Net input folder without altering source external data."""
    required = {"subject_id", "acquisition_id", "image_path", "mask_path"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"External manifest missing columns: {sorted(required - set(manifest.columns))}")
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in manifest.to_dict("records"):
        case = "External_" + _case_id(str(row["subject_id"]), str(row["acquisition_id"]))
        source = nib.as_closest_canonical(nib.load(str(row["image_path"])))
        nib.save(source, destination / f"{case}_0000.nii.gz")
        rows.append({"case_id": case, **row})
    return pd.DataFrame(rows)


def prepare_nnunet_dataset(config: dict[str, Any], fold: int, run_dir: Path) -> dict[str, Any]:
    options = config["model"].get("options", {})
    dataset_id = int(options.get("dataset_id", 501))
    dataset_name = f"Dataset{dataset_id:03d}_T2Kidney"
    workspace = run_dir / "nnunet_workspace"
    raw = workspace / "raw"
    preprocessed = workspace / "preprocessed"
    results = workspace / "results"
    dataset = raw / dataset_name
    images_tr = dataset / "imagesTr"
    labels_tr = dataset / "labelsTr"
    images_ts = run_dir / "nnunet_test_input"
    for path in (images_tr, labels_tr, images_ts, preprocessed, results):
        path.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(Path(config["data"]["folds"]) / "manifest_with_folds.csv")
    training = manifest[manifest.test_fold != fold].reset_index(drop=True)
    testing = manifest[manifest.test_fold == fold].reset_index(drop=True)
    labels = pd.read_csv(Path(config["data"]["folds"]) / f"labeled_fold{fold}.csv")
    fraction = float(config["split"]["labeled_fraction"])
    labeled_subjects = set(
        labels[(labels.labeled_fraction.astype(float) - fraction).abs() < 1e-8]
        .subject_id.astype(str)
    )
    labeled = training[training.subject_id.astype(str).isin(labeled_subjects)]
    bilateral = int(config.get("schema_version", 1)) == 2
    mapping_rows = []
    for row in labeled.to_dict("records"):
        case = _case_id(str(row["subject_id"]), str(row["acquisition_id"]))
        if bilateral:
            image = nib.as_closest_canonical(nib.load(str(row["image_path"])))
            mask = nib.as_closest_canonical(nib.load(str(row["mask_path"])))
            labels_bilateral = split_binary_kidneys(np.asarray(mask.dataobj) > 0, lr_axis=0)
            nib.save(image, images_tr / f"{case}_0000.nii.gz")
            nib.save(nib.Nifti1Image(labels_bilateral, mask.affine, mask.header), labels_tr / f"{case}.nii.gz")
        else:
            _link_or_copy(Path(row["image_path"]), images_tr / f"{case}_0000.nii.gz")
            _link_or_copy(Path(row["mask_path"]), labels_tr / f"{case}.nii.gz")
    for row in testing.to_dict("records"):
        case = _case_id(str(row["subject_id"]), str(row["acquisition_id"]))
        if bilateral:
            nib.save(
                nib.as_closest_canonical(nib.load(str(row["image_path"]))),
                images_ts / f"{case}_0000.nii.gz",
            )
        else:
            _link_or_copy(Path(row["image_path"]), images_ts / f"{case}_0000.nii.gz")
        mapping_rows.append({"case_id": case, **row})
    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(run_dir / "nnunet_test_mapping.csv", index=False)
    atomic_json(
        {
            "channel_names": {"0": "T2"},
            "labels": (
                {"background": 0, "right_kidney": RIGHT_KIDNEY, "left_kidney": LEFT_KIDNEY}
                if bilateral
                else {"background": 0, "kidney": 1}
            ),
            "numTraining": int(len(labeled)),
            "file_ending": ".nii.gz",
        },
        dataset / "dataset.json",
    )
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "workspace": workspace,
        "raw": raw,
        "preprocessed": preprocessed,
        "results": results,
        "test_input": images_ts,
        "mapping": mapping,
        "labeled_subjects": len(labeled_subjects),
        "training_cases": len(labeled),
        "bilateral": bilateral,
    }


def _entrypoint(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        path = shutil.which(f"{name}.exe")
    if path is None:
        local = Path(sys.executable).parent / f"{name}.exe"
        path = str(local) if local.exists() else None
    if path is None:
        raise RuntimeError(
            f"{name} is unavailable. Install the locked benchmark extra with "
            "`uv sync --extra benchmark`."
        )
    return path


def _gpu_memory_for_pid(pid: int) -> float:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return 0.0
    result = subprocess.run(
        [
            executable,
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return 0.0
    values = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == pid:
            try:
                values.append(float(fields[1]))
            except ValueError:
                pass
    return max(values, default=0.0)


def _run(command: list[str], env: dict[str, str], log_path: Path) -> dict[str, Any]:
    started = time.time()
    peak_gpu_memory_mb = 0.0
    last_heartbeat = started
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(10)
            peak_gpu_memory_mb = max(peak_gpu_memory_mb, _gpu_memory_for_pid(process.pid))
            now = time.time()
            if now - last_heartbeat >= 60:
                print(
                    f"NNUNET_HEARTBEAT pid={process.pid} elapsed={now - started:.0f}s "
                    f"peak_gpu_mb={peak_gpu_memory_mb:.0f}",
                    flush=True,
                )
                last_heartbeat = now
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"nnU-Net command failed with exit code {return_code}: {command}; log={log_path}"
        )
    result = {
        "command": command,
        "exit_code": return_code,
        "duration_seconds": time.time() - started,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
    }
    atomic_json(result, log_path.with_suffix(".json"))
    return result


def _checkpoint_parameter_count(
    results: Path, dataset_name: str, trainer: str, configuration: str
) -> int | None:
    folder = results / dataset_name / f"{trainer}__nnUNetPlans__{configuration}" / "fold_all"
    checkpoint_path = folder / "checkpoint_final.pth"
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("network_weights", {})
    return int(sum(value.numel() for value in state.values())) if state else None


def _evaluate_predictions(
    run_dir: Path, prepared: dict[str, Any], inference_seconds: float | None = None
) -> Path:
    prediction_dir = run_dir / "predictions"
    rows: list[dict[str, Any]] = []
    for row in prepared["mapping"].to_dict("records"):
        prediction_path = prediction_dir / f"{row['case_id']}.nii.gz"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Missing nnU-Net prediction: {prediction_path}")
        target_image = nib.as_closest_canonical(nib.load(str(row["mask_path"])))
        prediction_image = nib.load(str(prediction_path))
        if prediction_image.shape != target_image.shape or not np.allclose(
            prediction_image.affine, target_image.affine, atol=1e-4
        ):
            prediction_image = resample_from_to(
                prediction_image, (target_image.shape, target_image.affine), order=0
            )
        prediction_labels = np.asarray(prediction_image.dataobj).astype(np.uint8)
        if prepared.get("bilateral", False):
            target_labels = split_binary_kidneys(np.asarray(target_image.dataobj) > 0, lr_axis=0)
        else:
            target_labels = (np.asarray(target_image.dataobj) > 0).astype(np.uint8)
        prediction = prediction_labels > 0
        target = target_labels > 0
        spacing = tuple(float(value) for value in target_image.header.get_zooms()[:3])
        hd95, asd = surface_distances(prediction, target, spacing)
        pred_tkv = volume_ml(prediction, spacing)
        true_tkv = volume_ml(target, spacing)
        result_row = {
                "subject_id": str(row["subject_id"]),
                "cohort": str(row["cohort"]),
                "acquisition_id": str(row["acquisition_id"]),
                "is_repeat": str(row["is_repeat"]).lower() == "true",
                "dice": dice_score(prediction, target),
                "hd95_mm": hd95,
                "asd_mm": asd,
                "pred_tkv_ml": pred_tkv,
                "true_tkv_ml": true_tkv,
                "tkv_relative_error": abs(pred_tkv - true_tkv) / max(true_tkv, 1e-8),
                "prediction_path": str(prediction_path),
            }
        if prepared.get("bilateral", False):
            result_row.update(
                {
                    "dice_left": dice_score(
                        prediction_labels == LEFT_KIDNEY, target_labels == LEFT_KIDNEY
                    ),
                    "dice_right": dice_score(
                        prediction_labels == RIGHT_KIDNEY, target_labels == RIGHT_KIDNEY
                    ),
                }
            )
        rows.append(result_row)
    acquisition = pd.DataFrame(rows)
    acquisition.to_csv(run_dir / "metrics_acquisition.csv", index=False)
    numeric = ["dice", "hd95_mm", "asd_mm", "pred_tkv_ml", "true_tkv_ml", "tkv_relative_error"]
    if prepared.get("bilateral", False):
        numeric.extend(["dice_left", "dice_right"])
    subject = acquisition.groupby(
        ["subject_id", "cohort", "is_repeat"], as_index=False
    )[numeric].mean()
    subject.to_csv(run_dir / "metrics_subject.csv", index=False)
    summary = {
        "evaluation_unit": "subject",
        "subjects": int(subject.subject_id.nunique()),
        "acquisitions": int(len(acquisition)),
        "overall": {metric: float(subject[metric].mean()) for metric in numeric},
        "by_cohort": {
            cohort: {metric: float(group[metric].mean()) for metric in numeric}
            for cohort, group in subject.groupby("cohort")
        },
        "repeatability": repeatability_summary(rows),
        "model_name": "nnunetv2_3d_fullres" if prepared.get("bilateral", False) else "nnunetv2_2d",
        "inference_seconds": inference_seconds,
        "pilot_non_confirmatory": False,
    }
    atomic_json(summary, run_dir / "evaluation_summary.json")
    return run_dir / "evaluation_summary.json"


def run_nnunet(config_path: str | Path, fold: int, seed: int) -> Path:
    config = load_config(config_path)
    config["split"]["fold"] = int(fold)
    config["split"]["seed"] = int(seed)
    run_dir = (
        Path(config["runtime"]["output_root"]).resolve()
        / str(config["runtime"]["run_name"])
        / f"fold{fold}_seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, run_dir / "resolved_config.yaml")
    prepared = prepare_nnunet_dataset(config, fold, run_dir)
    env = os.environ.copy()
    env.update(
        {
            "nnUNet_raw": str(prepared["raw"]),
            "nnUNet_preprocessed": str(prepared["preprocessed"]),
            "nnUNet_results": str(prepared["results"]),
            "nnUNet_n_proc_DA": str(config["runtime"].get("num_workers", 0) or 1),
        }
    )
    options = config["model"].get("options", {})
    trainer = str(options.get("trainer_class", "nnUNetTrainer_100epochs"))
    configuration = str(options.get("configuration", "2d"))
    dataset_id = str(prepared["dataset_id"])
    started = time.time()
    plan_stats = _run(
        [
            _entrypoint("nnUNetv2_plan_and_preprocess"),
            "-d",
            dataset_id,
            "-c",
            configuration,
            "--verify_dataset_integrity",
        ],
        env,
        run_dir / "nnunet_plan.log",
    )
    train_stats = _run(
        [
            sys.executable,
            "-m",
            "kidney_ssl.nnunet_seeded_train",
            dataset_id,
            configuration,
            "all",
            "--trainer",
            trainer,
            "--seed",
            str(seed),
        ],
        env,
        run_dir / "nnunet_train.log",
    )
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    predict_command = [
            _entrypoint("nnUNetv2_predict"),
            "-i",
            str(prepared["test_input"]),
            "-o",
            str(prediction_dir),
            "-d",
            dataset_id,
            "-c",
            configuration,
            "-f",
            "all",
            "-tr",
            trainer,
        ]
    if bool(options.get("disable_tta", False)):
        predict_command.append("--disable_tta")
    predict_stats = _run(
        predict_command,
        env,
        run_dir / "nnunet_predict.log",
    )
    _evaluate_predictions(run_dir, prepared, predict_stats["duration_seconds"])
    atomic_json(
        {
            "status": "completed",
            "run_name": config["runtime"]["run_name"],
            "fold": fold,
            "seed": seed,
            "duration_seconds": time.time() - started,
            "model_name": str(config["model"].get("name", "nnunetv2_2d")),
            "trainer_class": trainer,
            "labeled_subjects": prepared["labeled_subjects"],
            "training_cases": prepared["training_cases"],
            "trainable_parameters": _checkpoint_parameter_count(
                prepared["results"], prepared["dataset_name"], trainer, configuration
            ),
            "peak_gpu_memory_mb": train_stats["peak_gpu_memory_mb"],
            "stage_durations_seconds": {
                "plan_and_preprocess": plan_stats["duration_seconds"],
                "train": train_stats["duration_seconds"],
                "predict": predict_stats["duration_seconds"],
            },
        },
        run_dir / "training_result.json",
    )
    return run_dir


def _external_manifest(config: dict[str, Any], dataset: str) -> Path:
    aliases = {
        "chaos": "chaos",
        "chaos_t2_spir": "chaos",
        "amos": "amos",
        "amos22_mri": "amos",
    }
    key = aliases.get(dataset.lower())
    if key is None:
        raise ValueError("dataset must be chaos_t2_spir or amos22_mri")
    value = config.get("external_validation", {}).get("manifests", {}).get(key)
    if not value:
        raise ValueError(f"external_validation.manifests.{key} is not configured")
    path = Path(str(value)).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _evaluate_external_nnunet_ensemble(
    output: Path,
    mapping: pd.DataFrame,
    ensemble_dir: Path,
    dataset: str,
    run_dirs: list[Path],
) -> Path:
    rows = []
    for row in mapping.to_dict("records"):
        prediction_path = ensemble_dir / f"{row['case_id']}.nii.gz"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing ensemble prediction: {prediction_path}")
        target_image = nib.as_closest_canonical(nib.load(str(row["mask_path"])))
        prediction_image = nib.load(str(prediction_path))
        if prediction_image.shape != target_image.shape or not np.allclose(
            prediction_image.affine, target_image.affine, atol=1e-4
        ):
            prediction_image = resample_from_to(
                prediction_image, (target_image.shape, target_image.affine), order=0
            )
        prediction = np.asarray(prediction_image.dataobj).astype(np.uint8)
        target = np.asarray(target_image.dataobj).astype(np.uint8)
        pred_union = prediction > 0
        target_union = target > 0
        spacing = tuple(float(value) for value in target_image.header.get_zooms()[:3])
        hd95, asd = surface_distances(pred_union, target_union, spacing)
        pred_tkv = volume_ml(pred_union, spacing)
        true_tkv = volume_ml(target_union, spacing)
        rows.append(
            {
                "subject_id": str(row["subject_id"]),
                "cohort": str(row.get("dataset", dataset)),
                "is_repeat": False,
                "dice": dice_score(pred_union, target_union),
                "dice_left": dice_score(prediction == LEFT_KIDNEY, target == LEFT_KIDNEY),
                "dice_right": dice_score(prediction == RIGHT_KIDNEY, target == RIGHT_KIDNEY),
                "hd95_mm": hd95,
                "asd_mm": asd,
                "surface_dice_2mm": surface_dice(pred_union, target_union, spacing, 2.0),
                "pred_tkv_ml": pred_tkv,
                "true_tkv_ml": true_tkv,
                "tkv_relative_error": abs(pred_tkv - true_tkv) / max(true_tkv, 1e-8),
                "prediction_path": str(prediction_path),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "metrics_subject.csv", index=False)
    numeric = table.select_dtypes(include=[np.number]).columns.tolist()
    atomic_json(
        {
            "schema_version": 2,
            "dataset": dataset,
            "source_only": True,
            "target_training_or_tuning": False,
            "ensemble_type": "official_nnunet_probability_average",
            "ensemble_runs": [str(path) for path in run_dirs],
            "subjects": int(len(table)),
            "overall": {column: float(table[column].mean()) for column in numeric},
        },
        output / "evaluation_summary.json",
    )
    return output / "evaluation_summary.json"


def evaluate_nnunet_external(
    internal_method_root: str | Path,
    dataset: str,
    seed: int,
    output_dir: str | Path,
) -> Path:
    """Probability-average the five source folds with official nnU-Net tools."""
    internal_method_root = Path(internal_method_root).resolve()
    run_dirs = sorted(
        path.parent
        for path in internal_method_root.glob(f"fold*_seed{seed}/training_result.json")
        if (path.parent / "resolved_config.yaml").is_file()
        and (path.parent / "nnunet_workspace" / "results").is_dir()
    )
    if len(run_dirs) != 5:
        raise RuntimeError(
            f"Expected five completed nnU-Net outer-fold runs for seed {seed}, found {len(run_dirs)}"
        )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(run_dirs[0] / "resolved_config.yaml")
    manifest = pd.read_csv(_external_manifest(config, dataset))
    input_dir = output / "input"
    mapping = prepare_nnunet_external_input(manifest, input_dir)
    mapping.to_csv(output / "external_mapping.csv", index=False)
    probability_folders = []
    for run in run_dirs:
        current = load_config(run / "resolved_config.yaml")
        options = current["model"].get("options", {})
        dataset_id = str(int(options.get("dataset_id", 502)))
        trainer = str(options.get("trainer_class", "nnUNetTrainer"))
        configuration = str(options.get("configuration", "3d_fullres"))
        workspace = run / "nnunet_workspace"
        env = os.environ.copy()
        env.update(
            {
                "nnUNet_raw": str(workspace / "raw"),
                "nnUNet_preprocessed": str(workspace / "preprocessed"),
                "nnUNet_results": str(workspace / "results"),
            }
        )
        prediction_folder = output / "fold_probabilities" / run.name
        prediction_folder.mkdir(parents=True, exist_ok=True)
        command = [
            _entrypoint("nnUNetv2_predict"),
            "-i",
            str(input_dir),
            "-o",
            str(prediction_folder),
            "-d",
            dataset_id,
            "-c",
            configuration,
            "-f",
            "all",
            "-tr",
            trainer,
            "--save_probabilities",
            "--continue_prediction",
        ]
        if bool(options.get("disable_tta", False)):
            command.append("--disable_tta")
        _run(command, env, output / f"predict_{run.name}.log")
        probability_folders.append(prediction_folder)
    ensemble_dir = output / "predictions"
    _run(
        [
            _entrypoint("nnUNetv2_ensemble"),
            "-i",
            *[str(path) for path in probability_folders],
            "-o",
            str(ensemble_dir),
            "-np",
            "1",
            "--save_npz",
        ],
        os.environ.copy(),
        output / "nnunet_ensemble.log",
    )
    return _evaluate_external_nnunet_ensemble(
        output, mapping, ensemble_dir, dataset, run_dirs
    )

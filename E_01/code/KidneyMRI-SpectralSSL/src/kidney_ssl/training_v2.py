from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from kidney_ssl.augmentation import AugmentationConfig, augment_bilateral_teacher_student
from kidney_ssl.baseline_losses_v2 import (
    cross_head_mutual_loss,
    frcnet_output_consistency,
    probability_mse,
    randdg_frequency_randomize,
    semi_urf_uncertainty_loss,
)
from kidney_ssl.bilateral import bilateral_soft_volumes_ml
from kidney_ssl.config import dump_config, load_config
from kidney_ssl.data.dataset import VolumeStore, volume_batch
from kidney_ssl.data.dataset_v2 import BilateralKidneySliceDataset, collate_bilateral_slices
from kidney_ssl.model import build_model, create_teacher, logits_heads, primary_logits, update_ema
from kidney_ssl.spectral import make_complementary_views
from kidney_ssl.utils import Heartbeat, atomic_json, choose_device, linear_ramp, seed_everything
from kidney_ssl.v2_losses import (
    UncertaintyAdaptiveBandLoss,
    dead_zone_bilateral_morphometry_loss,
    multiclass_supervised_loss,
    one_sided_pcgrad,
    spectral_disagreement_teacher_loss,
)


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def _scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup: int, minimum: float):
    base = optimizer.param_groups[0]["lr"]

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return max((epoch + 1) / max(warmup, 1), minimum / base)
        progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum / base + (1.0 - minimum / base) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _first_nonfinite(model: torch.nn.Module) -> str | None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            return f"parameter:{name}"
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return f"gradient:{name}"
    return None


def _write_failure(
    run_dir: Path,
    stage: str,
    epoch: int,
    batch: int | None,
    model: torch.nn.Module,
    **values: Any,
) -> None:
    payload: dict[str, Any] = {
        "stage": stage,
        "epoch": epoch,
        "batch": batch,
        "first_nonfinite_parameter_or_gradient": _first_nonfinite(model),
    }
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            detached = value.detach().float()
            finite_values = detached[torch.isfinite(detached)]
            payload[name] = {
                "shape": list(value.shape),
                "finite": bool(torch.isfinite(value).all()),
                "minimum_finite": float(finite_values.amin()) if finite_values.numel() else None,
                "maximum_finite": float(finite_values.amax()) if finite_values.numel() else None,
            }
        else:
            payload[name] = value
    atomic_json(payload, run_dir / "failure_diagnostics.json")


def _predict_bilateral_volumes(
    model: torch.nn.Module,
    store: VolumeStore,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    precision: str,
    stack_depth: int,
    gradient: bool,
) -> torch.Tensor:
    outputs = []
    context = nullcontext() if gradient else torch.no_grad()
    with context:
        for record in records:
            slices, volume = volume_batch(store, record, load_mask=False, stack_depth=stack_depth)
            pieces = []
            for start in range(0, len(slices), batch_size):
                image = slices[start : start + batch_size].to(device)
                with _autocast(device, precision):
                    if gradient:
                        logits = gradient_checkpoint(
                            lambda value: primary_logits(model(value)), image, use_reentrant=False
                        )
                    else:
                        logits = primary_logits(model(image))
                    pieces.append(torch.softmax(logits.float(), dim=1))
            probability = torch.cat(pieces, dim=0).permute(1, 2, 3, 0).unsqueeze(0)
            voxel_ml = float(torch.tensor(volume.spacing_resampled).prod()) / 1000.0
            outputs.append(bilateral_soft_volumes_ml(probability, voxel_ml)[0])
    return torch.stack(outputs)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def train_v2(config_path: str | Path, fold: int, seed: int) -> Path:
    config = load_config(config_path)
    if int(config.get("schema_version", 1)) != 2:
        raise ValueError("train_v2 requires schema_version: 2")
    config["split"]["fold"] = int(fold)
    config["split"]["seed"] = int(seed)
    seed_everything(seed)
    device = choose_device(str(config["runtime"]["device"]))
    precision = str(config["runtime"].get("precision", "bf16" if device.type == "cuda" else "fp32"))
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("runtime.precision must be fp32, fp16, or bf16")
    if device.type != "cuda":
        precision = "fp32"
    stack_depth = int(config["model"]["in_channels"])
    run_name = str(config["runtime"]["run_name"])
    run_dir = Path(config["runtime"]["output_root"]).resolve() / run_name / f"fold{fold}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, run_dir / "resolved_config.yaml")
    metrics_path = run_dir / "metrics.jsonl"

    manifest = pd.read_csv(Path(config["data"]["folds"]) / "manifest_with_folds.csv")
    train_records = manifest[manifest.test_fold != fold].reset_index(drop=True)
    label_table = pd.read_csv(Path(config["data"]["folds"]) / f"labeled_fold{fold}.csv")
    fraction = float(config["split"]["labeled_fraction"])
    labeled_subjects = set(
        label_table[
            (label_table.labeled_fraction.astype(float) - fraction).abs() < 1e-8
        ].subject_id.astype(str)
    )
    if not labeled_subjects:
        raise RuntimeError(f"No labeled subjects for fraction {fraction}")

    store = VolumeStore(
        tuple(config["data"]["image_size"]),
        tuple(config["data"]["clip_percentiles"]),
        bool(config["runtime"]["cache_volumes"]),
    )
    dataset = BilateralKidneySliceDataset(train_records, labeled_subjects, store, stack_depth)
    sampler = dataset.balanced_sampler(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["optimizer"]["batch_size"]),
        sampler=sampler,
        num_workers=int(config["runtime"]["num_workers"]),
        collate_fn=collate_bilateral_slices,
        pin_memory=device.type == "cuda",
    )
    student = build_model(config["model"]).to(device)
    teacher = create_teacher(student).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    epochs = int(config["optimizer"]["epochs"])
    scheduler = _scheduler(
        optimizer,
        epochs,
        int(config["optimizer"].get("warmup_epochs", 5)),
        float(config["optimizer"].get("minimum_learning_rate", 1e-6)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    adaptive = config["adaptive_band"]
    band_loss = UncertaintyAdaptiveBandLoss(
        int(adaptive["angles"]),
        float(adaptive["width_fraction"]),
        float(adaptive["uniform_mixture"]),
        float(adaptive["temperature"]),
    )
    supervised_name = str(config["model"].get("options", {}).get("loss", "dice_ce"))
    dice_plus_gamma = float(
        config["model"].get("options", {}).get("dice_plus_gamma", 2.0)
    )
    augmentation = AugmentationConfig.from_mapping(config.get("augmentation"))
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    repeat_safe_columns = [
        column
        for column in ("subject_id", "acquisition_id", "image_path", "shape", "spacing", "cohort")
        if column in train_records.columns
    ]
    repeat_groups = {
        str(subject): group.sort_values("acquisition_id")[repeat_safe_columns].to_dict("records")
        for subject, group in train_records.groupby("subject_id")
        if len(group) == 5
    }
    heartbeat = Heartbeat(int(config["runtime"]["heartbeat_seconds"]))
    start_epoch = 0
    checkpoint_path = run_dir / "last_checkpoint.pt"
    if bool(config["runtime"].get("resume", False)) and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        student.load_state_dict(checkpoint["student"])
        teacher.load_state_dict(checkpoint["teacher"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        _restore_rng(checkpoint["rng"])
        if checkpoint.get("sampler_state") is not None and sampler.generator is not None:
            sampler.generator.set_state(checkpoint["sampler_state"])
        start_epoch = int(checkpoint["epoch"])
    else:
        metrics_path.unlink(missing_ok=True)

    started = time.time()
    for epoch in range(start_epoch, epochs):
        student.train()
        sums = {
            "total": 0.0,
            "supervised": 0.0,
            "band": 0.0,
            "consistency": 0.0,
            "boundary": 0.0,
            "disagreement": 0.0,
            "baseline_ssl": 0.0,
        }
        updates = 0
        gradient_norm_sum = 0.0
        gradient_norm_max = 0.0
        reference_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        max_steps = config["runtime"].get("max_steps_per_epoch")
        for batch_index, batch in enumerate(loader):
            if max_steps is not None and batch_index >= int(max_steps):
                break
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            labeled_indices = batch["labeled_indices"].to(device)
            image, teacher_image, labels = augment_bilateral_teacher_student(
                image, labels, augmentation
            )
            if labeled_indices.numel():
                reference_batch = (
                    image.index_select(0, labeled_indices).detach(),
                    labels.index_select(0, labeled_indices).detach(),
                    teacher_image.index_select(0, labeled_indices).detach(),
                )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                raw_output = student(image)
                output = primary_logits(raw_output)
                total = output.sum() * 0.0
                supervised = output.new_zeros(())
                band = output.new_zeros(())
                consistency = output.new_zeros(())
                boundary = output.new_zeros(())
                disagreement = output.new_zeros(())
                baseline_ssl = output.new_zeros(())
                view_a_probability = None
                view_b_probability = None
                view_logits_a = None
                view_logits_b = None
                if bool(config["spectral_disagreement"]["enabled"]) or bool(adaptive["enabled"]):
                    views = make_complementary_views(
                        image,
                        float(config["spectral_disagreement"]["center_fraction"]),
                        float(config["spectral_disagreement"]["base_keep_rate"]),
                        int(config["spectral_disagreement"]["radial_power"]),
                    )
                    view_logits_a = primary_logits(student(views[0]))
                    view_logits_b = primary_logits(student(views[1]))
                    view_a_probability = torch.softmax(view_logits_a.float(), dim=1)
                    view_b_probability = torch.softmax(view_logits_b.float(), dim=1)
                if labeled_indices.numel():
                    supervised = torch.stack(
                        [
                            multiclass_supervised_loss(
                                head.index_select(0, labeled_indices),
                                labels.index_select(0, labeled_indices),
                                supervised_name,
                                dice_plus_gamma,
                            )
                            for head in logits_heads(raw_output)
                        ]
                    ).mean()
                    total = total + supervised
                    if bool(adaptive["enabled"]):
                        band_result = band_loss(
                            output.index_select(0, labeled_indices),
                            labels.index_select(0, labeled_indices),
                            view_a_probability.index_select(0, labeled_indices),
                            view_b_probability.index_select(0, labeled_indices),
                        )
                        band = band_result.loss
                        total = total + float(adaptive["weight"]) * band
                if bool(config["spectral_disagreement"]["enabled"]):
                    with torch.no_grad():
                        teacher_logits = primary_logits(teacher(teacher_image))
                    csd = spectral_disagreement_teacher_loss(
                        view_logits_a,
                        view_logits_b,
                        teacher_logits,
                        float(config["spectral_disagreement"]["boundary_weight"]),
                        int(config["spectral_disagreement"].get("boundary_width", 5)),
                    )
                    consistency = csd["probability"]
                    boundary = csd["boundary"]
                    disagreement = csd["disagreement"]
                    total = total + (
                        float(config["spectral_disagreement"]["weight"])
                        * linear_ramp(
                            epoch, int(config["spectral_disagreement"]["ramp_epochs"])
                        )
                        * csd["total"]
                    )
                baseline_config = config.get("baseline_ssl", {"enabled": False})
                if bool(baseline_config.get("enabled", False)):
                    mode = str(baseline_config["mode"])
                    with torch.no_grad():
                        baseline_teacher = primary_logits(teacher(teacher_image))
                    if mode == "tcsm":
                        baseline_ssl = probability_mse(output, baseline_teacher)
                    elif mode == "cmmt":
                        heads = logits_heads(raw_output)
                        if len(heads) != 2:
                            raise ValueError("CMMT baseline requires a dual-head model")
                        baseline_ssl = cross_head_mutual_loss(heads[0], heads[1])
                        baseline_ssl = baseline_ssl + probability_mse(output, baseline_teacher)
                    elif mode == "frcnet":
                        baseline_ssl = frcnet_output_consistency(output, baseline_teacher)["total"]
                    elif mode == "semi_urf":
                        baseline_ssl = semi_urf_uncertainty_loss(
                            output,
                            baseline_teacher,
                            min(1.0, (epoch + 1) / max(int(baseline_config.get("ramp_epochs", 30)), 1)),
                        )["total"]
                    elif mode == "randdg":
                        randomized = randdg_frequency_randomize(
                            image,
                            low_frequency_fraction=float(
                                baseline_config.get("low_frequency_fraction", 0.10)
                            ),
                        )
                        randomized_logits = primary_logits(student(randomized))
                        baseline_ssl = probability_mse(randomized_logits, baseline_teacher)
                    else:
                        raise ValueError(f"Unknown baseline_ssl.mode: {mode}")
                    total = total + (
                        float(baseline_config.get("weight", 1.0))
                        * linear_ramp(epoch, int(baseline_config.get("ramp_epochs", 30)))
                        * baseline_ssl
                    )
            if not torch.isfinite(total):
                _write_failure(
                    run_dir,
                    "forward_loss",
                    epoch,
                    batch_index,
                    student,
                    image=image,
                    logits=output,
                    total=total,
                    supervised=supervised,
                    band=band,
                    consistency=consistency,
                    boundary=boundary,
                    baseline_ssl=baseline_ssl,
                    precision=precision,
                )
                raise FloatingPointError(
                    f"Non-finite v2 loss at epoch={epoch} batch={batch_index}; "
                    f"first_nonfinite={_first_nonfinite(student)}"
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(config["optimizer"]["gradient_clip"])
            )
            nonfinite = _first_nonfinite(student)
            if nonfinite is not None or not torch.isfinite(grad_norm):
                _write_failure(
                    run_dir,
                    "backward_gradient",
                    epoch,
                    batch_index,
                    student,
                    total=total,
                    gradient_norm=grad_norm,
                    precision=precision,
                )
                raise FloatingPointError(
                    f"Non-finite v2 gradient at epoch={epoch} batch={batch_index}: {nonfinite}"
                )
            scaler.step(optimizer)
            scaler.update()
            gradient_norm_sum += float(grad_norm.detach())
            gradient_norm_max = max(gradient_norm_max, float(grad_norm.detach()))
            update_ema(student, teacher, float(config["spectral_disagreement"]["ema_decay"]))
            updates += 1
            for name, value in (
                ("total", total),
                ("supervised", supervised),
                ("band", band),
                ("consistency", consistency),
                ("boundary", boundary),
                ("disagreement", disagreement),
                ("baseline_ssl", baseline_ssl),
            ):
                sums[name] += float(value.detach())
            heartbeat.maybe(
                f"run={run_name} epoch={epoch + 1}/{epochs} batch={batch_index + 1}/{len(loader)}"
            )

        repeat_metrics = {"total": torch.zeros((), device=device), "cv_total": torch.zeros((), device=device)}
        repeat_update_applied = False
        repeat_subject: str | None = None
        repeat_config = config["repeat_morphometry"]
        if (
            bool(repeat_config["enabled"])
            and epoch + 1 >= int(repeat_config["start_epoch"])
            and repeat_groups
            and reference_batch is not None
        ):
            subject = sorted(repeat_groups)[epoch % len(repeat_groups)]
            repeat_subject = subject
            records = repeat_groups[subject]
            optimizer.zero_grad(set_to_none=True)
            student_volumes = _predict_bilateral_volumes(
                student,
                store,
                records,
                device,
                int(config["optimizer"]["batch_size"]),
                precision,
                stack_depth,
                True,
            )
            teacher_volumes = _predict_bilateral_volumes(
                teacher,
                store,
                records,
                device,
                int(config["optimizer"]["batch_size"]),
                precision,
                stack_depth,
                False,
            )
            repeat_metrics = dead_zone_bilateral_morphometry_loss(
                student_volumes,
                teacher_volumes,
                tuple(float(value) for value in repeat_config["cv_thresholds"]),
                float(repeat_config["anchor_weight"]),
            )
            weighted_repeat = float(repeat_config["weight"]) * repeat_metrics["total"]
            reference_image, reference_labels, _ = reference_batch
            with _autocast(device, precision):
                reference_loss = multiclass_supervised_loss(
                    primary_logits(student(reference_image)),
                    reference_labels,
                    supervised_name,
                    dice_plus_gamma,
                )
            main_gradients = list(
                torch.autograd.grad(reference_loss, parameters, allow_unused=True)
            )
            auxiliary_gradients = list(
                torch.autograd.grad(weighted_repeat, parameters, allow_unused=True)
            )
            projected = one_sided_pcgrad(main_gradients, auxiliary_gradients)
            for parameter, main_gradient, auxiliary_gradient in zip(
                parameters, main_gradients, projected
            ):
                if main_gradient is None:
                    parameter.grad = auxiliary_gradient
                elif auxiliary_gradient is None:
                    parameter.grad = main_gradient
                else:
                    parameter.grad = main_gradient + auxiliary_gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(config["optimizer"]["gradient_clip"])
            )
            if not torch.isfinite(weighted_repeat) or not torch.isfinite(grad_norm):
                _write_failure(
                    run_dir,
                    "repeat_morphometry_gradient",
                    epoch,
                    None,
                    student,
                    repeat_loss=weighted_repeat,
                    gradient_norm=grad_norm,
                    precision=precision,
                )
                raise FloatingPointError(f"Non-finite repeat morphometry update at epoch={epoch}")
            optimizer.step()
            update_ema(student, teacher, float(config["spectral_disagreement"]["ema_decay"]))
            repeat_update_applied = True
        scheduler.step()

        denominator = max(updates, 1)
        record = {
            "epoch": epoch + 1,
            **{f"loss_{key}": value / denominator for key, value in sums.items()},
            "loss_repeat": float(repeat_metrics["total"].detach()),
            "repeat_cv_total": float(repeat_metrics["cv_total"].detach()),
            "repeat_update_applied": repeat_update_applied,
            "repeat_subject": repeat_subject,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "updates": updates,
            "precision": precision,
            "amp_enabled": bool(device.type == "cuda" and precision in {"fp16", "bf16"}),
            "gradient_norm_mean": gradient_norm_sum / max(updates, 1),
            "gradient_norm_max": gradient_norm_max,
            "elapsed_seconds": time.time() - started,
        }
        _append_jsonl(metrics_path, record)
        print(json.dumps(record, allow_nan=False), flush=True)
        checkpoint = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": _rng_state(),
            "sampler_state": sampler.generator.get_state() if sampler.generator is not None else None,
            "epoch": epoch + 1,
            "seed": seed,
            "fold": fold,
        }
        torch.save(checkpoint, checkpoint_path)
        torch.save({"model": student.state_dict(), "epoch": epoch + 1}, run_dir / "last_student.pt")
        torch.save({"model": teacher.state_dict(), "epoch": epoch + 1}, run_dir / "last_teacher.pt")

    result = {
        "status": "completed",
        "schema_version": 2,
        "run_name": run_name,
        "fold": fold,
        "seed": seed,
        "epochs": epochs,
        "resumed": bool(start_epoch > 0),
        "resumed_from_epoch": int(start_epoch),
        "precision": precision,
        "duration_seconds": time.time() - started,
        "device": str(device),
        "labeled_subjects": len(labeled_subjects),
        "training_subjects": int(train_records.subject_id.nunique()),
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
        ),
    }
    atomic_json(result, run_dir / "training_result.json")
    return run_dir

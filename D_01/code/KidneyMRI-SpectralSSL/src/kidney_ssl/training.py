from __future__ import annotations

import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kidney_ssl.config import dump_config, load_config
from kidney_ssl.data.dataset import (
    KidneySliceDataset,
    VolumeStore,
    collate_slices,
    volume_batch,
)
from kidney_ssl.losses import (
    cross_head_consistency_loss,
    repeat_tkv_loss,
    supervised_loss,
    teacher_consistency_loss,
)
from kidney_ssl.model import build_model, create_teacher, logits_heads, primary_logits, update_ema
from kidney_ssl.spectral import RandomBandLoss, make_complementary_views
from kidney_ssl.utils import Heartbeat, atomic_json, choose_device, linear_ramp, seed_everything


def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return nullcontext()


def _soft_volume(
    model: torch.nn.Module,
    store: VolumeStore,
    record: dict[str, Any],
    device: torch.device,
    batch_size: int,
    amp: bool,
    stack_depth: int,
) -> torch.Tensor:
    slices, volume = volume_batch(store, record, load_mask=False, stack_depth=stack_depth)
    total = torch.zeros((), device=device)
    for start in range(0, len(slices), batch_size):
        image = slices[start : start + batch_size].to(device)
        with _autocast(device, amp):
            total = total + torch.sigmoid(primary_logits(model(image))).sum()
    return total * (float(torch.tensor(volume.spacing_resampled).prod()) / 1000.0)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def _dihedral(image: torch.Tensor, turns: int, flip: bool) -> torch.Tensor:
    transformed = torch.rot90(image, turns, dims=(-2, -1))
    return torch.flip(transformed, dims=(-1,)) if flip else transformed


def _center_scale(image: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply a differentiable center scaling while preserving tensor shape."""
    theta = image.new_zeros((image.shape[0], 2, 3))
    theta[:, 0, 0] = 1.0 / scale
    theta[:, 1, 1] = 1.0 / scale
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _tcsm_consistency(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    image: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    turns = random.randrange(4)
    flip = bool(random.getrandbits(1))
    scale = random.uniform(0.9, 1.1)
    transformed = _dihedral(_center_scale(image, scale), turns, flip)
    student_output = primary_logits(student(transformed))
    with torch.no_grad():
        teacher_output = _dihedral(
            _center_scale(primary_logits(teacher(image)), scale), turns, flip
        )
    return teacher_consistency_loss((student_output,), teacher_output, threshold)


def _cutmix_mask(image: torch.Tensor, probability: float) -> tuple[torch.Tensor, torch.Tensor]:
    permutation = torch.randperm(image.shape[0], device=image.device)
    mask = torch.ones((image.shape[0], 1, image.shape[-2], image.shape[-1]), device=image.device)
    if image.shape[0] < 2 or random.random() >= probability:
        return mask, permutation
    height, width = image.shape[-2:]
    for index in range(image.shape[0]):
        ratio = random.uniform(0.25, 0.50)
        cut_h, cut_w = max(1, int(height * ratio)), max(1, int(width * ratio))
        top = random.randint(0, height - cut_h)
        left = random.randint(0, width - cut_w)
        mask[index, :, top : top + cut_h, left : left + cut_w] = 0.0
    return mask, permutation


def _normalized_perturbation(value: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    flat = value.flatten(1)
    norm = flat.norm(dim=1).clamp_min(epsilon)
    return value / norm.view(-1, 1, 1, 1)


def _cmmt_consistency(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    image: torch.Tensor,
    threshold: float,
    noise_std: float,
    cutmix_probability: float,
    vat_xi: float,
    vat_epsilon: float,
    vat_weight: float,
) -> torch.Tensor:
    cutmix, permutation = _cutmix_mask(image, cutmix_probability)
    strong = cutmix * image + (1.0 - cutmix) * image.index_select(0, permutation)
    strong = strong + torch.randn_like(strong) * noise_std
    with torch.no_grad():
        teacher_original = logits_heads(teacher(image))
        teacher_permuted = tuple(head.index_select(0, permutation) for head in teacher_original)
        teacher_heads = tuple(
            cutmix * original + (1.0 - cutmix) * permuted
            for original, permuted in zip(teacher_original, teacher_permuted)
        )
    student_heads = logits_heads(student(strong))
    mutual = cross_head_consistency_loss(student_heads, teacher_heads, threshold)

    # Clean-room one-step MVAT adaptation: find an input perturbation that most
    # disagrees with the fixed cross-head teacher target, then train against it.
    direction = _normalized_perturbation(torch.randn_like(strong)) * vat_xi
    direction.requires_grad_(True)
    probe_heads = logits_heads(student(strong + direction))
    probe_loss = cross_head_consistency_loss(probe_heads, teacher_heads, threshold)
    if probe_loss.requires_grad:
        gradient = torch.autograd.grad(probe_loss, direction, retain_graph=False)[0]
        adversarial = _normalized_perturbation(gradient).detach() * vat_epsilon
        adversarial_heads = logits_heads(student(strong + adversarial))
        virtual_adversarial = cross_head_consistency_loss(
            adversarial_heads, teacher_heads, threshold
        )
    else:
        virtual_adversarial = mutual.new_zeros(())
    return mutual + vat_weight * virtual_adversarial


def train(config_path: str | Path, fold: int, seed: int) -> Path:
    config = load_config(config_path)
    if str(config["model"].get("trainer", "native")) == "nnunetv2":
        from kidney_ssl.nnunet_adapter import run_nnunet

        return run_nnunet(config_path, fold, seed)
    config["split"]["fold"] = int(fold)
    config["split"]["seed"] = int(seed)
    seed_everything(seed)
    device = choose_device(str(config["runtime"]["device"]))
    amp = bool(config["runtime"]["amp"] and device.type == "cuda")
    stack_depth = int(config["model"]["in_channels"])

    run_name = str(config["runtime"]["run_name"])
    run_dir = (
        Path(config["runtime"]["output_root"]).resolve()
        / run_name
        / f"fold{fold}_seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, run_dir / "resolved_config.yaml")
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

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
    dataset = KidneySliceDataset(train_records, labeled_subjects, store, stack_depth=stack_depth)
    loader = DataLoader(
        dataset,
        batch_size=int(config["optimizer"]["batch_size"]),
        sampler=dataset.balanced_sampler(seed),
        num_workers=int(config["runtime"]["num_workers"]),
        collate_fn=collate_slices,
        pin_memory=device.type == "cuda",
    )

    student = build_model(config["model"]).to(device)
    teacher = create_teacher(student).to(device)
    optimizer_name = str(config["optimizer"].get("name", "AdamW")).lower()
    optimizer_class = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    optimizer = optimizer_class(
        student.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    band_loss = RandomBandLoss(
        int(config["spectral_band"]["angles"]),
        float(config["spectral_band"]["width_fraction"]),
    )
    heartbeat = Heartbeat(int(config["runtime"]["heartbeat_seconds"]))
    repeat_groups = {
        str(subject): group.to_dict("records")
        for subject, group in train_records.groupby("subject_id")
        if len(group) > 1
    }
    started = time.time()
    epochs = int(config["optimizer"]["epochs"])
    for epoch in range(epochs):
        student.train()
        epoch_sums = {"total": 0.0, "supervised": 0.0, "band": 0.0, "consistency": 0.0}
        updates = 0
        for batch_index, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True)
            labeled_indices = batch["labeled_indices"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                output = student(image)
                logits = primary_logits(output)
                total_loss = logits.sum() * 0.0
                supervised = logits.new_zeros(())
                band = logits.new_zeros(())
                consistency = logits.new_zeros(())
                if labeled_indices.numel():
                    target = batch["mask"].to(device, non_blocking=True)
                    labeled_output = tuple(
                        head.index_select(0, labeled_indices) for head in logits_heads(output)
                    )
                    options = config["model"].get("options", {})
                    supervised = supervised_loss(
                        labeled_output,
                        target,
                        str(options.get("loss", "dice_ce")),
                        float(options.get("dice_plus_gamma", 2.0)),
                    )
                    total_loss = total_loss + supervised
                    if bool(config["spectral_band"]["enabled"]):
                        labeled_logits = primary_logits(labeled_output)
                        band = band_loss(torch.sigmoid(labeled_logits), target)
                        total_loss = total_loss + float(config["spectral_band"]["weight"]) * band
                if bool(config["mean_teacher"]["enabled"]):
                    mt = config["mean_teacher"]
                    mode = str(mt.get("mode", "frequency"))
                    if mode == "frequency":
                        view_theta, view_lambda, _, _ = make_complementary_views(
                            image,
                            float(mt["center_fraction"]),
                            float(mt["base_keep_rate"]),
                            int(mt["radial_power"]),
                        )
                        logits_theta = primary_logits(student(view_theta))
                        logits_lambda = primary_logits(student(view_lambda))
                        with torch.no_grad():
                            teacher_logits = primary_logits(teacher(image))
                        consistency = teacher_consistency_loss(
                            (logits_theta, logits_lambda),
                            teacher_logits,
                            float(mt["confidence_threshold"]),
                        )
                    elif mode == "tcsm":
                        consistency = _tcsm_consistency(
                            student, teacher, image, float(mt["confidence_threshold"])
                        )
                    elif mode == "cmmt":
                        consistency = _cmmt_consistency(
                            student,
                            teacher,
                            image,
                            float(mt["confidence_threshold"]),
                            float(mt.get("noise_std", 0.1)),
                            float(mt.get("cutmix_probability", 0.5)),
                            float(mt.get("vat_xi", 1e-3)),
                            float(mt.get("vat_epsilon", 1.0)),
                            float(mt.get("vat_weight", 0.1)),
                        )
                    else:
                        raise ValueError(f"Unknown mean_teacher.mode {mode!r}")
                    total_loss = total_loss + (
                        float(mt["weight"])
                        * linear_ramp(epoch, int(mt["ramp_epochs"]))
                        * consistency
                    )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
            if total_loss.requires_grad:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(config["optimizer"]["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
                update_ema(student, teacher, float(config["mean_teacher"]["ema_decay"]))
                updates += 1
            epoch_sums["total"] += float(total_loss.detach())
            epoch_sums["supervised"] += float(supervised.detach())
            epoch_sums["band"] += float(band.detach())
            epoch_sums["consistency"] += float(consistency.detach())
            heartbeat.maybe(f"run={run_name} epoch={epoch + 1}/{epochs} batch={batch_index + 1}/{len(loader)}")

        tkv_value = 0.0
        if bool(config["repeat_tkv"]["enabled"]):
            tkv = config["repeat_tkv"]
            repeat_subjects = sorted(repeat_groups)
            for _ in range(int(tkv["steps_per_epoch"])):
                subject_id = random.choice(repeat_subjects)
                record_a, record_b = random.sample(repeat_groups[subject_id], 2)
                optimizer.zero_grad(set_to_none=True)
                volume_a = _soft_volume(
                    student,
                    store,
                    record_a,
                    device,
                    int(config["optimizer"]["batch_size"]),
                    amp,
                    stack_depth,
                )
                volume_b = _soft_volume(
                    student,
                    store,
                    record_b,
                    device,
                    int(config["optimizer"]["batch_size"]),
                    amp,
                    stack_depth,
                )
                with _autocast(device, amp):
                    repeat_loss = repeat_tkv_loss(
                        volume_a, volume_b, float(tkv["epsilon_ml"])
                    )
                    weighted_repeat = (
                        float(tkv["weight"])
                        * linear_ramp(epoch, int(tkv["ramp_epochs"]))
                        * repeat_loss
                    )
                if not torch.isfinite(weighted_repeat):
                    raise FloatingPointError(f"Non-finite TKV loss at epoch {epoch}")
                scaler.scale(weighted_repeat).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(config["optimizer"]["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
                update_ema(student, teacher, float(config["mean_teacher"]["ema_decay"]))
                tkv_value += float(repeat_loss.detach())

        denominator = max(updates, 1)
        record = {
            "epoch": epoch + 1,
            "loss_total": epoch_sums["total"] / denominator,
            "loss_supervised": epoch_sums["supervised"] / denominator,
            "loss_band": epoch_sums["band"] / denominator,
            "loss_consistency": epoch_sums["consistency"] / denominator,
            "loss_tkv": tkv_value / max(int(config["repeat_tkv"]["steps_per_epoch"]), 1),
            "updates": updates,
            "elapsed_seconds": time.time() - started,
        }
        _append_jsonl(metrics_path, record)
        print(json.dumps(record, allow_nan=False), flush=True)
        torch.save(
            {"model": student.state_dict(), "epoch": epoch + 1, "seed": seed, "fold": fold},
            run_dir / "last_student.pt",
        )
        torch.save(
            {"model": teacher.state_dict(), "epoch": epoch + 1, "seed": seed, "fold": fold},
            run_dir / "last_teacher.pt",
        )

    result = {
        "status": "completed",
        "run_name": run_name,
        "fold": fold,
        "seed": seed,
        "epochs": epochs,
        "duration_seconds": time.time() - started,
        "device": str(device),
        "torch_version": torch.__version__,
        "labeled_subjects": len(labeled_subjects),
        "training_subjects": int(train_records.subject_id.nunique()),
        "model_name": str(config["model"].get("name", "monai_unet")),
        "trainable_parameters": int(sum(value.numel() for value in student.parameters() if value.requires_grad)),
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
        ),
    }
    atomic_json(result, run_dir / "training_result.json")
    (run_dir / "EXPERIMENT_RESULT.md").write_text(
        "\n".join(
            [
                "## Material Passport",
                "",
                "- Origin Skill: experiment-agent",
                "- Origin Mode: run",
                f"- Origin Date: {time.strftime('%Y-%m-%d')}",
                "- Verification Status: UNVERIFIED",
                "- Version Label: exp_result_v1",
                "",
                "## Experiment Result",
                "",
                f"- **ID**: {run_name}-fold{fold}-seed{seed}",
                "- **Type**: training",
                "- **Status**: completed",
                f"- **Working Directory**: {run_dir}",
                f"- **Duration**: {result['duration_seconds']:.1f} seconds",
                "- **Exit Code**: 0",
                "",
                "### Anomalies Detected",
                "",
                "None automatically detected. Statistical findings remain unverified pilot outputs.",
            ]
        ),
        encoding="utf-8",
    )
    return run_dir

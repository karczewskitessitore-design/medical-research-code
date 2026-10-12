from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from kidney_ssl.benchmark import _run_command
from kidney_ssl.config import dump_config, load_config
from kidney_ssl.data.folds import make_folds, validate_folds
from kidney_ssl.external_evaluation import evaluate_external
from kidney_ssl.nnunet_adapter import evaluate_nnunet_external
from kidney_ssl.protocol_v2 import validate_protocol_lock
from kidney_ssl.utils import atomic_json


def _suite(path: Path) -> dict[str, Any]:
    suite = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if int(suite.get("schema_version", 0)) != 2:
        raise ValueError("benchmark_v2 requires suite schema_version: 2")
    if not {"name", "output_root", "methods", "phases", "proposed_method"}.issubset(suite):
        raise ValueError("Incomplete v2 benchmark suite")
    return suite


def _project_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Could not locate pyproject.toml")


def _methods_for(suite: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    return [
        method
        for method in suite["methods"]
        if phase in method.get("phases", ["screen", "internal", "external"])
    ]


def _selected_methods(
    suite: dict[str, Any], phase: str, method_ids: list[str] | None
) -> list[dict[str, Any]]:
    methods = _methods_for(suite, phase)
    if not method_ids:
        return methods
    requested = list(dict.fromkeys(str(item) for item in method_ids))
    available = {str(method["id"]) for method in methods}
    missing = [item for item in requested if item not in available]
    if missing:
        raise ValueError(
            f"Methods are unavailable for phase {phase}: {missing}; "
            f"available={sorted(available)}"
        )
    requested_set = set(requested)
    return [method for method in methods if str(method["id"]) in requested_set]


def _run_table_path(root: Path, phase: str, method_ids: list[str] | None) -> Path:
    if not method_ids:
        return root / f"{phase}_runs.csv"
    safe = "-".join(
        "".join(character if character.isalnum() else "_" for character in method)
        for method in method_ids
    )
    return root / f"{phase}_runs__{safe}.csv"


def _method_config(
    method: dict[str, Any], suite_path: Path, suite: dict[str, Any]
) -> dict[str, Any]:
    source = (suite_path.parent / str(method["config"])).resolve()
    overrides = suite.get("config_overrides")
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("suite.config_overrides must be a mapping")
    return load_config(source, overrides=overrides)


def _prepare_splits(config: dict[str, Any], split_seed: int) -> Path:
    manifest = Path(config["data"]["manifest"]).resolve()
    split = manifest.parent.parent / "splits" / f"seed_{split_seed}"
    metadata = split / "split_metadata.json"
    if not metadata.exists() or 0.125 not in json.loads(metadata.read_text(encoding="utf-8")).get(
        "label_fractions", []
    ):
        make_folds(manifest, split_seed)
    validate_folds(split)
    return split


def _resolved_config(
    method: dict[str, Any],
    suite_path: Path,
    suite: dict[str, Any],
    phase_root: Path,
    split: Path,
    resume: bool,
) -> Path:
    config = _method_config(method, suite_path, suite)
    config["data"]["folds"] = str(split)
    config["runtime"]["output_root"] = str(phase_root)
    config["runtime"]["run_name"] = str(method["id"])
    config["runtime"]["phase"] = phase_root.name
    config["runtime"]["resume"] = bool(resume)
    output = phase_root / "resolved_configs" / f"{method['id']}.yaml"
    dump_config(config, output)
    return output


def _execute(
    project_root: Path,
    config_path: Path,
    run_dir: Path,
    fold: int,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    if resume and (run_dir / "evaluation_summary.json").is_file():
        return {"status": "skipped_completed", "run_dir": str(run_dir)}
    prior_status = run_dir / "benchmark_status.json"
    prior_log = run_dir / "benchmark_train.log"
    if resume and (prior_status.is_file() or prior_log.is_file()):
        attempts = run_dir / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        attempt_number = len(list(attempts.glob("attempt_*_train.log"))) + 1
        if prior_status.is_file():
            shutil.copy2(prior_status, attempts / f"attempt_{attempt_number:02d}_status.json")
        if prior_log.is_file():
            shutil.copy2(prior_log, attempts / f"attempt_{attempt_number:02d}_train.log")
    config = load_config(config_path)
    timeout = float(config["runtime"]["timeout_hours"]) * 3600
    stall = float(config["runtime"]["stall_minutes"]) * 60
    train = _run_command(
        [
            sys.executable,
            "-m",
            "kidney_ssl.cli",
            "train",
            "--config",
            str(config_path),
            "--fold",
            str(fold),
            "--seed",
            str(seed),
        ],
        project_root,
        run_dir / "benchmark_train.log",
        timeout,
        stall,
    )
    if train["status"] != "completed":
        atomic_json(train, run_dir / "benchmark_status.json")
        return train
    if str(config["model"].get("trainer")) == "nnunetv2":
        evaluation = {"status": "completed", "performed_by_adapter": True}
    else:
        evaluation = _run_command(
            [
                sys.executable,
                "-m",
                "kidney_ssl.cli",
                "evaluate",
                "--run-dir",
                str(run_dir),
            ],
            project_root,
            run_dir / "benchmark_evaluate.log",
            timeout,
            stall,
        )
    status = {
        "status": "completed" if evaluation["status"] == "completed" else "evaluation_failed",
        "train": train,
        "evaluation": evaluation,
        "run_dir": str(run_dir),
    }
    atomic_json(status, run_dir / "benchmark_status.json")
    return status


def _screen_decision(root: Path, suite: dict[str, Any]) -> Path:
    rows = []
    for method in _methods_for(suite, "screen"):
        summaries = []
        for path in sorted((root / str(method["id"])).glob("fold*_seed*/evaluation_summary.json")):
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        rows.append(
            {
                "method": str(method["id"]),
                "runs": len(summaries),
                "dice": np.mean([item["overall"]["dice"] for item in summaries]) if summaries else None,
                "hd95_mm": np.mean([item["overall"]["hd95_mm"] for item in summaries]) if summaries else None,
                "repeat_wcv_percent": np.mean(
                    [
                        item.get("repeatability", {}).get("within_subject_cv_percent_mean")
                        for item in summaries
                        if item.get("repeatability", {}).get("within_subject_cv_percent_mean") is not None
                    ]
                )
                if any(
                    item.get("repeatability", {}).get("within_subject_cv_percent_mean") is not None
                    for item in summaries
                )
                else None,
            }
        )
    table = pd.DataFrame(rows).set_index("method", drop=False)
    table.to_csv(root / "screening_summary.csv", index=False)
    rules = suite["selection_rules"]
    selection = suite.get("screen_selection") or {}
    baseline_id = str(selection.get("baseline_method", "V2_B0plus"))
    segmentation_modules = selection.get(
        "segmentation_modules", {"UABand": "V2_UABand", "CSD": "V2_CSD"}
    )
    repeat_module = selection.get(
        "repeat_module", {"name": "DZ-BiMorph", "method": "V2_DZ"}
    )
    full_method = str(selection.get("full_method", "KidneyBand_ReliableSSL_v2"))
    default_full_references = ["V2_UABand_CSD", "V2_UABand_DZ", "V2_CSD_DZ"]
    full_reference_methods = [
        str(item)
        for item in selection.get("full_reference_methods", default_full_references)
    ]
    expected_runs = len(suite["phases"]["screen"]["folds"]) * len(
        suite["phases"]["screen"]["seeds"]
    )
    complete = bool(len(table)) and bool((table.runs == expected_runs).all())
    baseline = table.loc[baseline_id] if baseline_id in table.index else None
    decisions: dict[str, Any] = {}
    for module, method_value in segmentation_modules.items():
        method = str(method_value)
        available = baseline is not None and method in table.index and pd.notna(table.loc[method, "dice"])
        decisions[module] = {
            "keep": bool(
                available
                and table.loc[method, "dice"] - baseline.dice
                >= float(rules["segmentation_min_dice_gain"])
                and table.loc[method, "hd95_mm"] - baseline.hd95_mm
                <= float(rules["segmentation_max_hd95_degradation_mm"])
            ),
            "dice_gain": float(table.loc[method, "dice"] - baseline.dice) if available else None,
            "hd95_change_mm": float(table.loc[method, "hd95_mm"] - baseline.hd95_mm)
            if available
            else None,
        }
    if repeat_module is not None:
        repeat_name = str(repeat_module.get("name", "DZ-BiMorph"))
        repeat_method = str(repeat_module.get("method", "V2_DZ"))
        repeat_available = (
            baseline is not None
            and repeat_method in table.index
            and pd.notna(table.loc[repeat_method, "repeat_wcv_percent"])
            and pd.notna(baseline.repeat_wcv_percent)
        )
        decisions[repeat_name] = {
            "keep": bool(
                repeat_available
                and baseline.repeat_wcv_percent
                - table.loc[repeat_method, "repeat_wcv_percent"]
                >= float(rules["repeat_min_wcv_gain_percent"])
                and table.loc[repeat_method, "dice"] - baseline.dice
                >= -float(rules["repeat_max_dice_degradation"])
            ),
            "wcv_gain_percent_points": (
                float(
                    baseline.repeat_wcv_percent
                    - table.loc[repeat_method, "repeat_wcv_percent"]
                )
                if repeat_available
                else None
            ),
            "dice_change": (
                float(table.loc[repeat_method, "dice"] - baseline.dice)
                if repeat_available
                else None
            ),
        }
    references = table.loc[
        [item for item in full_reference_methods if item in table.index]
    ].dropna(subset=["dice"])
    full_available = full_method in table.index and not references.empty
    full_delta = (
        float(table.loc[full_method, "dice"] - references.dice.max())
        if full_available
        else None
    )
    decisions["full_model"] = {
        "keep": bool(
            full_available and full_delta >= -float(rules["full_max_degradation_from_best_pair"])
        ),
        "difference_from_best_pair": full_delta,
    }
    depth_choice = selection.get("stack_depth")
    if depth_choice is None and {"V2_B0plus", "V2_B0plus_5slice"}.issubset(table.index):
        gain = float(
            table.loc["V2_B0plus_5slice", "dice"] - table.loc["V2_B0plus", "dice"]
        )
        depth_choice = 3 if abs(gain) < 0.002 else (5 if gain > 0 else 3)
    required_modules = [str(item) for item in selection.get("required_modules", [])]
    required_modules_kept = all(
        item in decisions and bool(decisions[item]["keep"]) for item in required_modules
    )
    formal_method_ready = bool(
        complete and decisions["full_model"]["keep"] and required_modules_kept
    )
    result = {
        "status": "complete" if complete else "incomplete",
        "expected_runs_per_method": expected_runs,
        "stack_depth_selected": depth_choice,
        "selection_baseline": baseline_id,
        "full_reference_methods": full_reference_methods,
        "module_decisions": decisions,
        "formal_method_ready": formal_method_ready,
        "next_step": (
            "freeze the retained-module configuration, then run lock-protocol"
            if formal_method_ready
            else (
                "screening is complete but the formal method is not ready; revise a versioned development suite"
                if complete
                else "complete every preregistered screening run; do not infer module retention"
            )
        ),
    }
    atomic_json(result, root / "screening_decision.json")
    return root / "screening_decision.json"


def _aggregate_external_seeds(method_root: Path, destination: Path) -> None:
    tables = []
    for path in sorted(method_root.glob("seed*/metrics_subject.csv")):
        table = pd.read_csv(path)
        table["seed"] = int(path.parent.name.removeprefix("seed"))
        tables.append(table)
    if not tables:
        return
    raw = pd.concat(tables, ignore_index=True)
    numeric = [column for column in raw.select_dtypes(include=[np.number]) if column != "seed"]
    result = raw.groupby(["subject_id", "cohort", "is_repeat"], as_index=False)[numeric].mean()
    destination.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination / "metrics_subject.csv", index=False)


def _method_mean_dice(internal_root: Path, method_id: str) -> float:
    values = []
    for path in (internal_root / method_id).glob("fold*_seed*/evaluation_summary.json"):
        values.append(float(json.loads(path.read_text(encoding="utf-8"))["overall"]["dice"]))
    return float(np.mean(values)) if values else float("-inf")


def _run_label_efficiency(
    suite: dict[str, Any],
    suite_path: Path,
    suite_root: Path,
    project_root: Path,
    split: Path,
    resume: bool,
) -> Path:
    internal_root = suite_root / "internal"
    methods_by_id = {str(method["id"]): method for method in suite["methods"]}
    strong_ids = [
        str(method["id"])
        for method in suite["methods"]
        if bool(method.get("strong_semisupervised", False))
    ]
    strongest_semi = max(strong_ids, key=lambda value: _method_mean_dice(internal_root, value))
    selected_ids = [
        str(suite.get("matched_baseline", "V2_B0plus")),
        "nnunet_3d_default",
        strongest_semi,
        str(suite["proposed_method"]),
    ]
    phase = suite["phases"]["internal"]
    seed = int(phase["seeds"][0])
    root = suite_root / "label_efficiency"
    statuses = []
    tokens = {0.125: "12p5", 0.5: "50", 1.0: "100"}
    for fraction, token in tokens.items():
        fraction_root = root / token
        for method_id in selected_ids:
            method = methods_by_id[method_id]
            source = (suite_path.parent / str(method["config"])).resolve()
            config = load_config(source, overrides=suite.get("config_overrides"))
            config["data"]["folds"] = str(split)
            config["split"]["labeled_fraction"] = fraction
            config["runtime"]["output_root"] = str(fraction_root)
            config["runtime"]["run_name"] = method_id
            config["runtime"]["resume"] = bool(resume)
            config_path = root / "resolved_configs" / f"{method_id}_{token}.yaml"
            dump_config(config, config_path)
            for fold in phase["folds"]:
                run_dir = fraction_root / method_id / f"fold{fold}_seed{seed}"
                status = _execute(
                    project_root, config_path, run_dir, int(fold), seed, resume
                )
                statuses.append(
                    {
                        "fraction": fraction,
                        "method": method_id,
                        "fold": int(fold),
                        "seed": seed,
                        "status": status["status"],
                        "run_dir": str(run_dir),
                    }
                )
    table = pd.DataFrame(statuses)
    table.to_csv(root / "label_efficiency_runs.csv", index=False)
    proposed = str(suite["proposed_method"])
    summaries = []
    for fraction, token in tokens.items():
        dice = []
        for path in (root / token / proposed).glob("fold*_seed*/evaluation_summary.json"):
            dice.append(float(json.loads(path.read_text(encoding="utf-8"))["overall"]["dice"]))
        summaries.append({"labeled_fraction": fraction, "proposed_mean_dice": float(np.mean(dice)) if dice else None})
    pd.DataFrame(summaries).to_csv(root / "label_efficiency_summary.csv", index=False)
    full = next(item for item in summaries if item["labeled_fraction"] == 1.0)
    atomic_json(full, root / "100pct_summary.json")
    return root / "label_efficiency_runs.csv"


def run_benchmark_v2(
    suite_path: str | Path,
    phase: str,
    resume: bool = False,
    project_root: str | Path | None = None,
    method_ids: list[str] | None = None,
) -> Path:
    if phase not in {"screen", "internal", "external"}:
        raise ValueError("phase must be screen, internal, or external")
    suite_path = Path(suite_path).resolve()
    suite = _suite(suite_path)
    project_root = Path(project_root).resolve() if project_root else _project_root(suite_path)
    suite_root = Path(suite["output_root"]).resolve() / str(suite["name"])
    phase_root = suite_root / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(suite_path, phase_root / "suite.yaml")
    if phase == "screen" and method_ids:
        raise ValueError("Method filtering is prohibited for screening decisions")
    methods = _selected_methods(suite, phase, method_ids)
    if phase in {"internal", "external"}:
        validate_protocol_lock(suite_path)
    if phase == "external":
        statuses = []
        internal_root = suite_root / "internal"
        for dataset in suite["phases"]["external"]["datasets"]:
            for method in methods:
                method_id = str(method["id"])
                for seed in suite["phases"]["external"]["seeds"]:
                    output = phase_root / dataset / method_id / f"seed{seed}"
                    if resume and (output / "evaluation_summary.json").is_file():
                        status = "skipped_completed"
                    else:
                        try:
                            if method_id == "nnunet_3d_default":
                                evaluate_nnunet_external(
                                    internal_root / method_id, dataset, int(seed), output
                                )
                            else:
                                evaluate_external(
                                    internal_root / method_id,
                                    dataset,
                                    int(seed),
                                    output,
                                )
                            status = "completed"
                        except Exception as exc:
                            status = "failed"
                            atomic_json(
                                {"status": status, "error": repr(exc)}, output / "benchmark_status.json"
                            )
                    statuses.append(
                        {"dataset": dataset, "method": method_id, "seed": seed, "status": status}
                    )
                _aggregate_external_seeds(
                    phase_root / dataset / method_id,
                    phase_root / dataset / method_id,
                )
        runs_path = _run_table_path(phase_root, phase, method_ids)
        pd.DataFrame(statuses).to_csv(runs_path, index=False)
        return runs_path

    first_method = methods[0]
    first_config = _method_config(first_method, suite_path, suite)
    phase_config = suite["phases"][phase]
    split = _prepare_splits(first_config, int(phase_config["split_seed"]))
    statuses = []
    for method in methods:
        config_path = _resolved_config(method, suite_path, suite, phase_root, split, resume)
        for fold in phase_config["folds"]:
            for seed in phase_config["seeds"]:
                run_dir = phase_root / str(method["id"]) / f"fold{fold}_seed{seed}"
                status = _execute(
                    project_root, config_path, run_dir, int(fold), int(seed), resume
                )
                statuses.append(
                    {
                        "method": method["id"],
                        "fold": int(fold),
                        "seed": int(seed),
                        "status": status["status"],
                        "run_dir": str(run_dir),
                    }
                )
    runs_path = _run_table_path(phase_root, phase, method_ids)
    pd.DataFrame(statuses).to_csv(runs_path, index=False)
    if phase == "screen":
        return _screen_decision(phase_root, suite)
    label_efficiency_enabled = bool(
        suite.get("label_efficiency", {}).get("enabled", True)
    )
    if (
        not method_ids
        and label_efficiency_enabled
        and set(item["status"] for item in statuses).issubset(
            {"completed", "skipped_completed"}
        )
    ):
        return _run_label_efficiency(
            suite, suite_path, suite_root, project_root, split, resume
        )
    return runs_path

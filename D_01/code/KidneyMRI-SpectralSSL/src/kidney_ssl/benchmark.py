from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from kidney_ssl.comparison import compare_benchmark
from kidney_ssl.config import dump_config, load_config
from kidney_ssl.data.folds import make_folds, validate_folds
from kidney_ssl.utils import atomic_json, sha256_file


def _load_suite(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        suite = yaml.safe_load(handle) or {}
    required = {"name", "output_root", "methods", "phases", "proposed_method"}
    missing = required - set(suite)
    if missing:
        raise ValueError(f"Benchmark suite missing keys: {sorted(missing)}")
    ids = [str(method["id"]) for method in suite["methods"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark method ids must be unique")
    if str(suite["proposed_method"]) not in ids:
        raise ValueError("proposed_method must refer to a configured method")
    return suite


def _find_project_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate pyproject.toml above {path}")


def _float_token(value: float) -> str:
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


def _config_hash(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_tree_hash(project_root: Path) -> tuple[str, list[dict[str, str]]]:
    candidates = [project_root / "pyproject.toml", project_root / "uv.lock"]
    candidates.extend(sorted((project_root / "src").rglob("*.py")))
    candidates.extend(sorted((project_root / "configs").rglob("*.yaml")))
    inventory = [
        {"path": path.relative_to(project_root).as_posix(), "sha256": sha256_file(path)}
        for path in candidates
        if path.is_file()
    ]
    payload = json.dumps(inventory, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), inventory


def _selected_definition_hash(
    method: dict[str, Any], suite_path: Path, learning_rate: float, tkv_weight: float | None
) -> str:
    source = (suite_path.parent / str(method["config"])).resolve()
    config = load_config(source)
    config["optimizer"]["learning_rate"] = float(learning_rate)
    if tkv_weight is not None:
        config["repeat_tkv"]["weight"] = float(tkv_weight)
    return _config_hash(config)


def _run_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    timeout_seconds: float,
    stall_seconds: float,
    check_seconds: float = 10.0,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        status = "running"
        while process.poll() is None:
            time.sleep(check_seconds)
            elapsed = time.time() - started
            age = time.time() - log_path.stat().st_mtime
            print(
                f"BENCHMARK_MONITOR pid={process.pid} elapsed={elapsed:.0f}s last_output={age:.0f}s",
                flush=True,
            )
            if elapsed > timeout_seconds:
                status = "timeout"
                _terminate_process_tree(process)
                break
            if age > stall_seconds:
                status = "stalled"
                _terminate_process_tree(process)
                break
        exit_code = process.wait()
    if status == "running":
        status = "completed" if exit_code == 0 else "failed"
    return {
        "status": status,
        "exit_code": int(exit_code),
        "duration_seconds": time.time() - started,
        "command": command,
        "log": str(log_path),
    }


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def _execute_run(
    project_root: Path,
    config_path: Path,
    run_dir: Path,
    fold: int,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    completion = run_dir / "evaluation_summary.json"
    if resume and completion.exists():
        return {"status": "skipped_completed", "run_dir": str(run_dir)}
    config = load_config(config_path)
    timeout = float(config["runtime"]["timeout_hours"]) * 3600
    stall = float(config["runtime"]["stall_minutes"]) * 60
    train_command = [
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
    ]
    train_status = _run_command(
        train_command, project_root, run_dir / "benchmark_train.log", timeout, stall
    )
    if train_status["status"] != "completed":
        atomic_json(train_status, run_dir / "benchmark_status.json")
        return train_status
    if str(config["model"].get("trainer", "native")) == "nnunetv2":
        # The adapter performs inference and writes the common metric files.
        evaluation_status = {"status": "completed", "external_adapter": True}
    else:
        evaluate_command = [
            sys.executable,
            "-m",
            "kidney_ssl.cli",
            "evaluate",
            "--run-dir",
            str(run_dir),
        ]
        evaluation_status = _run_command(
            evaluate_command,
            project_root,
            run_dir / "benchmark_evaluate.log",
            timeout,
            stall,
        )
    status = {
        "status": "completed" if evaluation_status["status"] == "completed" else "evaluation_failed",
        "train": train_status,
        "evaluation": evaluation_status,
        "run_dir": str(run_dir),
    }
    atomic_json(status, run_dir / "benchmark_status.json")
    return status


def _prepare_config(
    method: dict[str, Any],
    suite_path: Path,
    phase_root: Path,
    folds_path: Path,
    phase: str,
    run_name: str,
    learning_rate: float,
    tkv_weight: float | None,
) -> tuple[dict[str, Any], Path]:
    source = (suite_path.parent / str(method["config"])).resolve()
    config = load_config(source)
    config["data"]["folds"] = str(folds_path.resolve())
    config["optimizer"]["learning_rate"] = float(learning_rate)
    if tkv_weight is not None:
        config["repeat_tkv"]["weight"] = float(tkv_weight)
    config["runtime"]["output_root"] = str(phase_root.resolve())
    config["runtime"]["run_name"] = run_name
    config["runtime"]["phase"] = phase
    config["runtime"]["source_config"] = str(source)
    config["runtime"]["config_sha256"] = _config_hash(config)
    destination = phase_root / "resolved_configs" / f"{run_name}.yaml"
    dump_config(config, destination)
    return config, destination


def _dev_variants(method: dict[str, Any], phase: dict[str, Any]) -> list[tuple[float, float | None]]:
    learning_rates = method.get("learning_rates", phase.get("learning_rates", [0.001]))
    tkv_weights = method.get("tkv_weights", [None])
    return [(float(lr), None if weight is None else float(weight)) for lr in learning_rates for weight in tkv_weights]


def _select_dev(root: Path, method: dict[str, Any], variants: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for variant in variants:
        summary = Path(variant["run_dir"]) / "evaluation_summary.json"
        if variant["status"] not in {"completed", "skipped_completed"} or not summary.exists():
            continue
        values = json.loads(summary.read_text(encoding="utf-8"))
        candidates.append(
            {
                **variant,
                "dice": float(values["overall"]["dice"]),
                "tkv_relative_error": float(values["overall"]["tkv_relative_error"]),
            }
        )
    if not candidates:
        raise RuntimeError(f"No successful development run for {method['id']}")
    best_dice = max(item["dice"] for item in candidates)
    near_best = [item for item in candidates if item["dice"] >= best_dice - 0.005]
    if bool(method.get("tkv_lexicographic", False)):
        selected = min(near_best, key=lambda item: (item["tkv_relative_error"], -item["dice"]))
    else:
        selected = max(candidates, key=lambda item: (item["dice"], -item["tkv_relative_error"]))
    pd.DataFrame(candidates).to_csv(root / f"dev_selection_{method['id']}.csv", index=False)
    return selected


def run_benchmark(
    suite_path: str | Path,
    phase: str,
    resume: bool = False,
    project_root: str | Path | None = None,
) -> Path:
    suite_path = Path(suite_path).resolve()
    suite = _load_suite(suite_path)
    if phase not in {"dev", "confirmatory"}:
        raise ValueError("phase must be dev or confirmatory")
    project_root = Path(project_root).resolve() if project_root else _find_project_root(suite_path)
    output_root = Path(suite["output_root"]).resolve()
    suite_root = output_root / str(suite["name"])
    phase_root = suite_root / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(suite_path, phase_root / "suite.yaml")
    phase_config = suite["phases"][phase]
    split_seed = int(phase_config["split_seed"])
    first_config = load_config((suite_path.parent / str(suite["methods"][0]["config"])).resolve())
    manifest = Path(first_config["data"]["manifest"]).resolve()
    folds_path = manifest.parent.parent / "splits" / f"seed_{split_seed}"
    if not (folds_path / "split_metadata.json").exists():
        make_folds(manifest, split_seed)
    validate_folds(folds_path)

    if phase == "dev":
        selections = {}
        all_status = []
        fold = int(phase_config.get("fold", 0))
        seed = int(phase_config.get("seed", split_seed))
        for method in suite["methods"]:
            if phase not in method.get("phases", ["dev", "confirmatory"]):
                continue
            method_variants = []
            for learning_rate, tkv_weight in _dev_variants(method, phase_config):
                suffix = f"lr{_float_token(learning_rate)}"
                if tkv_weight is not None:
                    suffix += f"_tkv{_float_token(tkv_weight)}"
                run_name = f"{method['id']}__{suffix}"
                _, config_path = _prepare_config(
                    method,
                    suite_path,
                    phase_root,
                    folds_path,
                    phase,
                    run_name,
                    learning_rate,
                    tkv_weight,
                )
                run_dir = phase_root / run_name / f"fold{fold}_seed{seed}"
                status = _execute_run(project_root, config_path, run_dir, fold, seed, resume)
                variant = {
                    "method": method["id"],
                    "run_name": run_name,
                    "learning_rate": learning_rate,
                    "tkv_weight": tkv_weight,
                    "run_dir": str(run_dir),
                    "status": status["status"],
                }
                method_variants.append(variant)
                all_status.append(variant)
            selections[str(method["id"])] = _select_dev(phase_root, method, method_variants)
        # Methods with an externally prescribed schedule (for example the official
        # nnU-Net schedule) do not need a development run.  They are still frozen
        # into the protocol lock so the confirmatory phase cannot alter them.
        for method in suite["methods"]:
            method_id = str(method["id"])
            if method_id in selections or "confirmatory" not in method.get(
                "phases", ["dev", "confirmatory"]
            ):
                continue
            if "fixed_learning_rate" not in method:
                raise ValueError(
                    f"Confirmatory-only method {method_id} requires fixed_learning_rate"
                )
            selections[method_id] = {
                "learning_rate": float(method["fixed_learning_rate"]),
                "tkv_weight": method.get("fixed_tkv_weight"),
                "dice": None,
                "tkv_relative_error": None,
                "selection_basis": "prespecified_external_schedule",
            }
        source_tree_sha256, source_inventory = _source_tree_hash(project_root)
        methods_by_id = {str(method["id"]): method for method in suite["methods"]}
        lock = {
            "suite": str(suite_path),
            "suite_sha256": sha256_file(suite_path),
            "development_split_seed": split_seed,
            "formal_split_seed": int(suite["phases"]["confirmatory"]["split_seed"]),
            "formal_folds": list(suite["phases"]["confirmatory"]["folds"]),
            "formal_seeds": list(suite["phases"]["confirmatory"]["seeds"]),
            "source_tree_sha256": source_tree_sha256,
            "source_inventory": source_inventory,
            "selected": {
                method: {
                    "learning_rate": values["learning_rate"],
                    "tkv_weight": values["tkv_weight"],
                    "development_dice": values["dice"],
                    "development_tkv_relative_error": values["tkv_relative_error"],
                    "selection_basis": values.get("selection_basis", "development_fold"),
                    "configuration_sha256": _selected_definition_hash(
                        methods_by_id[method],
                        suite_path,
                        values["learning_rate"],
                        values["tkv_weight"],
                    ),
                }
                for method, values in selections.items()
            },
        }
        lock["lock_sha256"] = hashlib.sha256(
            json.dumps(lock, sort_keys=True).encode("utf-8")
        ).hexdigest()
        atomic_json(lock, suite_root / "protocol_lock.json")
        pd.DataFrame(all_status).to_csv(phase_root / "dev_runs.csv", index=False)
        return suite_root / "protocol_lock.json"

    lock_path = suite_root / "protocol_lock.json"
    if not lock_path.exists():
        raise RuntimeError("Development gate has not produced protocol_lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["suite_sha256"] != sha256_file(suite_path):
        raise RuntimeError("Suite YAML changed after protocol lock; rerun the development gate")
    current_source_sha256, _ = _source_tree_hash(project_root)
    if lock["source_tree_sha256"] != current_source_sha256:
        raise RuntimeError(
            "Source/configuration tree changed after protocol lock; rerun the development gate"
        )
    statuses = []
    for method in suite["methods"]:
        if phase not in method.get("phases", ["dev", "confirmatory"]):
            continue
        selected = lock["selected"][str(method["id"])]
        _, config_path = _prepare_config(
            method,
            suite_path,
            phase_root,
            folds_path,
            phase,
            str(method["id"]),
            float(selected["learning_rate"]),
            selected["tkv_weight"],
        )
        for fold in phase_config["folds"]:
            for seed in phase_config["seeds"]:
                run_dir = phase_root / str(method["id"]) / f"fold{int(fold)}_seed{int(seed)}"
                status = _execute_run(
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
    status_table = pd.DataFrame(statuses)
    status_table.to_csv(phase_root / "confirmatory_runs.csv", index=False)
    if not set(status_table.status).issubset({"completed", "skipped_completed"}):
        return phase_root / "confirmatory_runs.csv"
    return compare_benchmark(phase_root, phase_root / "suite.yaml")

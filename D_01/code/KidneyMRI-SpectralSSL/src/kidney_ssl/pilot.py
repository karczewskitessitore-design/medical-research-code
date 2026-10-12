from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from kidney_ssl.config import load_config
from kidney_ssl.utils import atomic_json


def _run_one(config_path: Path, fold: int, seed: int) -> dict:
    config = load_config(config_path)
    run_name = config["runtime"]["run_name"]
    run_dir = (
        Path(config["runtime"]["output_root"]).resolve()
        / run_name
        / f"fold{fold}_seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pilot_process.log"
    command = [
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
    timeout_seconds = float(config["runtime"]["timeout_hours"]) * 3600
    stall_seconds = float(config["runtime"]["stall_minutes"]) * 60
    check_seconds = float(config["runtime"]["heartbeat_seconds"])
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=config_path.parents[2])
        status = "running"
        while process.poll() is None:
            time.sleep(check_seconds)
            elapsed = time.time() - started
            last_output = time.time() - log_path.stat().st_mtime
            print(
                f"PILOT_MONITOR run={run_name} pid={process.pid} elapsed={elapsed:.0f}s "
                f"last_output={last_output:.0f}s",
                flush=True,
            )
            if elapsed > timeout_seconds:
                status = "timeout"
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            if last_output > stall_seconds:
                status = "stalled"
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
        exit_code = process.wait()
    if status == "running":
        status = "completed" if exit_code == 0 else "failed"
    result = {
        "run": run_name,
        "status": status,
        "exit_code": exit_code,
        "duration_seconds": time.time() - started,
        "command": command,
        "log": str(log_path),
    }
    if status == "completed":
        evaluation_command = [
            sys.executable,
            "-m",
            "kidney_ssl.cli",
            "evaluate",
            "--run-dir",
            str(run_dir),
        ]
        evaluation = subprocess.run(
            evaluation_command,
            cwd=config_path.parents[2],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        (run_dir / "evaluation_process.log").write_text(
            evaluation.stdout + evaluation.stderr, encoding="utf-8"
        )
        result["evaluation_exit_code"] = evaluation.returncode
        if evaluation.returncode != 0:
            result["status"] = "evaluation_failed"
    atomic_json(result, run_dir / "pilot_status.json")
    return result


def _report(output_root: Path, results: list[dict], fold: int, seed: int) -> Path:
    report_dir = output_root / "pilot_summary" / f"fold{fold}_seed{seed}"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    subject_tables: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(seed)

    def mean_ci(values: np.ndarray, draws: int = 10000) -> tuple[float, float]:
        values = np.asarray(values, dtype=float)
        bootstrap = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
        return float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))

    for result in results:
        run_dir = output_root / result["run"] / f"fold{fold}_seed{seed}"
        summary_path = run_dir / "evaluation_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        subject_path = run_dir / "metrics_subject.csv"
        if subject_path.exists():
            subject_tables[result["run"]] = pd.read_csv(subject_path).set_index("subject_id").sort_index()
        table = subject_tables.get(result["run"])
        dice_ci = mean_ci(table.dice.to_numpy()) if table is not None else (None, None)
        tkv_ci = mean_ci(table.tkv_relative_error.to_numpy()) if table is not None else (None, None)
        rows.append(
            {
                "run": result["run"],
                "status": result["status"],
                "dice": summary.get("overall", {}).get("dice"),
                "hd95_mm": summary.get("overall", {}).get("hd95_mm"),
                "asd_mm": summary.get("overall", {}).get("asd_mm"),
                "tkv_relative_error": summary.get("overall", {}).get("tkv_relative_error"),
                "repeat_cv_percent": summary.get("repeatability", {}).get("within_subject_cv_percent_mean"),
                "icc_2_1": summary.get("repeatability", {}).get("icc_2_1"),
                "dice_ci95_low": dice_ci[0],
                "dice_ci95_high": dice_ci[1],
                "tkv_error_ci95_low": tkv_ci[0],
                "tkv_error_ci95_high": tkv_ci[1],
            }
        )
    comparisons = []
    for previous, current in zip(("B0", "B1", "B2"), ("B1", "B2", "B3")):
        if previous not in subject_tables or current not in subject_tables:
            continue
        joined = subject_tables[previous][["dice", "tkv_relative_error"]].join(
            subject_tables[current][["dice", "tkv_relative_error"]],
            lsuffix="_previous",
            rsuffix="_current",
            how="inner",
        )
        for metric in ("dice", "tkv_relative_error"):
            difference = joined[f"{metric}_current"] - joined[f"{metric}_previous"]
            statistic, p_value = wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
            ci_low, ci_high = mean_ci(difference.to_numpy())
            comparisons.append(
                {
                    "comparison": f"{previous}_to_{current}",
                    "metric": metric,
                    "subjects": len(joined),
                    "mean_paired_difference": float(difference.mean()),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wilcoxon_statistic": float(statistic),
                    "wilcoxon_p_unadjusted": float(p_value),
                }
            )
    atomic_json(
        {"pilot_non_confirmatory": True, "runs": rows, "paired_comparisons": comparisons},
        report_dir / "summary.json",
    )
    pd.DataFrame(rows).to_csv(report_dir / "ablation_subject_ci.csv", index=False)
    pd.DataFrame(comparisons).to_csv(report_dir / "paired_subject_comparisons.csv", index=False)
    valid = [row for row in rows if row["dice"] is not None]
    if valid:
        figure, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].bar([row["run"] for row in valid], [row["dice"] for row in valid])
        axes[0].set_ylabel("Subject-mean Dice")
        axes[0].set_ylim(0, 1)
        axes[1].bar([row["run"] for row in valid], [row["tkv_relative_error"] for row in valid])
        axes[1].set_ylabel("Subject-mean TKV relative error")
        figure.suptitle("Pilot only — non-confirmatory")
        figure.tight_layout()
        figure.savefig(report_dir / "pilot_ablation.png", dpi=180)
        plt.close(figure)
    header = "| Run | Status | Dice [95% CI] | HD95 mm | ASD mm | TKV rel. error [95% CI] | Repeat CV % | ICC(2,1) |"
    lines = [
        "# KidneyBand-SSL pilot report",
        "",
        "> **Pilot、非论文确认性结果。** 单折单种子仅用于工程验证和候选方法观察。",
        "",
        header,
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        formatted = [
            row["run"],
            row["status"],
            "NA" if row["dice"] is None else f"{row['dice']:.4f} [{row['dice_ci95_low']:.4f}, {row['dice_ci95_high']:.4f}]",
            "NA" if row["hd95_mm"] is None else f"{row['hd95_mm']:.4f}",
            "NA" if row["asd_mm"] is None else f"{row['asd_mm']:.4f}",
            "NA" if row["tkv_relative_error"] is None else f"{row['tkv_relative_error']:.4f} [{row['tkv_error_ci95_low']:.4f}, {row['tkv_error_ci95_high']:.4f}]",
            "NA" if row["repeat_cv_percent"] is None else f"{row['repeat_cv_percent']:.4f}",
            "NA" if row["icc_2_1"] is None else f"{row['icc_2_1']:.4f}",
        ]
        lines.append("| " + " | ".join(formatted) + " |")
    lines.extend(
        [
            "",
            "## 相邻消融的受试者级配对比较",
            "",
            "> p 值未做多重比较校正，仅作 pilot 描述，不作为确认性推断。差值定义为后一个配置减前一个配置。",
            "",
            "| Comparison | Metric | Mean paired difference [95% bootstrap CI] | Wilcoxon p (unadjusted) |",
            "|---|---|---:|---:|",
        ]
    )
    for item in comparisons:
        lines.append(
            f"| {item['comparison']} | {item['metric']} | "
            f"{item['mean_paired_difference']:.4f} [{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] | "
            f"{item['wilcoxon_p_unadjusted']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Material Passport",
            "",
            "- Origin Skill: experiment-agent",
            "- Origin Mode: run",
            f"- Origin Date: {time.strftime('%Y-%m-%d')}",
            "- Verification Status: UNVERIFIED",
            "- Version Label: pilot_report_v1",
        ]
    )
    report_path = report_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_pilot(project_root: str | Path | None = None, fold: int = 0, seed: int = 20260802) -> Path:
    project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    config_paths = [project_root / "configs" / "pilot" / f"B{index}.yaml" for index in range(4)]
    base = load_config(config_paths[0])
    output_root = Path(base["runtime"]["output_root"]).resolve()
    results = []
    for config_path in config_paths:
        result = _run_one(config_path, fold, seed)
        results.append(result)
        if result["status"] != "completed":
            print(f"Pilot run {result['run']} failed with status {result['status']}; continuing without retry", flush=True)
    return _report(output_root, results, fold, seed)

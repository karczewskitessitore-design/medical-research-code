from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from kidney_ssl.comparison import holm_adjust
from kidney_ssl.utils import atomic_json


def _subject_tables(root: Path, method: str) -> pd.DataFrame:
    tables = []
    for path in sorted((root / method).glob("fold*_seed*/metrics_subject.csv")):
        table = pd.read_csv(path)
        run = path.parent.name
        table["seed"] = int(run.split("_seed")[-1])
        table["fold"] = int(run.split("_seed")[0].removeprefix("fold"))
        tables.append(table)
    if not tables:
        return pd.DataFrame()
    raw = pd.concat(tables, ignore_index=True)
    # Evaluation has already averaged repeated acquisitions. Average stochastic
    # initializations next so the subject remains the only independent unit.
    numeric = [
        column
        for column in raw.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "fold"}
    ]
    return raw.groupby(["subject_id", "cohort", "is_repeat"], as_index=False)[numeric].mean()


def _paired(
    proposed: pd.DataFrame,
    baseline: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    return proposed[["subject_id", "cohort", metric]].merge(
        baseline[["subject_id", metric]], on="subject_id", suffixes=("_proposed", "_baseline")
    )


def _stratified_bootstrap(values: pd.DataFrame, draws: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    difference = values["difference"].to_numpy(float)
    samples = np.zeros(draws, dtype=float)
    for _, indices in values.groupby("cohort").groups.items():
        group = values.loc[indices, "difference"].to_numpy(float)
        selected = rng.integers(0, len(group), size=(draws, len(group)))
        samples += group[selected].sum(axis=1) / len(values)
    return (
        float(difference.mean()),
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def _permutation(values: np.ndarray, draws: int, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    exceed = 0
    # Chunked generation prevents 100,000 x N sign matrices from consuming
    # unnecessary memory on the experiment workstation.
    completed = 0
    while completed < draws:
        count = min(10_000, draws - completed)
        signs = rng.choice((-1.0, 1.0), size=(count, len(values)))
        exceed += int(np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed))
        completed += count
    return float((exceed + 1) / (draws + 1))


def _comparison_rows(
    root: Path,
    suite: dict[str, Any],
    proposed_id: str,
    draws_bootstrap: int,
    draws_permutation: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    tables = {str(method["id"]): _subject_tables(root, str(method["id"])) for method in suite["methods"]}
    proposed = tables[proposed_id]
    internal_phase = suite.get("phases", {}).get("internal", {})
    expected_runs = len(internal_phase.get("folds", [])) * len(internal_phase.get("seeds", []))
    expected_subjects = int(suite.get("expected_subjects", 0))
    run_counts = {
        str(method["id"]): len(
            list((root / str(method["id"])).glob("fold*_seed*/metrics_subject.csv"))
        )
        for method in suite["methods"]
    }
    proposed_complete = bool(
        not proposed.empty
        and (not expected_runs or run_counts[proposed_id] == expected_runs)
        and (not expected_subjects or proposed.subject_id.nunique() == expected_subjects)
    )
    rows = []
    for index, method in enumerate(suite["methods"]):
        baseline_id = str(method["id"])
        if baseline_id == proposed_id or "internal" not in method.get(
            "phases", ["screen", "internal", "external"]
        ):
            continue
        baseline = tables[baseline_id]
        if proposed.empty or baseline.empty or "dice" not in baseline:
            rows.append(
                {
                    "baseline": baseline_id,
                    "family": str(method.get("family", baseline_id)),
                    "status": "missing",
                    "superior": False,
                }
            )
            continue
        paired = _paired(proposed, baseline, "dice")
        paired["difference"] = paired.dice_proposed - paired.dice_baseline
        effect, low, high = _stratified_bootstrap(paired, draws_bootstrap, seed + index)
        complete = bool(
            proposed_complete
            and (not expected_runs or run_counts[baseline_id] == expected_runs)
            and (not expected_subjects or baseline.subject_id.nunique() == expected_subjects)
            and (not expected_subjects or len(paired) == expected_subjects)
        )
        rows.append(
            {
                "baseline": baseline_id,
                "family": str(method.get("family", baseline_id)),
                "status": "analyzed" if complete else "incomplete",
                "subjects": len(paired),
                "runs_observed": run_counts[baseline_id],
                "runs_expected": expected_runs,
                "baseline_mean_dice": float(baseline.dice.mean()),
                "dice_difference": effect,
                "dice_ci95_low": low,
                "dice_ci95_high": high,
                "p_raw": _permutation(paired.difference.to_numpy(), draws_permutation, seed + 100 + index),
                "is_strong_semisupervised": bool(method.get("strong_semisupervised", False)),
                "is_nnunet_official_3d": bool(method.get("nnunet_official_3d", False)),
                "cohort_differences": {
                    str(cohort): float(group.difference.mean()) for cohort, group in paired.groupby("cohort")
                },
            }
        )
    analyzed = [row for row in rows if row["status"] == "analyzed"]
    adjusted = holm_adjust([float(row["p_raw"]) for row in analyzed])
    for row, value in zip(analyzed, adjusted):
        row["p_holm"] = value
        row["superior"] = bool(row["dice_ci95_low"] > 0 and value < 0.05)
    return rows, tables


def _repeat_wcv(root: Path, method: str) -> float | None:
    values = []
    for path in sorted((root / method).glob("fold*_seed*/evaluation_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        value = summary.get("repeatability", {}).get("within_subject_cv_percent_mean")
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else None


def _external_result(
    benchmark_root: Path,
    dataset: str,
    proposed_id: str,
    method_ids: list[str],
    margin: float,
    bootstrap_draws: int,
    permutation_draws: int,
    seed: int,
) -> dict[str, Any]:
    root = benchmark_root / "external" / dataset
    proposed_path = root / proposed_id / "metrics_subject.csv"
    if not proposed_path.exists():
        return {"dataset": dataset, "status": "missing", "superior_any": False, "noninferior": False}
    proposed = pd.read_csv(proposed_path)
    comparisons = []
    for index, method_id in enumerate(method_ids):
        path = root / method_id / "metrics_subject.csv"
        if not path.exists():
            continue
        baseline = pd.read_csv(path)
        paired = _paired(proposed, baseline, "dice")
        paired["difference"] = paired.dice_proposed - paired.dice_baseline
        effect, low, high = _stratified_bootstrap(paired, bootstrap_draws, seed + index)
        comparisons.append(
            {
                "baseline": method_id,
                "difference": effect,
                "ci95_low": low,
                "ci95_high": high,
                "p_raw": _permutation(paired.difference.to_numpy(), permutation_draws, seed + 100 + index),
            }
        )
    adjusted = holm_adjust([row["p_raw"] for row in comparisons])
    for row, p_value in zip(comparisons, adjusted):
        row["p_holm"] = p_value
        row["superior"] = bool(row["ci95_low"] > 0 and p_value < 0.05)
        row["noninferior"] = bool(row["ci95_low"] > margin)
    return {
        "dataset": dataset,
        "status": "analyzed" if comparisons else "missing_baselines",
        "proposed_mean_dice": float(proposed.dice.mean()),
        "superior_any": any(row["superior"] for row in comparisons),
        "noninferior": bool(comparisons) and all(row["noninferior"] for row in comparisons),
        "comparisons": comparisons,
    }


def compare_q2_gate(
    benchmark_dir: str | Path,
    suite_path: str | Path | None = None,
    seed: int = 20260803,
) -> Path:
    benchmark_dir = Path(benchmark_dir).resolve()
    suite_path = Path(suite_path).resolve() if suite_path else benchmark_dir / "suite.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    gate = suite["q2_gate"]
    proposed_id = str(suite["proposed_method"])
    internal_root = benchmark_dir / "internal"
    bootstrap_draws = int(gate.get("bootstrap_draws", 10_000))
    permutation_draws = int(gate.get("permutation_draws", 100_000))
    rows, tables = _comparison_rows(
        internal_root,
        suite,
        proposed_id,
        bootstrap_draws,
        permutation_draws,
        seed,
    )
    proposed = tables[proposed_id]
    analyzed = [row for row in rows if row["status"] == "analyzed"]
    strongest = max(analyzed, key=lambda row: row["baseline_mean_dice"], default=None)
    nnunet = [row for row in analyzed if row["is_nnunet_official_3d"]]
    strong_semi = [row for row in analyzed if row["is_strong_semisupervised"]]
    repeat_wcv = _repeat_wcv(internal_root, proposed_id)
    proposed_dice = float(proposed.dice.mean()) if not proposed.empty else None
    tkv_error = float(proposed.tkv_relative_error.mean()) if not proposed.empty else None
    matched_baseline = str(suite.get("matched_baseline", "V2_B0plus"))
    b0 = tables.get(matched_baseline, pd.DataFrame())
    dice_tradeoff = (
        proposed_dice - float(b0.dice.mean())
        if proposed_dice is not None and not b0.empty
        else None
    )
    external_ids = [
        str(method["id"])
        for method in suite["methods"]
        if bool(method.get("external_reference", False))
    ]
    external = [
        _external_result(
            benchmark_dir,
            dataset,
            proposed_id,
            external_ids,
            float(gate["external_noninferiority_margin"]),
            bootstrap_draws,
            permutation_draws,
            seed + 1000 * (index + 1),
        )
        for index, dataset in enumerate(suite["phases"]["external"]["datasets"])
    ]
    external_runs_path = benchmark_dir / "external" / "external_runs.csv"
    if external_runs_path.exists():
        external_runs = pd.read_csv(external_runs_path)
        external_complete = bool(len(external_runs)) and set(external_runs.status).issubset(
            {"completed", "skipped_completed"}
        )
    else:
        external_complete = False
    full_label_path = benchmark_dir / "label_efficiency" / "100pct_summary.json"
    full_label_dice = (
        float(json.loads(full_label_path.read_text(encoding="utf-8"))["proposed_mean_dice"])
        if full_label_path.exists()
        else None
    )
    checks = {
        "all_internal_methods_present": bool(rows) and all(row["status"] == "analyzed" for row in rows),
        "all_external_runs_present": external_complete,
        "internal_dice_at_least_target": proposed_dice is not None
        and proposed_dice >= float(gate["internal_min_dice"]),
        "nnunet_official_3d_superiority": bool(nnunet) and all(row["superior"] for row in nnunet),
        "two_strong_semisupervised_superiorities": sum(row["superior"] for row in strong_semi) >= 2,
        "effect_over_strongest_at_least_target": strongest is not None
        and strongest["dice_difference"] >= float(gate["internal_min_effect"]),
        "hc_ckd_direction_consistent": strongest is not None
        and all(strongest["cohort_differences"].get(cohort, -np.inf) >= 0 for cohort in ("HC", "CKD")),
        "tkv_error_at_most_target": tkv_error is not None
        and tkv_error <= float(gate["tkv_max_relative_error"]),
        "repeat_wcv_at_most_target": repeat_wcv is not None
        and repeat_wcv <= float(gate["repeat_max_wcv_percent"]),
        "dice_repeatability_tradeoff_acceptable": dice_tradeoff is not None
        and dice_tradeoff >= -float(gate["repeat_max_dice_tradeoff"]),
        "one_external_superiority": any(result["superior_any"] for result in external),
        "other_external_noninferiority": len(external) >= 2
        and sum(result["noninferior"] for result in external) >= 2,
        "full_label_sanity": full_label_dice is not None
        and full_label_dice >= float(gate.get("full_label_min_dice", 0.90)),
    }
    passed = all(checks.values())
    result = {
        "schema_version": 2,
        "status": "Q2_SUBMISSION_CONDITIONS_MET" if passed else "Q2_GATE_NOT_MET",
        "q2_gate_passed": passed,
        "independent_unit": "subject_after_repeat_acquisition_and_seed_averaging",
        "proposed_method": proposed_id,
        "proposed_internal_mean_dice": proposed_dice,
        "proposed_tkv_relative_error": tkv_error,
        "proposed_repeat_wcv_percent": repeat_wcv,
        "full_label_mean_dice": full_label_dice,
        "checks": checks,
        "internal_comparisons": rows,
        "external_evaluations": external,
        "bootstrap_draws": bootstrap_draws,
        "permutation_draws": permutation_draws,
    }
    atomic_json(result, benchmark_dir / "q2_gate_summary.json")
    _write_report(benchmark_dir / "Q2_投稿门槛评估.md", result)
    return benchmark_dir / "q2_gate_summary.json"


def _write_report(path: Path, result: dict[str, Any]) -> None:
    passed = bool(result["q2_gate_passed"])
    conclusion = (
        "全部预注册门槛通过，具备尝试 Q2 投稿的实验条件；这不代表保证录用或固定分区。"
        if passed
        else "预注册门槛尚未全部通过，报告不得宣称具备 Q2 投稿条件。"
    )
    lines = [
        "# KidneyBand-ReliableSSL v2：Q2 投稿门槛评估",
        "",
        f"> {conclusion}",
        "",
        f"- 状态：`{result['status']}`",
        f"- 内部平均 Dice：{result['proposed_internal_mean_dice']}",
        f"- TKV 相对误差：{result['proposed_tkv_relative_error']}",
        f"- scan–rescan wCV：{result['proposed_repeat_wcv_percent']}",
        "",
        "## 预注册检查",
        "",
    ]
    lines.extend(
        f"- [{'x' if value else ' '}] `{name}`" for name, value in result["checks"].items()
    )
    lines.extend(
        [
            "",
            "## 统计边界",
            "",
            "所有推断以受试者为独立单位；重复采集先在受试者内合并，再平均随机种子。"
            "公开论文分数不参与本地胜负判定，缺失或失败运行保留并使完整性门槛失败。",
            "",
            "## Material Passport",
            "",
            "- Origin Skill: academic-research-suite / experiment-agent",
            "- Origin Mode: implementation and validation",
            "- Verification Status: " + ("VERIFIED-GATE-PASSED" if passed else "UNVERIFIED-GATE-NOT-MET"),
            "- Version Label: q2_experiment_plan_v2",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

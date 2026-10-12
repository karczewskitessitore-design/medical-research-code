from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from kidney_ssl.utils import atomic_json


METRICS = ("dice", "hd95_mm", "asd_mm", "tkv_relative_error")


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def _stratified_bootstrap_difference(
    table: pd.DataFrame,
    proposed: str,
    baseline: str,
    metric: str,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    differences = table[f"{metric}_{proposed}"] - table[f"{metric}_{baseline}"]
    work = table.assign(_difference=differences)
    samples = np.zeros(draws, dtype=float)
    for _, group in work.groupby(["cohort", "is_repeat"], dropna=False):
        values = group._difference.to_numpy(float)
        indices = rng.integers(0, len(values), size=(draws, len(values)))
        samples += values[indices].sum(axis=1) / len(work)
    return float(differences.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _paired_permutation_p(difference: np.ndarray, draws: int, seed: int) -> float:
    values = np.asarray(difference, dtype=float)
    observed = abs(float(values.mean()))
    if not np.any(values):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(draws, len(values)))
    permuted = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(permuted >= observed)) / (draws + 1))


def _load_suite(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        suite = yaml.safe_load(handle) or {}
    required = {"name", "proposed_method", "methods"}
    missing = required - set(suite)
    if missing:
        raise ValueError(f"Benchmark suite missing keys: {sorted(missing)}")
    return suite


def _collect_subject_metrics(root: Path, methods: list[dict[str, Any]]) -> pd.DataFrame:
    tables = []
    for method in methods:
        method_id = str(method["id"])
        for path in sorted((root / method_id).glob("fold*_seed*/metrics_subject.csv")):
            table = pd.read_csv(path)
            run = path.parent.name
            fold_text, seed_text = run.split("_")
            table["method"] = method_id
            table["fold"] = int(fold_text.removeprefix("fold"))
            table["seed"] = int(seed_text.removeprefix("seed"))
            tables.append(table)
    if not tables:
        raise FileNotFoundError(f"No metrics_subject.csv files found under {root}")
    acquisition_averaged = pd.concat(tables, ignore_index=True)
    required = {"subject_id", "cohort", "is_repeat", "method", "seed", *METRICS}
    missing = required - set(acquisition_averaged.columns)
    if missing:
        raise ValueError(f"Subject metric files missing columns: {sorted(missing)}")
    return acquisition_averaged


def _collect_resource_metrics(root: Path, methods: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for method in methods:
        method_id = str(method["id"])
        for result_path in sorted((root / method_id).glob("fold*_seed*/training_result.json")):
            training = json.loads(result_path.read_text(encoding="utf-8"))
            evaluation_path = result_path.parent / "evaluation_summary.json"
            evaluation = (
                json.loads(evaluation_path.read_text(encoding="utf-8"))
                if evaluation_path.exists()
                else {}
            )
            rows.append(
                {
                    "method": method_id,
                    "run": result_path.parent.name,
                    "trainable_parameters": training.get("trainable_parameters"),
                    "peak_gpu_memory_mb": training.get("peak_gpu_memory_mb"),
                    "training_seconds": training.get("duration_seconds"),
                    "inference_seconds": evaluation.get("inference_seconds"),
                }
            )
    return pd.DataFrame(rows)


def compare_benchmark(
    benchmark_dir: str | Path,
    suite_path: str | Path | None = None,
    draws: int = 10000,
    seed: int = 20260803,
) -> Path:
    benchmark_dir = Path(benchmark_dir).resolve()
    suite_path = Path(suite_path).resolve() if suite_path else benchmark_dir / "suite.yaml"
    suite = _load_suite(suite_path)
    methods = suite["methods"]
    proposed = str(suite["proposed_method"])
    raw = _collect_subject_metrics(benchmark_dir, methods)
    resources = _collect_resource_metrics(benchmark_dir, methods)
    if not resources.empty:
        resources.to_csv(benchmark_dir / "resource_efficiency_runs.csv", index=False)
        resource_summary = (
            resources.groupby("method", as_index=False)
            .agg(
                trainable_parameters=("trainable_parameters", "max"),
                peak_gpu_memory_mb=("peak_gpu_memory_mb", "max"),
                training_seconds_mean=("training_seconds", "mean"),
                inference_seconds_mean=("inference_seconds", "mean"),
            )
        )
        resource_summary.to_csv(benchmark_dir / "resource_efficiency_summary.csv", index=False)
    else:
        resource_summary = pd.DataFrame()
    phases = suite.get("phases", {})
    confirmatory = phases.get("confirmatory") or phases.get("internal", {})
    expected_runs = None
    if confirmatory.get("folds") and confirmatory.get("seeds"):
        expected_runs = len(confirmatory["folds"]) * len(confirmatory["seeds"])
    expected_subjects = suite.get("expected_subjects")
    run_counts = raw.groupby("method")[["fold", "seed"]].apply(
        lambda frame: len(frame.drop_duplicates())
    )

    # Repeated acquisitions are already averaged by evaluate(); now average stochastic seeds.
    subject = raw.groupby(
        ["subject_id", "cohort", "is_repeat", "method"], as_index=False
    )[list(METRICS)].mean()
    subject.to_csv(benchmark_dir / "all_methods_subject_metrics.csv", index=False)

    wide = subject.pivot(index=["subject_id", "cohort", "is_repeat"], columns="method", values=list(METRICS))
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    if f"dice_{proposed}" not in wide:
        raise ValueError(f"Proposed method {proposed!r} has no complete metrics")
    proposed_subjects = int(subject[subject.method == proposed].subject_id.nunique())
    proposed_complete = bool(
        (expected_runs is None or int(run_counts.get(proposed, 0)) == expected_runs)
        and (expected_subjects is None or proposed_subjects == int(expected_subjects))
    )

    baselines = [
        method
        for method in methods
        if bool(method.get("comparison_baseline", method.get("sci_baseline", False)))
        and bool(method.get("primary_for_family", True))
    ]
    rows: list[dict[str, Any]] = []
    for index, method in enumerate(baselines):
        baseline = str(method["id"])
        needed = [f"dice_{baseline}", f"tkv_relative_error_{baseline}"]
        if any(column not in wide for column in needed):
            rows.append(
                {
                    "baseline": baseline,
                    "family": str(method.get("family", baseline)),
                    "is_sci_journal": bool(method.get("sci_baseline", False)),
                    "is_strong_semisupervised": bool(
                        method.get("strong_semisupervised", False)
                    ),
                    "is_nnunet_official_3d": bool(
                        method.get("nnunet_official_3d", False)
                    ),
                    "status": "missing",
                    "outperformed": False,
                }
            )
            continue
        paired = wide.dropna(
            subset=[f"dice_{proposed}", f"dice_{baseline}", f"tkv_relative_error_{proposed}", f"tkv_relative_error_{baseline}"]
        )
        baseline_subjects = int(subject[subject.method == baseline].subject_id.nunique())
        complete = bool(
            proposed_complete
            and (expected_runs is None or int(run_counts.get(baseline, 0)) == expected_runs)
            and (expected_subjects is None or baseline_subjects == int(expected_subjects))
            and (expected_subjects is None or len(paired) == int(expected_subjects))
        )
        dice_mean, dice_low, dice_high = _stratified_bootstrap_difference(
            paired, proposed, baseline, "dice", draws, seed + index
        )
        tkv_mean, tkv_low, tkv_high = _stratified_bootstrap_difference(
            paired, proposed, baseline, "tkv_relative_error", draws, seed + 100 + index
        )
        dice_difference = paired[f"dice_{proposed}"] - paired[f"dice_{baseline}"]
        dice_sd = float(dice_difference.std(ddof=1))
        rows.append(
            {
                "baseline": baseline,
                "family": str(method.get("family", baseline)),
                "journal": str(method.get("journal", "")),
                "is_sci_journal": bool(method.get("sci_baseline", False)),
                "is_strong_semisupervised": bool(
                    method.get("strong_semisupervised", False)
                ),
                "is_nnunet_official_3d": bool(method.get("nnunet_official_3d", False)),
                "subjects": int(len(paired)),
                "runs_observed": int(run_counts.get(baseline, 0)),
                "runs_expected": expected_runs,
                "status": "analyzed" if complete else "incomplete",
                "dice_difference": dice_mean,
                "dice_ci95_low": dice_low,
                "dice_ci95_high": dice_high,
                "dice_p_permutation": _paired_permutation_p(
                    dice_difference.to_numpy(float), draws, seed + 200 + index
                ),
                "dice_effect_dz": (
                    float(dice_difference.mean() / dice_sd) if dice_sd > 0 else None
                ),
                "tkv_error_difference": tkv_mean,
                "tkv_error_ci95_low": tkv_low,
                "tkv_error_ci95_high": tkv_high,
            }
        )

    analyzed = [row for row in rows if row["status"] == "analyzed"]
    adjusted = holm_adjust([float(row["dice_p_permutation"]) for row in analyzed])
    for row, p_adjusted in zip(analyzed, adjusted):
        row["dice_p_holm"] = p_adjusted
        row["dice_superiority"] = bool(row["dice_ci95_low"] > 0 and p_adjusted < 0.05)
        row["tkv_noninferior"] = bool(row["tkv_error_ci95_high"] <= 0.02)
        row["outperformed"] = bool(row["dice_superiority"] and row["tkv_noninferior"])

    comparison = pd.DataFrame(rows)
    comparison.to_csv(benchmark_dir / "paired_method_comparisons.csv", index=False)
    comparison[comparison.get("is_sci_journal", False) == True].to_csv(  # noqa: E712
        benchmark_dir / "paired_sci_comparisons.csv", index=False
    )
    outperformed = [row for row in rows if row.get("outperformed")]
    sci_rows = [row for row in rows if row.get("is_sci_journal", False)]
    sci_analyzed = [row for row in analyzed if row.get("is_sci_journal", False)]
    sci_outperformed = [row for row in outperformed if row.get("is_sci_journal", False)]
    proposed_dice = float(subject[subject.method == proposed].dice.mean())
    strong_semi = any(
        row.get("is_strong_semisupervised", False) for row in sci_outperformed
    )
    nnunet_outperformed = any(
        row.get("is_nnunet_official_3d", False) for row in sci_outperformed
    )
    strong_semi_count = sum(
        bool(row.get("is_strong_semisupervised", False)) for row in sci_outperformed
    )
    if nnunet_outperformed and strong_semi_count >= 2 and proposed_dice >= 0.90:
        tier = "JCR Q2较有说服力；中科院2–3区可尝试"
        candidates = [
            "Magnetic Resonance in Medicine",
            "Medical Physics",
            "Computerized Medical Imaging and Graphics",
        ]
    elif len(sci_outperformed) >= 3 and strong_semi:
        tier = "可评估JCR Q3并尝试Q2；中科院约2–3区"
        candidates = [
            "Computerized Medical Imaging and Graphics",
            "BMC Medical Imaging",
            "Magnetic Resonance Materials in Physics, Biology and Medicine",
        ]
    else:
        tier = "尚无强算法优势；若转应用论文，现实范围约JCR Q3–Q4/中科院3–4区"
        candidates = [
            "BMC Medical Imaging",
            "Journal of Imaging Informatics in Medicine",
            "Magnetic Resonance Materials in Physics, Biology and Medicine",
        ]
    summary = {
        "proposed_method": proposed,
        "evaluation_unit": "subject_after_acquisition_and_seed_averaging",
        "bootstrap_draws": draws,
        "permutation_draws": draws,
        "holm_family_size": len(analyzed),
        "method_families_planned": len(baselines),
        "method_families_analyzed": len(analyzed),
        "sci_families_planned": len(sci_rows),
        "sci_families_analyzed": len(sci_analyzed),
        "confirmatory_complete": bool(proposed_complete and len(analyzed) == len(baselines)),
        "method_outperformed_count": len(outperformed),
        "method_outperformed_families": [row["family"] for row in outperformed],
        "outperformed_count": len(sci_outperformed),
        "outperformed_families": [row["family"] for row in sci_outperformed],
        "proposed_mean_dice": proposed_dice,
        "claim": (
            f"在预注册标准下显著优于{len(sci_outperformed)}个SCI期刊算法家族。"
            if sci_outperformed
            else "未证明优于预注册的SCI期刊算法家族。"
        ),
        "comparisons": rows,
        "submission_assessment": {
            "conditional_tier": tier,
            "candidate_journals": candidates,
            "jcr_reference": "2026 JCR release (2025 citation data); verify category quartile in institutional JCR on submission date",
            "cas_reference": "2025 CAS journal partition table; verify medicine/engineering category in the official platform on submission date",
            "q1_recommendation": "not recommended without independent multicenter MRI validation and clinical assessment",
        },
        "resources": (
            json.loads(resource_summary.to_json(orient="records"))
            if not resource_summary.empty
            else []
        ),
    }
    atomic_json(summary, benchmark_dir / "comparison_summary.json")
    _write_report(benchmark_dir, summary)
    return benchmark_dir / "comparison_summary.json"


def _write_report(root: Path, summary: dict[str, Any]) -> None:
    count = int(summary["outperformed_count"])
    lines = [
        "# SCI算法公平比较结果",
        "",
        "> 本报告只统计相同受试者划分、25%标注和统一评价流程下的本地重跑结果。论文公开分数不参与胜负判定。",
        "",
        f"- 主方法：`{summary['proposed_method']}`",
        f"- 完成方法比较：{summary['method_families_analyzed']}/{summary['method_families_planned']}个算法家族",
        f"- 其中SCI期刊算法：{summary['sci_families_analyzed']}/{summary['sci_families_planned']}个",
        f"- 满足Dice显著优越且TKV非劣的SCI期刊算法：{count}个",
        f"- 结论：{summary['claim']}",
        "",
        "| 对比算法家族 | 类型 | Dice差值 [95% CI] | 配对dz | Holm p | TKV误差差值 [95% CI] | 判定 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["comparisons"]:
        if row["status"] != "analyzed":
            kind = "SCI期刊" if row.get("is_sci_journal", False) else "会议/方法学"
            lines.append(f"| {row['family']} | {kind} | NA | NA | NA | NA | 缺失/不完整，不能宣称超过 |")
            continue
        verdict = "超过" if row["outperformed"] else "未证明超过"
        kind = "SCI期刊" if row.get("is_sci_journal", False) else "会议/方法学"
        effect = "NA" if row["dice_effect_dz"] is None else f"{row['dice_effect_dz']:.3f}"
        lines.append(
            f"| {row['family']} | {kind} | {row['dice_difference']:.4f} "
            f"[{row['dice_ci95_low']:.4f}, {row['dice_ci95_high']:.4f}] | {effect} | "
            f"{row['dice_p_holm']:.4g} | {row['tkv_error_difference']:.4f} "
            f"[{row['tkv_error_ci95_low']:.4f}, {row['tkv_error_ci95_high']:.4f}] | {verdict} |"
        )
    if summary.get("resources"):
        lines.extend(
            [
                "",
                "## 资源效率",
                "",
                "| 方法 | 参数量 | 峰值显存(MB) | 平均训练(s) | 平均推理(s) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in summary["resources"]:
            parameters = "NA" if row["trainable_parameters"] is None else f"{row['trainable_parameters']:.0f}"
            memory = "NA" if row["peak_gpu_memory_mb"] is None else f"{row['peak_gpu_memory_mb']:.1f}"
            training = "NA" if row["training_seconds_mean"] is None else f"{row['training_seconds_mean']:.1f}"
            inference = "NA" if row["inference_seconds_mean"] is None else f"{row['inference_seconds_mean']:.1f}"
            lines.append(
                f"| {row['method']} | {parameters} | {memory} | {training} | {inference} |"
            )
    lines.extend(
        [
            "",
            "## 条件式投稿评估",
            "",
            f"- B2受试者平均Dice：{summary['proposed_mean_dice']:.4f}",
            f"- 证据档位：{summary['submission_assessment']['conditional_tier']}",
            "- 梯度候选（投稿时必须在单位订阅中重新核验JCR类别与中科院分区）："
            + "、".join(summary["submission_assessment"]["candidate_journals"]),
            "- JCR口径：2026 JCR发布、使用2025引文数据；中科院口径：2025分区表。两套体系不混用。",
            "- JCR Q1/中科院1区：当前单中心单数据集不建议。",
        ]
    )
    lines.extend(
        [
            "",
            "## Material Passport",
            "",
            "- Origin Skill: experiment-agent",
            "- Origin Mode: validate",
            "- Verification Status: ANALYZED",
            "- Version Label: sci_comparison_v1",
        ]
    )
    (root / "SCI_算法公平比较.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

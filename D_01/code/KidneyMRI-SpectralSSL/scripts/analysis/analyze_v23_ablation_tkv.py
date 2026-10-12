from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from kidney_ssl.comparison import holm_adjust
from kidney_ssl.metrics import icc_2_1


PRIOR_METHODS = {
    "B0+": "V22_B0plus_5slice",
    "Full CSD-MT": "KidneyBand_ReliableSSL_v2_2_CSD",
    "nnU-Net v2 3D": "nnunet_3d_default",
}
NEW_METHODS = {
    "Binary-Sup": "V23_BinarySup_5slice",
    "Mean Teacher": "V23_MeanTeacher_5slice",
    "M1-only": "V23_M1_CSD_Global_5slice",
}
ABLATION_ORDER = ["Binary-Sup", "B0+", "Mean Teacher", "M1-only", "Full CSD-MT"]
DIRECT_CONTRASTS = [
    ("B0+", "Binary-Sup", "M3"),
    ("Mean Teacher", "B0+", "generic semi-supervision"),
    ("M1-only", "Mean Teacher", "M1"),
    ("Full CSD-MT", "M1-only", "M2"),
]


@dataclass(frozen=True)
class AnalysisSettings:
    bootstrap_draws: int
    permutation_draws: int
    seed: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_parts(run_name: str) -> tuple[int, int]:
    fold_part, seed_part = run_name.split("_seed", 1)
    return int(fold_part.removeprefix("fold")), int(seed_part)


def read_archive_method(archive: Path, method_id: str) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    suffix = "/metrics_acquisition.csv"
    prefix = f"internal/{method_id}/"
    with tarfile.open(archive, "r:gz") as bundle:
        members = sorted(
            (
                member for member in bundle.getmembers()
                if member.isfile() and member.name.startswith(prefix) and member.name.endswith(suffix)
            ),
            key=lambda member: member.name,
        )
        for member in members:
            run_name = Path(member.name).parent.name
            fold, seed = _run_parts(run_name)
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read {member.name} from {archive}")
            table = pd.read_csv(extracted)
            table["fold"] = fold
            table["seed"] = seed
            tables.append(table)
    if not tables:
        raise FileNotFoundError(f"No acquisition tables for {method_id} in {archive}")
    return pd.concat(tables, ignore_index=True)


def read_directory_method(root: Path, method_id: str) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for path in sorted((root / method_id).glob("fold*_seed*/metrics_acquisition.csv")):
        fold, seed = _run_parts(path.parent.name)
        table = pd.read_csv(path)
        table["fold"] = fold
        table["seed"] = seed
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def validate_method_table(table: pd.DataFrame, method: str, expected_runs: int = 15) -> None:
    required = {
        "subject_id", "cohort", "acquisition_id", "is_repeat", "dice", "hd95_mm",
        "pred_tkv_ml", "true_tkv_ml", "tkv_relative_error", "fold", "seed",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{method}: missing columns {missing}")
    runs = table[["fold", "seed"]].drop_duplicates()
    if len(runs) != expected_runs:
        raise ValueError(f"{method}: expected {expected_runs} runs, found {len(runs)}")
    if table[["dice", "hd95_mm", "pred_tkv_ml", "true_tkv_ml"]].isna().any().any():
        raise ValueError(f"{method}: NaN found in required metrics")
    subjects = table.subject_id.nunique()
    if subjects != 60:
        raise ValueError(f"{method}: expected 60 unique subjects, found {subjects}")
    per_seed = table.groupby("seed").subject_id.nunique()
    if not (per_seed == 60).all():
        raise ValueError(f"{method}: incomplete subject coverage by seed: {per_seed.to_dict()}")


def subject_seed_table(table: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column for column in (
            "dice", "hd95_mm", "pred_tkv_ml", "true_tkv_ml", "tkv_relative_error"
        ) if column in table.columns
    ]
    return table.groupby(
        ["subject_id", "cohort", "is_repeat", "seed"], as_index=False
    )[numeric].mean()


def subject_table(table: pd.DataFrame) -> pd.DataFrame:
    per_seed = subject_seed_table(table)
    numeric = [
        column for column in (
            "dice", "hd95_mm", "pred_tkv_ml", "true_tkv_ml", "tkv_relative_error"
        ) if column in per_seed.columns
    ]
    result = per_seed.groupby(
        ["subject_id", "cohort", "is_repeat"], as_index=False
    )[numeric].mean()
    result["tkv_error_ml"] = result.pred_tkv_ml - result.true_tkv_ml
    result["tkv_abs_error_ml"] = result.tkv_error_ml.abs()
    result["tkv_ape_percent"] = result.tkv_abs_error_ml / result.true_tkv_ml * 100.0
    return result


def lin_ccc(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    covariance = np.mean((reference - reference.mean()) * (estimate - estimate.mean()))
    denominator = reference.var() + estimate.var() + (reference.mean() - estimate.mean()) ** 2
    return float(2.0 * covariance / denominator) if denominator > 0 else float("nan")


def _stratified_indices(table: pd.DataFrame, draws: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    groups = [np.asarray(indices, dtype=int) for indices in table.groupby("cohort", sort=True).indices.values()]
    for _ in range(draws):
        yield np.concatenate([group[rng.integers(0, len(group), len(group))] for group in groups])


def bootstrap_stat(
    table: pd.DataFrame,
    statistic,
    settings: AnalysisSettings,
    offset: int,
    stratified: bool = True,
) -> tuple[float, float, float]:
    point = float(statistic(table))
    rng = np.random.default_rng(settings.seed + offset)
    values = np.empty(settings.bootstrap_draws, dtype=float)
    if stratified and "cohort" in table:
        iterator = _stratified_indices(table.reset_index(drop=True), settings.bootstrap_draws, rng)
    else:
        n = len(table)
        iterator = (rng.integers(0, n, n) for _ in range(settings.bootstrap_draws))
    reset = table.reset_index(drop=True)
    for index, selected in enumerate(iterator):
        values[index] = statistic(reset.iloc[selected])
    finite = values[np.isfinite(values)]
    if not len(finite):
        return point, float("nan"), float("nan")
    return point, float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def paired_permutation(values: np.ndarray, settings: AnalysisSettings, offset: int) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(settings.seed + offset)
    exceed = 0
    completed = 0
    while completed < settings.permutation_draws:
        count = min(10_000, settings.permutation_draws - completed)
        signs = rng.choice((-1.0, 1.0), size=(count, len(values)))
        exceed += int(np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed))
        completed += count
    return float((exceed + 1) / (settings.permutation_draws + 1))


def repeat_subject_table(table: pd.DataFrame) -> pd.DataFrame:
    repeated = table[table.is_repeat.astype(bool)].copy()
    rows: list[dict[str, float | int | str]] = []
    for (subject, cohort, seed), group in repeated.groupby(["subject_id", "cohort", "seed"]):
        predicted = group.sort_values("acquisition_id").pred_tkv_ml.to_numpy(float)
        reference = group.sort_values("acquisition_id").true_tkv_ml.to_numpy(float)
        if len(predicted) != 5:
            raise ValueError(f"{subject}, seed {seed}: expected five repeat acquisitions, found {len(predicted)}")
        rows.append({
            "subject_id": subject,
            "cohort": cohort,
            "seed": int(seed),
            "wcv_percent": float(predicted.std(ddof=1) / predicted.mean() * 100.0),
            "within_subject_sd_ml": float(predicted.std(ddof=1)),
            "manual_wcv_percent": float(reference.std(ddof=1) / reference.mean() * 100.0),
            "manual_within_subject_sd_ml": float(reference.std(ddof=1)),
        })
    per_seed = pd.DataFrame(rows)
    if per_seed.subject_id.nunique() != 10:
        raise ValueError(f"Expected 10 repeat subjects, found {per_seed.subject_id.nunique()}")
    return per_seed


def _repeat_stats_matrices(matrices: list[np.ndarray]) -> dict[str, float]:
    wcv_by_seed: list[float] = []
    icc_by_seed: list[float] = []
    sw_by_seed: list[float] = []
    for matrix in matrices:
        subject_sd = matrix.std(axis=1, ddof=1)
        wcv_by_seed.append(float(np.mean(subject_sd / matrix.mean(axis=1) * 100.0)))
        icc_by_seed.append(float(icc_2_1(matrix)))
        residuals = matrix - matrix.mean(axis=1, keepdims=True)
        sw_by_seed.append(float(np.sqrt(np.sum(residuals**2) / (matrix.shape[0] * (matrix.shape[1] - 1)))))
    sw = float(np.mean(sw_by_seed))
    return {
        "mean_wcv_percent": float(np.mean(wcv_by_seed)),
        "icc_2_1": float(np.mean(icc_by_seed)),
        "within_subject_sd_ml": sw,
        "repeatability_coefficient_ml": 2.77 * sw,
    }


def repeat_summary_with_ci(
    table: pd.DataFrame,
    value: str,
    settings: AnalysisSettings,
    offset: int,
) -> dict[str, tuple[float, float, float]]:
    repeated = table[table.is_repeat.astype(bool)].copy()
    subjects = repeated[["subject_id", "cohort"]].drop_duplicates().reset_index(drop=True)
    subject_order = subjects.subject_id.astype(str).tolist()
    matrices: list[np.ndarray] = []
    for _, seed_table in repeated.groupby("seed", sort=True):
        matrix = np.asarray([
            seed_table[seed_table.subject_id == subject].sort_values("acquisition_id")[value].to_numpy(float)
            for subject in subject_order
        ])
        if matrix.shape != (10, 5):
            raise ValueError(f"Repeatability matrix has unexpected shape {matrix.shape}")
        matrices.append(matrix)
    point = _repeat_stats_matrices(matrices)
    rng = np.random.default_rng(settings.seed + offset)
    draws = {key: np.empty(settings.bootstrap_draws, dtype=float) for key in point}
    for draw_index, indices in enumerate(_stratified_indices(subjects, settings.bootstrap_draws, rng)):
        result = _repeat_stats_matrices([matrix[indices] for matrix in matrices])
        for key, value_result in result.items():
            draws[key][draw_index] = value_result
    return {
        key: (
            point[key],
            float(np.percentile(values[np.isfinite(values)], 2.5)),
            float(np.percentile(values[np.isfinite(values)], 97.5)),
        )
        for key, values in draws.items()
    }


def tkv_accuracy_rows(
    method: str,
    table: pd.DataFrame,
    settings: AnalysisSettings,
    offset: int,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    subjects = subject_table(table)
    rows: list[dict[str, object]] = []
    for cohort_index, cohort in enumerate(("All", "HC", "CKD")):
        subset = subjects if cohort == "All" else subjects[subjects.cohort == cohort]
        bias = bootstrap_stat(subset, lambda x: x.tkv_error_ml.mean(), settings, offset + cohort_index * 20)
        mae = bootstrap_stat(subset, lambda x: x.tkv_abs_error_ml.mean(), settings, offset + cohort_index * 20 + 1)
        ccc = bootstrap_stat(
            subset,
            lambda x: lin_ccc(x.true_tkv_ml.to_numpy(), x.pred_tkv_ml.to_numpy()),
            settings,
            offset + cohort_index * 20 + 2,
            stratified=cohort == "All",
        )
        differences = subset.tkv_error_ml.to_numpy(float)
        rows.append({
            "method": method,
            "cohort": cohort,
            "subjects": len(subset),
            "bias_ml": bias[0], "bias_ci95_low": bias[1], "bias_ci95_high": bias[2],
            "mae_ml": mae[0], "mae_ci95_low": mae[1], "mae_ci95_high": mae[2],
            "ape_median_percent": float(np.median(subset.tkv_ape_percent)),
            "ape_q1_percent": float(np.percentile(subset.tkv_ape_percent, 25)),
            "ape_q3_percent": float(np.percentile(subset.tkv_ape_percent, 75)),
            "ape_below_10_percent": float(np.mean(subset.tkv_ape_percent < 10) * 100.0),
            "ape_below_20_percent": float(np.mean(subset.tkv_ape_percent < 20) * 100.0),
            "ape_below_30_percent": float(np.mean(subset.tkv_ape_percent < 30) * 100.0),
            "ccc": ccc[0], "ccc_ci95_low": ccc[1], "ccc_ci95_high": ccc[2],
            "ba_lower_loa_ml": float(differences.mean() - 1.96 * differences.std(ddof=1)),
            "ba_upper_loa_ml": float(differences.mean() + 1.96 * differences.std(ddof=1)),
        })
    subjects.insert(0, "method", method)
    return rows, subjects


def tkv_analysis(
    tables: dict[str, pd.DataFrame],
    out_dir: Path,
    settings: AnalysisSettings,
) -> dict[str, object]:
    accuracy_rows: list[dict[str, object]] = []
    subject_rows: list[pd.DataFrame] = []
    repeat_rows: list[dict[str, object]] = []
    repeat_subject_rows: list[pd.DataFrame] = []
    for method_index, (method, table) in enumerate(tables.items()):
        rows, subjects = tkv_accuracy_rows(method, table, settings, 1000 + method_index * 100)
        accuracy_rows.extend(rows)
        subject_rows.append(subjects)
        repeated = repeat_subject_table(table)
        repeated.insert(0, "method", method)
        repeat_subject_rows.append(repeated)
        summary = repeat_summary_with_ci(table, "pred_tkv_ml", settings, 2000 + method_index * 100)
        repeat_rows.append({
            "method": method,
            "cohort": "All",
            "repeat_subjects": 10,
            **{
                key: values[0] for key, values in summary.items()
            },
            **{
                f"{key}_ci95_low": values[1] for key, values in summary.items()
            },
            **{
                f"{key}_ci95_high": values[2] for key, values in summary.items()
            },
        })
    reference_source = next(iter(tables.values()))
    reference_repeat = repeat_summary_with_ci(reference_source, "true_tkv_ml", settings, 2900)
    repeat_rows.append({
        "method": "Manual reference",
        "cohort": "All",
        "repeat_subjects": 10,
        **{key: values[0] for key, values in reference_repeat.items()},
        **{f"{key}_ci95_low": values[1] for key, values in reference_repeat.items()},
        **{f"{key}_ci95_high": values[2] for key, values in reference_repeat.items()},
    })
    accuracy = pd.DataFrame(accuracy_rows)
    subjects = pd.concat(subject_rows, ignore_index=True)
    repeat = pd.DataFrame(repeat_rows)
    repeat_subjects = pd.concat(repeat_subject_rows, ignore_index=True)
    accuracy.to_csv(out_dir / "tkv_accuracy_summary.csv", index=False)
    subjects.to_csv(out_dir / "tkv_subject_agreement.csv", index=False)
    repeat.to_csv(out_dir / "tkv_repeatability_summary.csv", index=False)
    repeat_subjects.to_csv(out_dir / "tkv_repeatability_subject.csv", index=False)
    write_table7(out_dir / "Table7_TKVMeasurement.md", accuracy, repeat)
    return {
        "status": "complete",
        "methods": list(tables),
        "subjects_per_method": {method: int(subject_table(table).subject_id.nunique()) for method, table in tables.items()},
        "repeat_subjects_per_method": {method: int(table[table.is_repeat.astype(bool)].subject_id.nunique()) for method, table in tables.items()},
    }


def write_table7(path: Path, accuracy: pd.DataFrame, repeat: pd.DataFrame) -> None:
    overall = accuracy[accuracy.cohort == "All"].set_index("method")
    lines = [
        "**Table 7. Subject-level total kidney volume agreement and scan–rescan repeatability.**",
        "",
        "| Method | Bias, mL (95% CI) | MAE, mL (95% CI) | Absolute percentage error, median [IQR] | <10% / <20% / <30% | Lin CCC (95% CI) | wCV, % (95% CI) | ICC(2,1) (95% CI) | RC, mL (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in repeat.iterrows():
        method = str(row.method)
        if method in overall.index:
            acc = overall.loc[method]
            lines.append(
                f"| {method} | {acc.bias_ml:.1f} ({acc.bias_ci95_low:.1f} to {acc.bias_ci95_high:.1f}) "
                f"| {acc.mae_ml:.1f} ({acc.mae_ci95_low:.1f} to {acc.mae_ci95_high:.1f}) "
                f"| {acc.ape_median_percent:.1f} [{acc.ape_q1_percent:.1f}–{acc.ape_q3_percent:.1f}] "
                f"| {acc.ape_below_10_percent:.1f} / {acc.ape_below_20_percent:.1f} / {acc.ape_below_30_percent:.1f} "
                f"| {acc.ccc:.3f} ({acc.ccc_ci95_low:.3f} to {acc.ccc_ci95_high:.3f}) "
                f"| {row.mean_wcv_percent:.2f} ({row.mean_wcv_percent_ci95_low:.2f} to {row.mean_wcv_percent_ci95_high:.2f}) "
                f"| {row.icc_2_1:.3f} ({row.icc_2_1_ci95_low:.3f} to {row.icc_2_1_ci95_high:.3f}) "
                f"| {row.repeatability_coefficient_ml:.1f} ({row.repeatability_coefficient_ml_ci95_low:.1f} to {row.repeatability_coefficient_ml_ci95_high:.1f}) |"
            )
        else:
            lines.append(
                f"| {method} | — | — | — | — | — "
                f"| {row.mean_wcv_percent:.2f} ({row.mean_wcv_percent_ci95_low:.2f} to {row.mean_wcv_percent_ci95_high:.2f}) "
                f"| {row.icc_2_1:.3f} ({row.icc_2_1_ci95_low:.3f} to {row.icc_2_1_ci95_high:.3f}) "
                f"| {row.repeatability_coefficient_ml:.1f} ({row.repeatability_coefficient_ml_ci95_low:.1f} to {row.repeatability_coefficient_ml_ci95_high:.1f}) |"
            )
    lines.extend([
        "",
        "Agreement statistics use 60 independent subjects after averaging acquisitions within subject and then seeds. Repeatability statistics use the 10 subjects with five acquisitions (five healthy controls and five participants with CKD). CCC, concordance correlation coefficient; ICC, intraclass correlation coefficient; MAE, mean absolute error; RC, repeatability coefficient; wCV, within-subject coefficient of variation.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _paired_subjects(tables: dict[str, pd.DataFrame], treatment: str, control: str) -> pd.DataFrame:
    first = subject_table(tables[treatment])
    second = subject_table(tables[control])
    columns = ["subject_id", "cohort", "dice", "hd95_mm", "tkv_ape_percent"]
    return first[columns].merge(
        second[columns], on=["subject_id", "cohort"], suffixes=("_treatment", "_control"), validate="one_to_one"
    )


def ablation_analysis(
    tables: dict[str, pd.DataFrame],
    out_dir: Path,
    settings: AnalysisSettings,
) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    subject_frames: list[pd.DataFrame] = []
    for method in ABLATION_ORDER:
        subjects = subject_table(tables[method])
        subjects.insert(0, "method", method)
        subject_frames.append(subjects)
        repeat = repeat_subject_table(tables[method]).groupby(["subject_id", "cohort"], as_index=False).wcv_percent.mean()
        for cohort in ("All", "HC", "CKD"):
            subset = subjects if cohort == "All" else subjects[subjects.cohort == cohort]
            summaries.append({
                "method": method,
                "cohort": cohort,
                "subjects": len(subset),
                "dice_mean": float(subset.dice.mean()),
                "dice_sd": float(subset.dice.std(ddof=1)),
                "hd95_mean_mm": float(subset.hd95_mm.mean()),
                "tkv_ape_mean_percent": float(subset.tkv_ape_percent.mean()),
                "repeat_wcv_mean_percent": float(repeat.wcv_percent.mean()) if cohort == "All" else float(repeat[repeat.cohort == cohort].wcv_percent.mean()),
            })
    pairs: list[dict[str, object]] = []
    p_values: list[float] = []
    for index, (treatment, control, component) in enumerate(DIRECT_CONTRASTS):
        paired = _paired_subjects(tables, treatment, control)
        paired["dice_difference"] = paired.dice_treatment - paired.dice_control
        dice_ci = bootstrap_stat(paired, lambda x: x.dice_difference.mean(), settings, 4000 + index)
        hd_difference = paired.hd95_mm_treatment - paired.hd95_mm_control
        tkv_difference = paired.tkv_ape_percent_treatment - paired.tkv_ape_percent_control
        treatment_repeat = repeat_subject_table(tables[treatment]).groupby(["subject_id", "cohort"], as_index=False).wcv_percent.mean()
        control_repeat = repeat_subject_table(tables[control]).groupby(["subject_id", "cohort"], as_index=False).wcv_percent.mean()
        repeat_pair = treatment_repeat.merge(control_repeat, on=["subject_id", "cohort"], suffixes=("_treatment", "_control"))
        repeat_pair["difference"] = repeat_pair.wcv_percent_treatment - repeat_pair.wcv_percent_control
        repeat_ci = bootstrap_stat(repeat_pair, lambda x: x.difference.mean(), settings, 4200 + index)
        p_value = paired_permutation(paired.dice_difference.to_numpy(), settings, 4400 + index)
        p_values.append(p_value)
        pairs.append({
            "contrast": f"{treatment} vs {control}",
            "component": component,
            "treatment": treatment,
            "control": control,
            "subjects": len(paired),
            "dice_difference": dice_ci[0],
            "dice_ci95_low": dice_ci[1],
            "dice_ci95_high": dice_ci[2],
            "p_raw": p_value,
            "hd95_difference_mm": float(hd_difference.mean()),
            "tkv_ape_difference_percent": float(tkv_difference.mean()),
            "wcv_difference_percent": repeat_ci[0],
            "wcv_ci95_low": repeat_ci[1],
            "wcv_ci95_high": repeat_ci[2],
            "hc_dice_difference": float(paired.loc[paired.cohort == "HC", "dice_difference"].mean()),
            "ckd_dice_difference": float(paired.loc[paired.cohort == "CKD", "dice_difference"].mean()),
        })
    for row, adjusted in zip(pairs, holm_adjust(p_values)):
        row["p_holm"] = float(adjusted)
        row["significant"] = bool(adjusted < 0.05)
    method_summary = pd.DataFrame(summaries)
    pairwise = pd.DataFrame(pairs)
    subjects = pd.concat(subject_frames, ignore_index=True)
    method_summary.to_csv(out_dir / "ablation_method_summary.csv", index=False)
    pairwise.to_csv(out_dir / "ablation_pairwise.csv", index=False)
    subjects.to_csv(out_dir / "ablation_subject_metrics.csv", index=False)
    write_table6(out_dir / "Table6_FormalAblation.md", method_summary, pairwise)
    return {"status": "complete", "methods": ABLATION_ORDER, "contrasts": len(pairs)}


def write_table6(path: Path, summary: pd.DataFrame, pairs: pd.DataFrame) -> None:
    overall = summary[summary.cohort == "All"].set_index("method")
    lines = [
        "**Table 6. Prespecified component ablation at 25% labeled training subjects.**",
        "",
        "| Method | Dice, mean (SD) | HD95, mm | TKV absolute percentage error, % | scan–rescan wCV, % |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ABLATION_ORDER:
        row = overall.loc[method]
        lines.append(
            f"| {method} | {row.dice_mean:.4f} ({row.dice_sd:.4f}) | {row.hd95_mean_mm:.2f} "
            f"| {row.tkv_ape_mean_percent:.2f} | {row.repeat_wcv_mean_percent:.2f} |"
        )
    lines.extend([
        "",
        "| Prespecified contrast | Component interpretation | ΔDice (95% CI) | Raw p | Holm p | HC ΔDice | CKD ΔDice |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for _, row in pairs.iterrows():
        lines.append(
            f"| {row.contrast} | {row.component} | {row.dice_difference:+.4f} "
            f"({row.dice_ci95_low:+.4f} to {row.dice_ci95_high:+.4f}) | {row.p_raw:.4g} "
            f"| {row.p_holm:.4g} | {row.hc_dice_difference:+.4f} | {row.ckd_dice_difference:+.4f} |"
        )
    lines.extend([
        "",
        "Dice was the primary endpoint. Confidence intervals used 10,000 cohort-stratified subject bootstrap samples; p values used 100,000 paired sign permutations and were adjusted across the four locked contrasts with the Holm procedure. All five methods used identical folds, label assignments, seeds, training budget, augmentations, and optimization settings.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the locked v2.3 ablation and TKV measurement endpoints.")
    parser.add_argument("--prior-archive", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--permutation-draws", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--tkv-only", action="store_true", help="Permit absent v2.3 runs and analyze only prior TKV data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.prior_archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = AnalysisSettings(args.bootstrap_draws, args.permutation_draws, args.seed)

    prior_tables = {name: read_archive_method(archive, method_id) for name, method_id in PRIOR_METHODS.items()}
    for name, table in prior_tables.items():
        validate_method_table(table, name)
    tkv_status = tkv_analysis(prior_tables, out_dir, settings)

    ablation_tables = {
        "B0+": prior_tables["B0+"],
        "Full CSD-MT": prior_tables["Full CSD-MT"],
    }
    missing: dict[str, str] = {}
    if args.ablation_root:
        root = args.ablation_root.resolve()
        for name, method_id in NEW_METHODS.items():
            table = read_directory_method(root, method_id)
            if table.empty:
                missing[name] = str(root / method_id)
            else:
                validate_method_table(table, name)
                ablation_tables[name] = table
    else:
        missing = {name: "--ablation-root not supplied" for name in NEW_METHODS}

    if missing and not args.tkv_only:
        raise FileNotFoundError(f"Formal v2.3 ablation is incomplete: {missing}")
    ablation_status = (
        ablation_analysis(ablation_tables, out_dir, settings)
        if not missing
        else {
            "status": "pending_autodl_results",
            "missing_methods": missing,
            "required_new_runs": 45,
            "table6_generated": False,
            "figure8_data_generated": False,
        }
    )
    provenance = {
        "schema_version": 1,
        "independent_unit": "subject_after_within_subject_acquisition_and_seed_averaging",
        "prior_archive": str(archive),
        "prior_archive_sha256": sha256(archive),
        "ablation_root": str(args.ablation_root.resolve()) if args.ablation_root else None,
        "bootstrap_draws": settings.bootstrap_draws,
        "permutation_draws": settings.permutation_draws,
        "seed": settings.seed,
        "tkv": tkv_status,
        "formal_ablation": ablation_status,
        "locked_gate_unchanged": "Q2_GATE_NOT_MET",
    }
    (out_dir / "analysis_status.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()

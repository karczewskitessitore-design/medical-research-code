from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage, stats

from kidney_ssl.metrics import dice_score, icc_2_1


def surface_dice(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    tolerance_mm: float = 2.0,
) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    structure = ndimage.generate_binary_structure(3, 1)
    pred_surface = prediction ^ ndimage.binary_erosion(prediction, structure=structure)
    target_surface = target ^ ndimage.binary_erosion(target, structure=structure)
    distance_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distance_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    numerator = (distance_target[pred_surface] <= tolerance_mm).sum()
    numerator += (distance_pred[target_surface] <= tolerance_mm).sum()
    denominator = pred_surface.sum() + target_surface.sum()
    return float(numerator / max(denominator, 1))


def calibration_metrics(
    foreground_probability: np.ndarray,
    target: np.ndarray,
    bins: int = 15,
) -> dict[str, float]:
    probability = np.asarray(foreground_probability, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    probability = np.clip(probability, 1e-7, 1.0 - 1e-7)
    brier = float(np.mean((probability - truth) ** 2))
    nll = float(-np.mean(truth * np.log(probability) + (1.0 - truth) * np.log(1.0 - probability)))
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probability >= boundaries[index]) & (probability <= boundaries[index + 1])
        else:
            selected = (probability >= boundaries[index]) & (probability < boundaries[index + 1])
        if selected.any():
            ece += selected.mean() * abs(probability[selected].mean() - truth[selected].mean())
    return {"brier": brier, "nll": nll, "ece": float(ece)}


def error_detection_metrics(uncertainty: np.ndarray, error: np.ndarray) -> dict[str, float | None]:
    score = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    truth = np.asarray(error, dtype=bool).reshape(-1)
    positives = int(truth.sum())
    negatives = int((~truth).sum())
    if positives == 0 or negatives == 0:
        return {"error_auroc": None, "error_auprc": None}
    ranks = stats.rankdata(score, method="average")
    auc = (ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    order = np.argsort(-score, kind="stable")
    sorted_truth = truth[order]
    cumulative = np.cumsum(sorted_truth)
    precision = cumulative / np.arange(1, len(sorted_truth) + 1)
    average_precision = float(precision[sorted_truth].sum() / positives)
    return {"error_auroc": float(auc), "error_auprc": average_precision}


def concordance_correlation_coefficient(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    covariance = np.mean((reference - reference.mean()) * (estimate - estimate.mean()))
    denominator = reference.var() + estimate.var() + (reference.mean() - estimate.mean()) ** 2
    return float(2.0 * covariance / denominator) if denominator > 0 else float("nan")


def repeated_measurement_summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    repeated: dict[str, list[tuple[str, float, float]]] = {}
    for row in rows:
        repeated.setdefault(str(row["subject_id"]), []).append(
            (
                str(row["acquisition_id"]),
                float(row["pred_tkv_ml"]),
                float(row["true_tkv_ml"]),
            )
        )
    repeated = {
        subject: sorted(values)
        for subject, values in repeated.items()
        if len(values) > 1
    }
    predicted_cvs = []
    true_cvs = []
    predicted_matrix = []
    residuals = []
    method_differences = []
    for values in repeated.values():
        predicted = np.asarray([value[1] for value in values], dtype=float)
        truth = np.asarray([value[2] for value in values], dtype=float)
        if predicted.mean() > 0:
            predicted_cvs.append(predicted.std(ddof=1) / predicted.mean() * 100.0)
        if truth.mean() > 0:
            true_cvs.append(truth.std(ddof=1) / truth.mean() * 100.0)
        predicted_matrix.append(predicted)
        residuals.extend((predicted - predicted.mean()).tolist())
        method_differences.extend((predicted - truth).tolist())
    matrix = np.asarray(predicted_matrix, dtype=float)
    icc = icc_2_1(matrix) if matrix.ndim == 2 and matrix.shape[0] >= 2 else float("nan")
    residuals_array = np.asarray(residuals, dtype=float)
    degrees = sum(max(len(values) - 1, 0) for values in repeated.values())
    within_sd = float(np.sqrt(np.sum(residuals_array**2) / degrees)) if degrees else float("nan")
    differences = np.asarray(method_differences, dtype=float)
    bias = float(differences.mean()) if differences.size else float("nan")
    # Repeated-measures limits use the within-subject residual variance rather
    # than treating every acquisition pair as independent.
    loa_sd = float(np.sqrt(np.mean((differences - bias) ** 2))) if differences.size else float("nan")
    truth_all = np.asarray(
        [value[2] for values in repeated.values() for value in values], dtype=float
    )
    predicted_all = np.asarray(
        [value[1] for values in repeated.values() for value in values], dtype=float
    )
    ccc = (
        concordance_correlation_coefficient(truth_all, predicted_all)
        if truth_all.size
        else float("nan")
    )
    return {
        "repeat_subjects": len(repeated),
        "within_subject_cv_percent_mean": float(np.mean(predicted_cvs)) if predicted_cvs else None,
        "manual_within_subject_cv_percent_mean": float(np.mean(true_cvs)) if true_cvs else None,
        "icc_2_1": float(icc) if np.isfinite(icc) else None,
        "ccc_vs_manual": float(ccc) if np.isfinite(ccc) else None,
        "within_subject_sd_ml": within_sd if np.isfinite(within_sd) else None,
        "repeatability_coefficient_ml": 2.77 * within_sd if np.isfinite(within_sd) else None,
        "bland_altman_repeated_bias_ml": bias if np.isfinite(bias) else None,
        "bland_altman_repeated_lower_loa_ml": bias - 1.96 * loa_sd if np.isfinite(loa_sd) else None,
        "bland_altman_repeated_upper_loa_ml": bias + 1.96 * loa_sd if np.isfinite(loa_sd) else None,
        "independent_unit": "subject",
    }


def risk_coverage_curve(
    uncertainty: np.ndarray,
    error: np.ndarray,
    points: int = 20,
) -> list[dict[str, float]]:
    uncertainty = np.asarray(uncertainty, dtype=float).reshape(-1)
    error = np.asarray(error, dtype=float).reshape(-1)
    order = np.argsort(uncertainty)
    result = []
    for coverage in np.linspace(0.05, 1.0, points):
        count = max(1, int(round(len(order) * coverage)))
        result.append({"coverage": float(coverage), "risk": float(error[order[:count]].mean())})
    return result

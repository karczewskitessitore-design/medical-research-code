from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from scipy import ndimage


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    denominator = prediction.sum() + target.sum()
    return 1.0 if denominator == 0 else float(2.0 * np.logical_and(prediction, target).sum() / denominator)


def surface_distances(
    prediction: np.ndarray, target: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[float, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    finite_cap = float(np.linalg.norm(np.asarray(prediction.shape) * np.asarray(spacing)))
    if not prediction.any() and not target.any():
        return 0.0, 0.0
    if not prediction.any() or not target.any():
        return finite_cap, finite_cap
    structure = ndimage.generate_binary_structure(3, 1)
    pred_surface = prediction ^ ndimage.binary_erosion(prediction, structure=structure, border_value=0)
    target_surface = target ^ ndimage.binary_erosion(target, structure=structure, border_value=0)
    distance_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([distance_to_target[pred_surface], distance_to_pred[target_surface]])
    return float(np.percentile(distances, 95)), float(distances.mean())


def volume_ml(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    return float(mask.astype(np.float64).sum() * np.prod(spacing) / 1000.0)


def icc_2_1(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_cols = n * np.sum((col_means - grand) ** 2) / (k - 1)
    residual = matrix - row_means[:, None] - col_means[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator != 0 else float("nan")


def repeatability_summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    by_subject: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        by_subject.setdefault(str(row["subject_id"]), []).append(
            (str(row["acquisition_id"]), float(row["pred_tkv_ml"]))
        )
    repeated = {key: sorted(value) for key, value in by_subject.items() if len(value) > 1}
    cvs = []
    pair_differences = []
    for values in repeated.values():
        volumes = np.asarray([value for _, value in values], dtype=float)
        mean = volumes.mean()
        if mean > 0:
            cvs.append(float(volumes.std(ddof=1) / mean * 100.0))
        pair_differences.extend(a - b for a, b in itertools.combinations(volumes.tolist(), 2))
    common_k = min((len(value) for value in repeated.values()), default=0)
    matrix = np.asarray(
        [[value for _, value in values[:common_k]] for values in repeated.values()], dtype=float
    )
    icc = icc_2_1(matrix) if matrix.ndim == 2 and matrix.size else float("nan")
    differences = np.asarray(pair_differences, dtype=float)
    mean_difference = float(differences.mean()) if differences.size else float("nan")
    sd = float(differences.std(ddof=1)) if differences.size > 1 else float("nan")
    return {
        "repeat_subjects": len(repeated),
        "within_subject_cv_percent_mean": float(np.mean(cvs)) if cvs else None,
        "icc_2_1": icc if np.isfinite(icc) else None,
        "bland_altman_pairwise_mean_difference_ml": mean_difference if np.isfinite(mean_difference) else None,
        "bland_altman_lower_loa_ml": mean_difference - 1.96 * sd if np.isfinite(sd) else None,
        "bland_altman_upper_loa_ml": mean_difference + 1.96 * sd if np.isfinite(sd) else None,
        "bland_altman_pairs": int(differences.size),
    }


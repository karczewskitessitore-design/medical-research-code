from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kidney_ssl.utils import sha256_file


def _stratified_subject_order(subjects: pd.DataFrame, seed: int) -> dict[tuple[str, bool], list[str]]:
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, bool], list[str]] = {}
    for (cohort, repeat), group in subjects.groupby(["cohort", "is_repeat"]):
        values = sorted(group.subject_id.tolist())
        rng.shuffle(values)
        groups[(str(cohort), bool(repeat))] = values
    return groups


def make_folds(manifest_path: str | Path, seed: int = 20260802) -> Path:
    manifest_path = Path(manifest_path).resolve()
    manifest = pd.read_csv(manifest_path)
    required = {"subject_id", "cohort", "is_repeat"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(manifest.columns))}")
    manifest["is_repeat"] = manifest["is_repeat"].astype(str).str.lower().eq("true")
    subjects = manifest[["subject_id", "cohort", "is_repeat"]].drop_duplicates()
    groups = _stratified_subject_order(subjects, seed)
    if {key: len(value) for key, value in groups.items()} != {
        ("CKD", False): 25,
        ("CKD", True): 5,
        ("HC", False): 25,
        ("HC", True): 5,
    }:
        raise ValueError(f"Unexpected cohort/repeat structure: { {key: len(value) for key, value in groups.items()} }")

    subject_to_fold: dict[str, int] = {}
    for values in groups.values():
        for index, subject_id in enumerate(values):
            subject_to_fold[subject_id] = index % 5
    assignments = subjects.copy()
    assignments["test_fold"] = assignments.subject_id.map(subject_to_fold).astype(int)

    out = manifest_path.parent.parent / "splits" / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    assignments.sort_values("subject_id").to_csv(out / "subject_folds.csv", index=False)
    manifest_with_folds = manifest.merge(assignments[["subject_id", "test_fold"]], on="subject_id")
    manifest_with_folds.to_csv(out / "manifest_with_folds.csv", index=False)

    for fold in range(5):
        training = assignments[assignments.test_fold != fold].copy()
        ordered = _stratified_subject_order(training, seed + 1000 + fold)
        selected: dict[float, set[str]] = {
            0.125: set(),
            0.25: set(),
            0.50: set(),
            1.00: set(),
        }
        for (_, repeat), values in ordered.items():
            n = len(values)
            count125 = 1 if repeat else 2
            count25 = 1 if repeat else 5
            count50 = 2 if repeat else 10
            if n != (4 if repeat else 20):
                raise ValueError(f"Unexpected training stratum size in fold {fold}: {n}")
            selected[0.125].update(values[:count125])
            selected[0.25].update(values[:count25])
            selected[0.50].update(values[:count50])
            selected[1.00].update(values)
        if not selected[0.125] < selected[0.25] < selected[0.50] < selected[1.00]:
            raise AssertionError("Label subsets are not strictly nested")
        rows = []
        for fraction, values in selected.items():
            rows.extend({"subject_id": sid, "labeled_fraction": fraction} for sid in sorted(values))
        pd.DataFrame(rows).to_csv(out / f"labeled_fold{fold}.csv", index=False)

    metadata = {
        "seed": seed,
        "n_folds": 5,
        "split_unit": "subject",
        "label_fractions": [0.125, 0.25, 0.5, 1.0],
    }
    (out / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    validate_folds(out)
    split_files = sorted(path for path in out.iterdir() if path.is_file() and path.name != "split_files_sha256.csv")
    pd.DataFrame(
        [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in split_files]
    ).to_csv(out / "split_files_sha256.csv", index=False)
    return out


def validate_folds(split_dir: str | Path) -> None:
    split_dir = Path(split_dir)
    folds = pd.read_csv(split_dir / "subject_folds.csv")
    if folds.subject_id.duplicated().any() or set(folds.test_fold) != set(range(5)):
        raise AssertionError("Invalid subject fold assignments")
    folds["is_repeat"] = folds.is_repeat.astype(str).str.lower().eq("true")
    for fold in range(5):
        test = folds[folds.test_fold == fold]
        table = test.groupby(["cohort", "is_repeat"]).size().to_dict()
        expected = {("CKD", False): 5, ("CKD", True): 1, ("HC", False): 5, ("HC", True): 1}
        if table != expected:
            raise AssertionError(f"Fold {fold} has wrong structure: {table}")
        labels = pd.read_csv(split_dir / f"labeled_fold{fold}.csv")
        fractions = (0.125, 0.25, 0.5, 1.0) if 0.125 in set(labels.labeled_fraction) else (0.25, 0.5, 1.0)
        sets = {
            fraction: set(labels[labels.labeled_fraction == fraction].subject_id)
            for fraction in fractions
        }
        expected_sizes = [6, 12, 24, 48] if 0.125 in sets else [12, 24, 48]
        if [len(sets[f]) for f in fractions] != expected_sizes:
            raise AssertionError(f"Fold {fold} has wrong label subset sizes")
        if not all(sets[left] < sets[right] for left, right in zip(fractions, fractions[1:])):
            raise AssertionError(f"Fold {fold} label subsets are not nested")
        if set(test.subject_id) & sets[1.0]:
            raise AssertionError(f"Fold {fold} test leakage")

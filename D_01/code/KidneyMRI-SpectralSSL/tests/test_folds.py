from __future__ import annotations

from pathlib import Path

import pandas as pd

from kidney_ssl.data.folds import make_folds, validate_folds


def test_folds_are_subject_level_balanced_and_nested(tmp_path: Path):
    rows = []
    for cohort in ("HC", "CKD"):
        for index in range(30):
            subject = f"{cohort}_{index:02d}"
            repeats = 5 if index < 5 else 1
            for acquisition in range(repeats):
                rows.append(
                    {
                        "subject_id": subject,
                        "cohort": cohort,
                        "is_repeat": index < 5,
                        "acquisition_id": acquisition,
                    }
                )
    audit = tmp_path / "audit"
    audit.mkdir()
    manifest = audit / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    split_dir = make_folds(manifest, 20260802)
    validate_folds(split_dir)
    assignments = pd.read_csv(split_dir / "manifest_with_folds.csv")
    assert assignments.groupby("subject_id").test_fold.nunique().max() == 1


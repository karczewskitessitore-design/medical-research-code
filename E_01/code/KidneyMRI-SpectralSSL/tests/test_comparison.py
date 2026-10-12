from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from kidney_ssl.comparison import compare_benchmark, holm_adjust


def _write_method(root: Path, method: str, dice_shift: float, tkv_shift: float):
    for fold in range(2):
        run = root / method / f"fold{fold}_seed11"
        run.mkdir(parents=True)
        rows = []
        for index in range(6):
            rows.append(
                {
                    "subject_id": f"S{fold}_{index}",
                    "cohort": "HC" if index < 3 else "CKD",
                    "is_repeat": index in {0, 3},
                    "dice": 0.8 + dice_shift + index * 0.001,
                    "hd95_mm": 10.0,
                    "asd_mm": 2.0,
                    "tkv_relative_error": 0.1 + tkv_shift,
                }
            )
        pd.DataFrame(rows).to_csv(run / "metrics_subject.csv", index=False)


def test_holm_is_monotone_in_sorted_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted[0] <= adjusted[2] <= adjusted[1]


def test_comparison_claim_guard(tmp_path: Path):
    root = tmp_path / "confirmatory"
    _write_method(root, "B2", 0.10, 0.0)
    _write_method(root, "baseline", 0.0, 0.0)
    suite = {
        "name": "test",
        "proposed_method": "B2",
        "methods": [
            {"id": "B2", "sci_baseline": False},
            {
                "id": "baseline",
                "family": "Journal baseline",
                "journal": "Test Journal",
                "sci_baseline": True,
                "primary_for_family": True,
            },
        ],
    }
    suite_path = root / "suite.yaml"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    result = compare_benchmark(root, suite_path, draws=2000, seed=3)
    summary = json.loads(result.read_text(encoding="utf-8"))
    assert summary["outperformed_count"] == 1
    assert "显著优于1个" in summary["claim"]


def test_conference_method_is_compared_but_not_counted_as_sci(tmp_path: Path):
    root = tmp_path / "confirmatory"
    _write_method(root, "B2", 0.10, 0.0)
    _write_method(root, "journal", 0.0, 0.0)
    _write_method(root, "conference", 0.0, 0.0)
    suite = {
        "name": "test",
        "proposed_method": "B2",
        "methods": [
            {"id": "B2", "sci_baseline": False},
            {
                "id": "journal",
                "family": "Journal baseline",
                "sci_baseline": True,
                "primary_for_family": True,
            },
            {
                "id": "conference",
                "family": "Conference baseline",
                "comparison_baseline": True,
                "sci_baseline": False,
                "primary_for_family": True,
            },
        ],
    }
    suite_path = root / "suite.yaml"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    result = compare_benchmark(root, suite_path, draws=2000, seed=3)
    summary = json.loads(result.read_text(encoding="utf-8"))
    assert summary["method_outperformed_count"] == 2
    assert summary["outperformed_count"] == 1
    assert summary["method_families_planned"] == 2
    assert summary["sci_families_planned"] == 1
    sci = pd.read_csv(root / "paired_sci_comparisons.csv")
    all_methods = pd.read_csv(root / "paired_method_comparisons.csv")
    assert sci["baseline"].tolist() == ["journal"]
    assert set(all_methods["baseline"]) == {"journal", "conference"}


def test_incomplete_confirmatory_matrix_cannot_claim_superiority(tmp_path: Path):
    root = tmp_path / "confirmatory"
    _write_method(root, "B2", 0.10, 0.0)
    _write_method(root, "baseline", 0.0, 0.0)
    suite = {
        "name": "test",
        "proposed_method": "B2",
        "expected_subjects": 12,
        "methods": [
            {"id": "B2", "sci_baseline": False},
            {
                "id": "baseline",
                "family": "Journal baseline",
                "sci_baseline": True,
                "primary_for_family": True,
            },
        ],
        "phases": {"confirmatory": {"folds": [0, 1], "seeds": [11, 12]}},
    }
    suite_path = root / "suite.yaml"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    result = compare_benchmark(root, suite_path, draws=100, seed=3)
    summary = json.loads(result.read_text(encoding="utf-8"))
    assert summary["outperformed_count"] == 0
    assert not summary["confirmatory_complete"]
    assert summary["comparisons"][0]["status"] == "incomplete"

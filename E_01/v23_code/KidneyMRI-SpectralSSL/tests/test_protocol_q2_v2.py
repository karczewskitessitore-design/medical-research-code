from __future__ import annotations

import json
from pathlib import Path

import yaml

from kidney_ssl.q2_comparison import compare_q2_gate


def test_q2_gate_fails_closed_when_preregistered_results_are_missing(tmp_path: Path) -> None:
    suite = {
        "schema_version": 2,
        "name": "test",
        "proposed_method": "proposed",
        "methods": [
            {"id": "proposed", "config": "unused.yaml", "phases": ["internal", "external"]},
            {
                "id": "nnunet_3d_default",
                "config": "unused.yaml",
                "phases": ["internal", "external"],
                "nnunet_official_3d": True,
                "external_reference": True,
            },
        ],
        "phases": {"external": {"datasets": ["chaos", "amos"]}},
        "q2_gate": {
            "internal_min_dice": 0.88,
            "internal_min_effect": 0.01,
            "tkv_max_relative_error": 0.10,
            "repeat_max_wcv_percent": 4.0,
            "repeat_max_dice_tradeoff": 0.005,
            "external_noninferiority_margin": -0.01,
            "full_label_min_dice": 0.90,
            "bootstrap_draws": 10,
            "permutation_draws": 20,
        },
    }
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    result_path = compare_q2_gate(tmp_path, suite_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "Q2_GATE_NOT_MET"
    assert not result["q2_gate_passed"]
    report = (tmp_path / "Q2_投稿门槛评估.md").read_text(encoding="utf-8")
    assert "不得宣称具备 Q2 投稿条件" in report


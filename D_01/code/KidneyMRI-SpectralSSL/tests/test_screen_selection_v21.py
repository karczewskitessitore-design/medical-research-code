from __future__ import annotations

import json
from pathlib import Path

from kidney_ssl.benchmark_v2 import _screen_decision


def _summary(dice: float, hd95: float) -> dict:
    return {
        "overall": {"dice": dice, "hd95_mm": hd95},
        "repeatability": {"within_subject_cv_percent_mean": 2.0},
    }


def test_configurable_five_slice_screen_selection(tmp_path: Path) -> None:
    values = {
        "V21_B0plus_5slice": (0.880, 8.0),
        "V21_UABand_5slice": (0.887, 7.8),
        "V21_CSD_5slice": (0.895, 7.0),
        "KidneyBand_ReliableSSL_v2_1": (0.896, 6.9),
    }
    methods = []
    for method, (dice, hd95) in values.items():
        methods.append({"id": method, "phases": ["screen"]})
        for fold in (0, 1):
            run = tmp_path / method / f"fold{fold}_seed20260802"
            run.mkdir(parents=True)
            (run / "evaluation_summary.json").write_text(
                json.dumps(_summary(dice, hd95)), encoding="utf-8"
            )

    suite = {
        "methods": methods,
        "phases": {"screen": {"folds": [0, 1], "seeds": [20260802]}},
        "screen_selection": {
            "baseline_method": "V21_B0plus_5slice",
            "segmentation_modules": {
                "UABand": "V21_UABand_5slice",
                "CSD": "V21_CSD_5slice",
            },
            "repeat_module": None,
            "full_method": "KidneyBand_ReliableSSL_v2_1",
            "full_reference_methods": ["V21_UABand_5slice", "V21_CSD_5slice"],
            "stack_depth": 5,
            "required_modules": ["UABand", "CSD"],
        },
        "selection_rules": {
            "segmentation_min_dice_gain": 0.005,
            "segmentation_max_hd95_degradation_mm": 1.0,
            "repeat_min_wcv_gain_percent": 0.5,
            "repeat_max_dice_degradation": 0.005,
            "full_max_degradation_from_best_pair": 0.003,
        },
    }

    result_path = _screen_decision(tmp_path, suite)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    assert result["stack_depth_selected"] == 5
    assert result["selection_baseline"] == "V21_B0plus_5slice"
    assert result["module_decisions"]["UABand"]["keep"] is True
    assert result["module_decisions"]["CSD"]["keep"] is True
    assert "DZ-BiMorph" not in result["module_decisions"]
    assert result["module_decisions"]["full_model"]["keep"] is True
    assert result["formal_method_ready"] is True

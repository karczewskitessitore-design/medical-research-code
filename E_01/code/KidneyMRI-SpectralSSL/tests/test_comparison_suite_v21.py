from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kidney_ssl.benchmark_v2 import _run_table_path, _selected_methods
from kidney_ssl.config import load_config


def _suite() -> tuple[Path, dict]:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "autodl" / "q2_v21_comparison.yaml"
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))


def test_v21_comparison_suite_is_complete_and_explicit() -> None:
    path, suite = _suite()
    methods = suite["methods"]
    assert suite["proposed_method"] == "KidneyBand_ReliableSSL_v2_1"
    assert suite["matched_baseline"] == "V21_B0plus_5slice"
    assert suite["label_efficiency"]["enabled"] is False
    assert len(methods) == 11
    assert len({method["id"] for method in methods}) == 11
    assert suite["phases"]["internal"]["folds"] == [0, 1, 2, 3, 4]
    assert suite["phases"]["internal"]["seeds"] == [20260811, 20260812, 20260813]

    by_id = {method["id"]: method for method in methods}
    assert by_id["frcnet_5slice"]["comparison_baseline"] is True
    assert by_id["frcnet_5slice"]["sci_baseline"] is False
    assert by_id["nnunet_3d_default"]["implementation_type"] == "official_package"
    for method in ("frcnet_5slice", "semi_urf_5slice", "randdg_5slice"):
        assert by_id[method]["implementation_type"] == "clean_room_proxy"
        assert "not" in by_id[method]["protocol_deviation"].lower()

    five_slice = {
        "V21_B0plus_5slice",
        "KidneyBand_ReliableSSL_v2_1",
        "unetpp_5slice",
        "tcsm_5slice",
        "cmmt_5slice",
        "frcnet_5slice",
        "semi_urf_5slice",
        "randdg_5slice",
    }
    for method_id in five_slice:
        method = by_id[method_id]
        config_path = (path.parent / method["config"]).resolve()
        config = load_config(config_path)
        assert config["model"]["in_channels"] == 5


def test_method_filtering_is_validated_and_has_distinct_run_tables(tmp_path: Path) -> None:
    _, suite = _suite()
    selected = _selected_methods(
        suite, "internal", ["cmmt_5slice", "V21_B0plus_5slice"]
    )
    assert [method["id"] for method in selected] == [
        "V21_B0plus_5slice",
        "cmmt_5slice",
    ]
    with pytest.raises(ValueError, match="unavailable"):
        _selected_methods(suite, "internal", ["missing_method"])
    assert _run_table_path(tmp_path, "internal", None).name == "internal_runs.csv"
    shard = _run_table_path(tmp_path, "internal", ["cmmt_5slice", "tcsm_5slice"])
    assert shard.name == "internal_runs__cmmt_5slice-tcsm_5slice.csv"

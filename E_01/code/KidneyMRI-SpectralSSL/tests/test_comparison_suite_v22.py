from __future__ import annotations

from pathlib import Path

import yaml

from kidney_ssl.config import load_config


def _suite(name: str) -> tuple[Path, dict]:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "autodl" / name
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))


def test_v22_formal_suite_freezes_csd_only_selection() -> None:
    path, suite = _suite("q2_v22_csd_comparison.yaml")
    assert suite["proposed_method"] == "KidneyBand_ReliableSSL_v2_2_CSD"
    assert suite["matched_baseline"] == "V22_B0plus_5slice"
    assert suite["development_selection"]["selected_method"] == "V21_CSD_5slice"
    assert suite["development_selection"]["observed"]["uaband_keep"] is False
    assert suite["development_selection"]["observed"]["full_v21_formal_method_ready"] is False
    assert suite["label_efficiency"]["enabled"] is False
    assert suite["phases"]["internal"]["folds"] == [0, 1, 2, 3, 4]
    assert suite["phases"]["internal"]["seeds"] == [20260811, 20260812, 20260813]

    methods = suite["methods"]
    by_id = {method["id"]: method for method in methods}
    assert len(methods) == 11
    assert len(by_id) == 11
    assert by_id["frcnet_5slice"]["comparison_baseline"] is True
    assert by_id["frcnet_5slice"]["sci_baseline"] is False
    assert by_id["nnunet_3d_default"]["implementation_type"] == "official_package"

    proposed_path = (path.parent / by_id[suite["proposed_method"]]["config"]).resolve()
    proposed = load_config(proposed_path)
    assert proposed["model"]["in_channels"] == 5
    assert proposed["spectral_disagreement"]["enabled"] is True
    assert proposed["adaptive_band"]["enabled"] is False
    assert proposed["repeat_morphometry"]["enabled"] is False


def test_v22_smoke_suite_matches_formal_non_nnunet_methods() -> None:
    _, formal = _suite("q2_v22_csd_comparison.yaml")
    _, smoke = _suite("q2_v22_csd_comparison_smoke.yaml")
    formal_ids = {method["id"] for method in formal["methods"]}
    smoke_ids = {method["id"] for method in smoke["methods"]}
    assert smoke_ids == formal_ids - {"nnunet_3d_default"}
    assert smoke["proposed_method"] == formal["proposed_method"]
    assert smoke["matched_baseline"] == formal["matched_baseline"]
    assert smoke["config_overrides"]["optimizer"]["epochs"] == 1
    assert smoke["config_overrides"]["runtime"]["max_steps_per_epoch"] == 2


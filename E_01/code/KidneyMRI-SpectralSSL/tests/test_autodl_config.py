from __future__ import annotations

from pathlib import Path

from kidney_ssl.config import load_config


def test_config_overrides_replace_paths_without_changing_method_definition() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source = project_root / "configs" / "v2" / "full.yaml"
    config = load_config(
        source,
        overrides={
            "data": {
                "root": "/root/autodl-tmp/kidney/data/T2KidneyMRI",
                "manifest": "/root/autodl-tmp/kidney/data/T2KidneyMRI/audit/manifest.csv",
            },
            "runtime": {"num_workers": 4},
        },
    )

    assert config["data"]["root"] == "/root/autodl-tmp/kidney/data/T2KidneyMRI"
    assert config["data"]["manifest"].endswith("/audit/manifest.csv")
    assert config["runtime"]["num_workers"] == 4
    assert config["adaptive_band"]["enabled"] is True
    assert config["spectral_disagreement"]["enabled"] is True
    assert config["repeat_morphometry"]["enabled"] is True
    assert config["optimizer"]["learning_rate"] == 0.0003

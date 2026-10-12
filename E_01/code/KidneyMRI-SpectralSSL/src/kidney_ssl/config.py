from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REQUIRED_BLOCKS = (
    "data",
    "split",
    "model",
    "spectral_band",
    "mean_teacher",
    "repeat_tkv",
    "optimizer",
    "runtime",
)

V2_REQUIRED_BLOCKS = (
    "augmentation",
    "bilateral_labels",
    "adaptive_band",
    "spectral_disagreement",
    "repeat_morphometry",
    "external_validation",
)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(
    path: str | Path, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    base_ref = current.pop("_base_", None)
    config: dict[str, Any] = {}
    if base_ref:
        config = load_config(path.parent / base_ref)
    config = _merge(config, current)
    if overrides:
        config = _merge(config, overrides)
    missing = [key for key in REQUIRED_BLOCKS if key not in config]
    if missing:
        raise ValueError(f"Configuration is missing required blocks: {missing}")
    if int(config.get("schema_version", 1)) == 2:
        missing_v2 = [key for key in V2_REQUIRED_BLOCKS if key not in config]
        if missing_v2:
            raise ValueError(f"Schema v2 configuration is missing blocks: {missing_v2}")
        if int(config["model"].get("out_channels", 0)) != 3:
            raise ValueError("Schema v2 requires model.out_channels: 3")
    return config


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

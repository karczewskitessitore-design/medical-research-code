from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from kidney_ssl.config import load_config
from kidney_ssl.utils import atomic_json, sha256_file


def _project_root(suite_path: Path) -> Path:
    for candidate in (suite_path.parent, *suite_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate project root above {suite_path}")


def _inventory(paths: list[Path], root: Path) -> list[dict[str, str]]:
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    return [
        {
            "path": path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path),
            "sha256": sha256_file(path),
        }
        for path in unique
    ]


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_protocol_lock(suite_path: str | Path) -> dict[str, Any]:
    suite_path = Path(suite_path).resolve()
    root = _project_root(suite_path)
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    if int(suite.get("schema_version", 0)) != 2:
        raise ValueError("Q2 protocol locking requires suite schema_version: 2")
    configs = []
    data_files = []
    method_definitions = []
    for method in suite.get("methods", []):
        config_path = (suite_path.parent / str(method["config"])).resolve()
        config = load_config(config_path, overrides=suite.get("config_overrides"))
        if int(config.get("schema_version", 1)) != 2:
            raise ValueError(f"Method {method['id']} does not use schema_version 2")
        configs.append(config_path)
        for key in ("manifest",):
            candidate = Path(str(config["data"].get(key, "")))
            if candidate.is_file():
                data_files.append(candidate)
        folds = Path(str(config["data"].get("folds", "")))
        if folds.is_dir():
            data_files.extend(sorted(folds.glob("*.csv")))
            data_files.extend(sorted(folds.glob("*.json")))
        method_definitions.append(
            {
                "id": str(method["id"]),
                "family": str(method.get("family", method["id"])),
                "config": config_path.relative_to(root).as_posix(),
                "config_sha256": sha256_file(config_path),
                "phases": method.get("phases", ["screen", "internal", "external"]),
                "reference": {
                    key: method.get(key)
                    for key in (
                        "journal",
                        "venue",
                        "doi",
                        "official_code",
                        "license",
                        "implementation_type",
                        "protocol_deviation",
                    )
                    if method.get(key) is not None
                },
            }
        )
    source_paths = [root / "pyproject.toml", root / "uv.lock"]
    source_paths.extend(sorted((root / "src").rglob("*.py")))
    source_paths.extend(configs)
    source_inventory = _inventory(source_paths, root)
    data_inventory = _inventory(data_files, root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    lock = {
        "schema_version": 2,
        "suite": str(suite_path),
        "suite_sha256": sha256_file(suite_path),
        "git_commit": commit,
        "source_tree_sha256": _digest(source_inventory),
        "source_inventory": source_inventory,
        "data_inventory_sha256": _digest(data_inventory),
        "data_inventory": data_inventory,
        "methods": method_definitions,
        "phases": suite.get("phases", {}),
        "selection_rules": suite.get("selection_rules", {}),
        "q2_gate": suite.get("q2_gate", {}),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    lock["lock_sha256"] = _digest(lock)
    return lock


def lock_protocol(suite_path: str | Path) -> Path:
    suite_path = Path(suite_path).resolve()
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    output_root = Path(str(suite["output_root"])).resolve() / str(suite["name"])
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "protocol_lock_v2.json"
    current = build_protocol_lock(suite_path)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("lock_sha256") != current["lock_sha256"]:
            raise RuntimeError(
                f"Protocol is already locked and the suite/source/data changed: {path}. "
                "Create a new versioned suite instead of silently overwriting the lock."
            )
        return path
    atomic_json(current, path)
    return path


def validate_protocol_lock(suite_path: str | Path) -> Path:
    suite_path = Path(suite_path).resolve()
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    path = Path(str(suite["output_root"])).resolve() / str(suite["name"]) / "protocol_lock_v2.json"
    if not path.exists():
        raise RuntimeError("Protocol has not been locked; run kidney-seg lock-protocol first")
    previous = json.loads(path.read_text(encoding="utf-8"))
    current = build_protocol_lock(suite_path)
    if previous.get("lock_sha256") != current["lock_sha256"]:
        raise RuntimeError("Locked suite/source/data hash mismatch; execution is prohibited")
    return path

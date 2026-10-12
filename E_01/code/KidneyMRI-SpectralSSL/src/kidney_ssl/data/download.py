from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from kidney_ssl.utils import md5_file


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    url: str
    md5: str
    extract: bool


FILES = (
    DownloadSpec(
        "CKD.zip",
        "https://zenodo.org/api/records/5153568/files/CKD.zip/content",
        "8fe112954083212cbee80a6087db84c5",
        True,
    ),
    DownloadSpec(
        "Healthy_Control.zip",
        "https://zenodo.org/api/records/5153568/files/Healthy_Control.zip/content",
        "561badbc875874c5e69cf725a8545330",
        True,
    ),
    DownloadSpec(
        "subject_information.csv",
        "https://zenodo.org/api/records/5153568/files/subject_information.csv/content",
        "3bc3d595339cb7c1f9208a07f7335ef5",
        False,
    ),
)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise ValueError(f"Unsafe ZIP member: {member.filename}") from exc
        bundle.extractall(destination)


def _download(spec: DownloadSpec, target: Path) -> str:
    if target.exists():
        actual = md5_file(target)
        if actual != spec.md5:
            raise RuntimeError(
                f"Existing file is not overwritten and has wrong MD5: {target}; "
                f"expected {spec.md5}, got {actual}"
            )
        return "verified-existing"

    partial = target.with_suffix(target.suffix + ".part")
    try:
        with requests.get(spec.url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            with partial.open("wb") as handle:
                shutil.copyfileobj(response.raw, handle, length=1024 * 1024)
        actual = md5_file(partial)
        if actual != spec.md5:
            raise RuntimeError(
                f"MD5 mismatch for {spec.name}: expected {spec.md5}, got {actual}"
            )
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return "downloaded"


def download_dataset(data_root: str | Path) -> list[dict[str, str]]:
    root = Path(data_root).resolve()
    downloads = root / "downloads"
    raw = root / "raw"
    downloads.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, str]] = []
    for spec in FILES:
        target = downloads / spec.name
        status = _download(spec, target)
        if spec.extract:
            marker = raw / f".{spec.name}.extracted"
            if not marker.exists():
                _safe_extract(target, raw)
                marker.touch()
            extract_status = "verified-or-extracted"
        else:
            destination = root / spec.name
            if not destination.exists():
                shutil.copy2(target, destination)
            elif md5_file(destination) != spec.md5:
                raise RuntimeError(f"Existing metadata file has wrong MD5: {destination}")
            extract_status = "copied-metadata"
        outcomes.append(
            {"file": spec.name, "status": status, "postprocess": extract_status, "md5": spec.md5}
        )
    return outcomes

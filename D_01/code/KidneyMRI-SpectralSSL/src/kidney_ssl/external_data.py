from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import requests

from kidney_ssl.utils import atomic_json, sha256_file


@dataclass(frozen=True)
class ExternalArchive:
    dataset: str
    name: str
    url: str
    md5: str
    size: int
    license: str
    record_url: str


ARCHIVES = {
    "chaos": ExternalArchive(
        dataset="chaos",
        name="CHAOS_Train_Sets.zip",
        url="https://zenodo.org/api/records/3431873/files/CHAOS_Train_Sets.zip/content",
        md5="df21053002a1cc86df918a87da3b2c19",
        size=890_771_694,
        license="CC-BY-NC-SA-4.0",
        record_url="https://zenodo.org/records/3431873",
    ),
    "amos": ExternalArchive(
        dataset="amos",
        name="amos22.zip",
        url="https://zenodo.org/api/records/7262581/files/amos22.zip/content",
        md5="67717b2a483ac0744c89c3016b7aaef7",
        size=24_234_336_519,
        license="CC-BY-4.0",
        record_url="https://zenodo.org/records/7262581",
    ),
}


@contextmanager
def _exclusive_lock(path: Path):
    """Prevent concurrent writers from corrupting a resumable archive."""

    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        owner = path.read_text(encoding="utf-8", errors="replace") if path.exists() else "unknown"
        raise RuntimeError(f"External preparation is already locked: {path}; owner={owner}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("utf-8"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _validate_world_side_order(labels: np.ndarray, affine: np.ndarray, context: str) -> None:
    centroids = {}
    for value, name in ((1, "left"), (2, "right")):
        coordinates = np.argwhere(labels == value)
        if not len(coordinates):
            raise ValueError(f"{context}: missing {name} kidney label")
        centroids[name] = float(nib.affines.apply_affine(affine, coordinates).mean(axis=0)[0])
    # NIfTI RAS+ x increases toward anatomical Right, hence a left-kidney
    # centroid must have a smaller world-x coordinate than the right kidney.
    if not centroids["left"] < centroids["right"]:
        raise ValueError(
            f"{context}: bilateral semantics disagree with RAS world x; "
            f"left={centroids['left']:.3f}, right={centroids['right']:.3f}"
        )


def _download_resumable(spec: ExternalArchive, destination: Path) -> str:
    """Download a large archive without silently discarding a valid partial file."""

    if destination.exists():
        actual = _md5(destination)
        if actual != spec.md5:
            raise RuntimeError(
                f"Existing archive is not overwritten: {destination}; expected MD5 "
                f"{spec.md5}, got {actual}"
            )
        return "verified-existing"
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(spec.url, stream=True, headers=headers, timeout=(30, 180)) as response:
        response.raise_for_status()
        if existing and response.status_code != 206:
            raise RuntimeError(
                f"Server did not honor Range for existing partial archive {partial}; "
                "the partial file was preserved"
            )
        mode = "ab" if existing else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if partial.stat().st_size != spec.size:
        raise RuntimeError(
            f"Downloaded size mismatch for {spec.name}: expected {spec.size}, "
            f"got {partial.stat().st_size}; partial file was preserved"
        )
    actual = _md5(partial)
    if actual != spec.md5:
        raise RuntimeError(
            f"MD5 mismatch for {spec.name}: expected {spec.md5}, got {actual}; "
            "partial file was preserved for inspection"
        )
    partial.replace(destination)
    return "downloaded"


def _prepare_amos(extracted: Path, prepared: Path) -> list[dict[str, Any]]:
    image_files = sorted(extracted.rglob("imagesTr/amos_*.nii.gz"))
    if not image_files:
        image_files = sorted(extracted.rglob("imagesTr/amos_*.nii"))
    rows: list[dict[str, Any]] = []
    label_root = prepared / "labels_bilateral"
    label_root.mkdir(parents=True, exist_ok=True)
    for image_path in image_files:
        identifier = int(image_path.name.split("_")[-1].split(".")[0])
        # AMOS identifiers 0500--0599 are MRI. Only a subset has public
        # training labels in imagesTr/labelsTr; CT cases are intentionally
        # excluded from this source-only MRI evaluation manifest.
        if identifier < 500:
            continue
        label_path = next(iter(extracted.rglob(f"labelsTr/amos_{identifier:04d}.nii*")), None)
        if label_path is None:
            continue
        image = nib.load(str(image_path))
        label_image = nib.load(str(label_path))
        labels = np.asarray(label_image.dataobj)
        bilateral = np.zeros(labels.shape, dtype=np.uint8)
        bilateral[labels == 3] = 1  # official AMOS left kidney
        bilateral[labels == 2] = 2  # official AMOS right kidney
        _validate_world_side_order(bilateral, label_image.affine, f"AMOS {identifier}")
        output_label = label_root / f"amos_{identifier:04d}_kidneys.nii.gz"
        nib.save(nib.Nifti1Image(bilateral, label_image.affine, label_image.header), output_label)
        if image.shape[:3] != label_image.shape[:3]:
            raise ValueError(f"AMOS image/label shape mismatch for {identifier}")
        canonical = nib.as_closest_canonical(image)
        canonical_shape = tuple(int(value) for value in canonical.shape[:3])
        canonical_zooms = tuple(float(value) for value in canonical.header.get_zooms()[:3])
        slice_axis = int(np.argmax(canonical_zooms))
        if canonical_zooms[slice_axis] <= 1.2 * min(canonical_zooms):
            slice_axis = int(np.argmin(canonical_shape))
        model_shape = canonical_shape[:slice_axis] + canonical_shape[slice_axis + 1 :] + (canonical_shape[slice_axis],)
        rows.append(
            {
                "dataset": "amos",
                "subject_id": f"AMOS_{identifier:04d}",
                "acquisition_id": "MRI",
                "image_path": str(image_path.resolve()),
                "mask_path": str(output_label.resolve()),
                "shape": "x".join(str(int(value)) for value in model_shape),
                "left_value": 1,
                "right_value": 2,
                "source_left_value": 3,
                "source_right_value": 2,
                "cohort": "AMOS",
                "is_repeat": False,
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"No labeled AMOS MRI cases (identifier >= 0500) found below {extracted}"
        )
    return rows


def _dicom_stack(dicom_dir: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], list[Any]]:
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("Preparing CHAOS requires the pydicom dependency") from exc
    # Windows globbing is case-insensitive, so the two patterns can return the
    # same file twice. Resolve through a set before stacking.
    files = sorted({*dicom_dir.glob("*.dcm"), *dicom_dir.glob("*.DCM")})
    datasets = [pydicom.dcmread(str(path)) for path in files]
    if not datasets:
        raise FileNotFoundError(f"No DICOM files in {dicom_dir}")
    datasets.sort(key=lambda value: (int(getattr(value, "InstanceNumber", 0)), str(value.filename)))
    array = np.stack([np.asarray(value.pixel_array, dtype=np.float32) for value in datasets], axis=-1)
    first = datasets[0]
    pixel_spacing = tuple(float(value) for value in first.PixelSpacing)
    orientation = np.asarray(
        getattr(first, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), dtype=float
    )
    column_direction = orientation[:3]
    row_direction = orientation[3:]
    normal = np.cross(column_direction, row_direction)
    positions = [
        np.asarray(value.ImagePositionPatient, dtype=float)
        for value in datasets
        if hasattr(value, "ImagePositionPatient")
    ]
    slice_vector = None
    if len(positions) == len(datasets) and len(positions) > 1:
        differences = np.diff(np.stack(positions), axis=0)
        nonzero = differences[np.linalg.norm(differences, axis=1) > 1e-6]
        if len(nonzero):
            slice_vector = np.median(nonzero, axis=0)
    fallback_spacing = float(
        getattr(first, "SpacingBetweenSlices", getattr(first, "SliceThickness", 1.0))
    )
    if slice_vector is None or np.linalg.norm(slice_vector) <= 1e-6:
        slice_vector = normal * max(abs(fallback_spacing), 1e-3)
    slice_spacing = float(np.linalg.norm(slice_vector))
    origin = positions[0] if positions else np.zeros(3, dtype=float)
    affine_lps = np.eye(4, dtype=float)
    # NumPy axis 0 is DICOM row and axis 1 is DICOM column.
    affine_lps[:3, 0] = row_direction * pixel_spacing[0]
    affine_lps[:3, 1] = column_direction * pixel_spacing[1]
    affine_lps[:3, 2] = slice_vector
    affine_lps[:3, 3] = origin
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine = lps_to_ras @ affine_lps
    return array, affine, (pixel_spacing[0], pixel_spacing[1], slice_spacing), datasets


def _prepare_chaos(extracted: Path, prepared: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_dirs = sorted(path for path in extracted.rglob("T2SPIR") if path.is_dir())
    output_images = prepared / "images"
    output_labels = prepared / "labels_bilateral"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    for t2_dir in case_dirs:
        dicom_dir = t2_dir / "DICOM_anon"
        ground_dir = t2_dir / "Ground"
        if not dicom_dir.is_dir() or not ground_dir.is_dir():
            continue
        subject_id = next((part for part in t2_dir.parts[::-1] if part.isdigit()), t2_dir.parent.name)
        image_array, affine, _, datasets = _dicom_stack(dicom_dir)
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Preparing CHAOS requires the Pillow dependency") from exc
        ground_files = sorted({*ground_dir.glob("*.png"), *ground_dir.glob("*.PNG")})
        if len(ground_files) != image_array.shape[-1]:
            raise ValueError(
                f"CHAOS {subject_id}: {len(ground_files)} labels for {image_array.shape[-1]} slices"
            )
        # Ground-truth filenames encode DICOM instance numbers. Match by the
        # integer suffix where possible, with sorted order as a checked fallback.
        by_instance = {
            int(getattr(dataset, "InstanceNumber", index + 1)): index
            for index, dataset in enumerate(datasets)
        }
        label_array = np.zeros(image_array.shape, dtype=np.uint8)
        for fallback_index, ground_path in enumerate(ground_files):
            digits = "".join(character for character in ground_path.stem if character.isdigit())
            target_index = by_instance.get(int(digits), fallback_index) if digits else fallback_index
            raw = np.asarray(Image.open(ground_path))
            # CHAOS MRI encoding: 63 liver, 126 right kidney,
            # 189 left kidney, 252 spleen. The side assignment is additionally
            # audited in RAS world coordinates after conversion.
            label_array[:, :, target_index][raw == 189] = 1
            label_array[:, :, target_index][raw == 126] = 2
        if not np.any(label_array == 1) or not np.any(label_array == 2):
            raise ValueError(f"CHAOS {subject_id}: both kidney labels were not found")
        _validate_world_side_order(label_array, affine, f"CHAOS {subject_id}")
        image_path = output_images / f"CHAOS_T2_{subject_id}.nii.gz"
        label_path = output_labels / f"CHAOS_T2_{subject_id}_kidneys.nii.gz"
        nib.save(nib.Nifti1Image(image_array, affine), image_path)
        nib.save(nib.Nifti1Image(label_array, affine), label_path)
        rows.append(
            {
                "dataset": "chaos",
                "subject_id": f"CHAOS_{subject_id}",
                "acquisition_id": "T2SPIR",
                "image_path": str(image_path.resolve()),
                "mask_path": str(label_path.resolve()),
                "shape": "x".join(str(int(value)) for value in image_array.shape),
                "left_value": 1,
                "right_value": 2,
                "source_left_value": 189,
                "source_right_value": 126,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No labeled CHAOS T2-SPIR cases found below {extracted}")
    return rows


def prepare_external(dataset: str, data_root: str | Path, download: bool = True) -> Path:
    dataset = dataset.lower()
    if dataset not in ARCHIVES:
        raise ValueError("dataset must be chaos or amos")
    spec = ARCHIVES[dataset]
    root = Path(data_root).resolve()
    downloads = root / "downloads"
    extracted = root / "raw"
    prepared = root / "prepared"
    downloads.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)
    prepared.mkdir(parents=True, exist_ok=True)
    archive = downloads / spec.name
    with _exclusive_lock(downloads / f".{dataset}.prepare.lock"):
        status = _download_resumable(spec, archive) if download else "download-disabled"
    if not archive.exists():
        raise FileNotFoundError(
            f"Expected archive {archive}. Re-run without --no-download or place the official archive there."
        )
    marker = extracted / f".{spec.name}.extracted.json"
    if not marker.exists():
        _safe_extract(archive, extracted)
        atomic_json({"archive": str(archive), "md5": spec.md5}, marker)
    rows = _prepare_chaos(extracted, prepared) if dataset == "chaos" else _prepare_amos(extracted, prepared)
    manifest = prepared / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    inventory = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in sorted(prepared.rglob("*"))
        if path.is_file() and path != prepared / "inventory.json"
    ]
    atomic_json(
        {
            "dataset": dataset,
            "archive": asdict(spec),
            "download_status": status,
            "cases": len(rows),
            "manifest": str(manifest),
            "files": inventory,
        },
        prepared / "inventory.json",
    )
    return manifest

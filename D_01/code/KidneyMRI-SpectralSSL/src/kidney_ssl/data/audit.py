from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd

from kidney_ssl.utils import sha256_file


NAME_RE = re.compile(
    r"^(?P<cohort>CKD|HC)_(?P<subject>.+?)_T2W(?:_(?P<acq>\d+))?(?P<mask>_mask)?\.nii(?:\.gz)?$",
    re.IGNORECASE,
)


def _parse(path: Path) -> dict[str, str] | None:
    match = NAME_RE.match(path.name)
    if not match:
        return None
    cohort = match.group("cohort").upper()
    subject_raw = match.group("subject")
    return {
        "cohort": cohort,
        "subject_id": f"{cohort}_{subject_raw}",
        "acquisition_id": match.group("acq") or "1",
        "is_mask": str(bool(match.group("mask"))),
    }


def _vector(values: tuple[int, ...] | tuple[float, ...]) -> str:
    return "x".join(str(value) for value in values)


def audit_dataset(data_root: str | Path, strict_expected_counts: bool = True) -> dict[str, Any]:
    root = Path(data_root).resolve()
    raw = root / "raw"
    if not raw.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw}; run download first")
    out = root / "audit"
    out.mkdir(parents=True, exist_ok=True)

    nii_files = sorted({*raw.rglob("*.nii"), *raw.rglob("*.nii.gz")})
    parsed: dict[tuple[str, str], dict[str, Path]] = {}
    anomalies: list[str] = []
    hash_rows: list[dict[str, str]] = []
    auxiliary_files: list[Path] = []
    if (root / "downloads").exists():
        auxiliary_files.extend((root / "downloads").glob("*"))
    if (root / "subject_information.csv").exists():
        auxiliary_files.append(root / "subject_information.csv")
    for path in sorted(path for path in auxiliary_files if path.is_file()):
        hash_rows.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    for path in nii_files:
        info = _parse(path)
        hash_rows.append({"path": str(path), "sha256": sha256_file(path)})
        if info is None:
            anomalies.append(f"UNRECOGNIZED_NAME\t{path}")
            continue
        key = (info["subject_id"], info["acquisition_id"])
        slot = "mask" if info["is_mask"] == "True" else "image"
        if slot in parsed.setdefault(key, {}):
            anomalies.append(f"DUPLICATE_{slot.upper()}\t{key}\t{path}")
        parsed[key][slot] = path

    preliminary: list[dict[str, Any]] = []
    for (subject_id, acquisition_id), pair in sorted(parsed.items()):
        if "image" not in pair or "mask" not in pair:
            anomalies.append(f"UNPAIRED\t{subject_id}\t{acquisition_id}\t{pair}")
            continue
        image = nib.load(str(pair["image"]))
        mask = nib.load(str(pair["mask"]))
        image_canonical = nib.as_closest_canonical(image)
        canonical_shape = tuple(int(v) for v in image_canonical.shape)
        canonical_spacing = tuple(float(v) for v in image_canonical.header.get_zooms()[:3])
        slice_axis = int(np.argmax(canonical_spacing))
        if canonical_spacing[slice_axis] <= 1.2 * min(canonical_spacing):
            slice_axis = int(np.argmin(canonical_shape))
        model_shape = tuple(np.moveaxis(np.empty(canonical_shape, dtype=np.uint8), slice_axis, -1).shape)
        model_spacing_list = list(canonical_spacing)
        through_plane_spacing = model_spacing_list.pop(slice_axis)
        model_spacing = tuple([*model_spacing_list, through_plane_spacing])
        shape_ok = image.shape == mask.shape
        affine_ok = bool(np.allclose(image.affine, mask.affine, atol=1e-4))
        spacing_image = tuple(float(v) for v in image.header.get_zooms()[:3])
        spacing_mask = tuple(float(v) for v in mask.header.get_zooms()[:3])
        spacing_ok = bool(np.allclose(spacing_image, spacing_mask, atol=1e-5))
        if not shape_ok:
            anomalies.append(f"SHAPE_MISMATCH\t{subject_id}\t{acquisition_id}")
        if not affine_ok:
            anomalies.append(f"AFFINE_MISMATCH\t{subject_id}\t{acquisition_id}")
        if not spacing_ok:
            anomalies.append(f"SPACING_MISMATCH\t{subject_id}\t{acquisition_id}")
        values = np.unique(np.asarray(mask.dataobj))
        labels_ok = bool(np.all(np.isin(values, [0, 1])))
        if not labels_ok:
            anomalies.append(
                f"UNEXPECTED_LABELS\t{subject_id}\t{acquisition_id}\t{values.tolist()}"
            )
        preliminary.append(
            {
                "subject_id": subject_id,
                "cohort": subject_id.split("_", 1)[0],
                "acquisition_id": acquisition_id,
                "image_path": str(pair["image"].resolve()),
                "mask_path": str(pair["mask"].resolve()),
                "shape": _vector(model_shape),
                "spacing": _vector(model_spacing),
                "ras_slice_axis": slice_axis,
                "affine_ok": affine_ok,
                "labels_ok": labels_ok,
            }
        )

    manifest = pd.DataFrame(preliminary)
    if manifest.empty:
        raise RuntimeError("No valid image/mask pairs found")
    counts = manifest.groupby("subject_id").size().to_dict()
    manifest["n_acquisitions_subject"] = manifest["subject_id"].map(counts).astype(int)
    manifest["is_repeat"] = manifest["n_acquisitions_subject"] > 1
    manifest = manifest.sort_values(["cohort", "subject_id", "acquisition_id"])
    manifest.to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(hash_rows).to_csv(out / "files_sha256.csv", index=False)

    subjects = manifest[["subject_id", "cohort", "is_repeat", "n_acquisitions_subject"]].drop_duplicates()
    summary = {
        "subjects": int(subjects.subject_id.nunique()),
        "acquisitions": int(len(manifest)),
        "hc_subjects": int((subjects.cohort == "HC").sum()),
        "ckd_subjects": int((subjects.cohort == "CKD").sum()),
        "hc_repeat_subjects": int(((subjects.cohort == "HC") & subjects.is_repeat).sum()),
        "ckd_repeat_subjects": int(((subjects.cohort == "CKD") & subjects.is_repeat).sum()),
        "anomalies": len(anomalies),
    }
    expected = {
        "subjects": 60,
        "acquisitions": 100,
        "hc_subjects": 30,
        "ckd_subjects": 30,
        "hc_repeat_subjects": 5,
        "ckd_repeat_subjects": 5,
    }
    for key, value in expected.items():
        if summary[key] != value:
            anomalies.append(f"COUNT_MISMATCH\t{key}\texpected={value}\tactual={summary[key]}")
    (out / "audit_anomalies.log").write_text(
        "\n".join(anomalies) + ("\n" if anomalies else ""), encoding="utf-8"
    )
    summary["anomalies"] = len(anomalies)
    pd.DataFrame([summary]).to_csv(out / "audit_summary.csv", index=False)
    if strict_expected_counts and anomalies:
        raise RuntimeError(f"Audit found {len(anomalies)} anomaly/anomalies; see {out / 'audit_anomalies.log'}")
    return summary

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np


def discover_chaos_t2(root: str | Path, require_labels: bool = True) -> list[dict[str, str]]:
    root = Path(root).resolve()
    cases: list[dict[str, str]] = []
    for t2_dir in sorted(path for path in root.rglob("*T2*SPIR*") if path.is_dir()):
        dicom_files = sorted({*t2_dir.rglob("*.dcm"), *t2_dir.rglob("*.DCM")})
        if not dicom_files:
            continue
        subject_id = next((part for part in t2_dir.parts[::-1] if part.isdigit()), t2_dir.parent.name)
        label_candidates = sorted(root.rglob(f"*{subject_id}*T2*label*.nii*"))
        if require_labels and not label_candidates:
            raise FileNotFoundError(f"Missing T2-SPIR ground truth for CHAOS subject {subject_id}")
        cases.append(
            {
                "subject_id": subject_id,
                "dicom_dir": str(t2_dir),
                "label_path": str(label_candidates[0]) if label_candidates else "",
            }
        )
    if not cases:
        raise FileNotFoundError(f"No CHAOS T2-SPIR DICOM directories found below {root}")
    return cases


def merge_kidney_labels(label_image: nib.Nifti1Image, left_values=(1,), right_values=(2,)) -> nib.Nifti1Image:
    data = np.asarray(label_image.dataobj)
    merged = np.isin(data, [*left_values, *right_values]).astype(np.uint8)
    return nib.Nifti1Image(merged, label_image.affine, label_image.header)


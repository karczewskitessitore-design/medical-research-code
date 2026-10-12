from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from kidney_ssl.data.audit import audit_dataset


def test_audit_pairs_and_flags_counts_without_strict_failure(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    image = np.random.default_rng(2).normal(size=(8, 8, 3)).astype(np.float32)
    mask = np.zeros((8, 8, 3), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    nib.save(nib.Nifti1Image(image, np.eye(4)), raw / "HC_01_T2W_1.nii.gz")
    nib.save(nib.Nifti1Image(mask, np.eye(4)), raw / "HC_01_T2W_1_mask.nii.gz")
    summary = audit_dataset(tmp_path, strict_expected_counts=False)
    assert summary["subjects"] == 1
    assert summary["acquisitions"] == 1
    assert (tmp_path / "audit" / "manifest.csv").exists()
    assert (tmp_path / "audit" / "files_sha256.csv").exists()


def test_audit_logs_affine_spacing_and_label_anomalies(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    image = np.ones((8, 8, 3), dtype=np.float32)
    mask = np.full((8, 8, 3), 2, dtype=np.uint8)
    nib.save(nib.Nifti1Image(image, np.eye(4)), raw / "CKD_01_T2W.nii.gz")
    changed_affine = np.diag([2.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(mask, changed_affine), raw / "CKD_01_T2W_mask.nii.gz")
    audit_dataset(tmp_path, strict_expected_counts=False)
    anomalies = (tmp_path / "audit" / "audit_anomalies.log").read_text(encoding="utf-8")
    assert "AFFINE_MISMATCH" in anomalies
    assert "SPACING_MISMATCH" in anomalies
    assert "UNEXPECTED_LABELS" in anomalies

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from kidney_ssl.data.chaos import discover_chaos_t2, merge_kidney_labels


def test_chaos_sequence_discovery_and_label_merge(tmp_path: Path):
    dicom_dir = tmp_path / "Train" / "001" / "T2-SPIR"
    dicom_dir.mkdir(parents=True)
    (dicom_dir / "slice001.dcm").touch()
    label_path = tmp_path / "001_T2_label.nii.gz"
    values = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
    nib.save(nib.Nifti1Image(values, np.eye(4)), label_path)
    cases = discover_chaos_t2(tmp_path)
    assert len(cases) == 1 and cases[0]["label_path"] == str(label_path)
    merged = np.asarray(merge_kidney_labels(nib.load(label_path)).dataobj)
    assert merged.tolist() == [[[0, 1, 1, 0]]]


def test_chaos_missing_truth_is_explicit(tmp_path: Path):
    dicom_dir = tmp_path / "002" / "T2-SPIR"
    dicom_dir.mkdir(parents=True)
    (dicom_dir / "slice.dcm").touch()
    with pytest.raises(FileNotFoundError, match="ground truth"):
        discover_chaos_t2(tmp_path, require_labels=True)


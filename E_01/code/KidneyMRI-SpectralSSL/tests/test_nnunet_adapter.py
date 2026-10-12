from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from kidney_ssl.nnunet_adapter import prepare_nnunet_dataset, prepare_nnunet_external_input


def _case(root: Path, subject: str, acquisition: str) -> dict:
    affine = np.diag([1.2, 1.3, 4.0, 1.0])
    image = root / f"{subject}_{acquisition}.nii.gz"
    mask = root / f"{subject}_{acquisition}_mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((8, 9, 3), dtype=np.float32), affine), image)
    nib.save(nib.Nifti1Image(np.ones((8, 9, 3), dtype=np.uint8), affine), mask)
    return {
        "subject_id": subject,
        "cohort": subject.split("_")[0],
        "acquisition_id": acquisition,
        "image_path": str(image),
        "mask_path": str(mask),
        "shape": "8x9x3",
        "spacing": "1.2x1.3x4.0",
        "is_repeat": False,
    }


def test_nnunet_conversion_preserves_geometry_and_ids(tmp_path: Path):
    split = tmp_path / "splits"
    split.mkdir()
    rows = [_case(tmp_path, "HC_01", "1"), _case(tmp_path, "CKD_01", "1")]
    manifest = pd.DataFrame(rows)
    manifest["test_fold"] = [1, 0]
    manifest.to_csv(split / "manifest_with_folds.csv", index=False)
    pd.DataFrame([{"subject_id": "HC_01", "labeled_fraction": 0.25}]).to_csv(
        split / "labeled_fold0.csv", index=False
    )
    config = {
        "data": {"folds": str(split)},
        "split": {"labeled_fraction": 0.25},
        "model": {"options": {"dataset_id": 501}},
    }
    prepared = prepare_nnunet_dataset(config, 0, tmp_path / "run")
    converted = next((prepared["raw"] / prepared["dataset_name"] / "imagesTr").glob("*.nii.gz"))
    original = nib.load(rows[0]["image_path"])
    linked = nib.load(str(converted))
    assert original.shape == linked.shape
    assert np.allclose(original.affine, linked.affine)
    assert prepared["mapping"].iloc[0].subject_id == "CKD_01"


def test_nnunet_external_input_is_canonical_and_has_stable_ids(tmp_path: Path):
    row = _case(tmp_path, "CHAOS_01", "T2-SPIR")
    destination = tmp_path / "external_input"
    mapping = prepare_nnunet_external_input(pd.DataFrame([row]), destination)
    assert mapping.iloc[0].case_id == "External_Kidney_CHAOS_01_T2_SPIR"
    converted = nib.load(str(destination / f"{mapping.iloc[0].case_id}_0000.nii.gz"))
    assert nib.aff2axcodes(converted.affine) == ("R", "A", "S")

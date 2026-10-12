from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from kidney_ssl.data.dataset import KidneySliceDataset, VolumeStore


def _nifti(path: Path, data: np.ndarray):
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)


def test_unlabeled_item_has_no_hidden_label(tmp_path: Path):
    image = tmp_path / "HC_01_T2W_1.nii.gz"
    mask = tmp_path / "HC_01_T2W_1_mask.nii.gz"
    _nifti(image, np.random.default_rng(1).normal(size=(8, 8, 3)).astype(np.float32))
    _nifti(mask, np.ones((8, 8, 3), dtype=np.uint8))
    records = pd.DataFrame(
        [
            {
                "subject_id": "HC_01",
                "acquisition_id": "1",
                "image_path": str(image),
                "mask_path": str(mask),
                "shape": "8x8x3",
            }
        ]
    )
    dataset = KidneySliceDataset(records, labeled_subjects=set(), store=VolumeStore((8, 8)))
    item = dataset[0]
    assert "mask" not in item
    assert "mask_path" not in item
    assert not item["is_labeled"]
    assert all("mask" not in key.lower() for key in item)


from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from kidney_ssl.data.dataset import VolumeStore
from kidney_ssl.data.dataset_v2 import BilateralKidneySliceDataset


def test_unlabeled_v2_item_never_passes_mask_path_to_store(tmp_path: Path) -> None:
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 2), dtype=np.float32), np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 2), dtype=np.uint8), np.eye(4)), mask_path)
    records = pd.DataFrame(
        [
            {
                "subject_id": "S1",
                "acquisition_id": "A1",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "shape": "8x8x2",
            }
        ]
    )
    store = VolumeStore((8, 8), cache=False)
    dataset = BilateralKidneySliceDataset(records, labeled_subjects=[], store=store)
    item = dataset[0]
    assert "labels" not in item
    assert "mask" not in item
    assert "mask_path" not in item
    assert not item["is_labeled"]


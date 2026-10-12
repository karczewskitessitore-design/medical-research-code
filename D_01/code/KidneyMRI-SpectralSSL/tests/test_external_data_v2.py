from __future__ import annotations

import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from kidney_ssl.external_data import (
    _exclusive_lock,
    _prepare_amos,
    _safe_extract,
    _validate_world_side_order,
)


def test_prepare_amos_keeps_only_labeled_mri_and_maps_sides(tmp_path: Path) -> None:
    images = tmp_path / "raw" / "amos22" / "imagesTr"
    labels = tmp_path / "raw" / "amos22" / "labelsTr"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    affine = np.eye(4)
    for identifier in (42, 500):
        image = np.zeros((8, 7, 4), dtype=np.float32)
        label = np.zeros_like(image, dtype=np.uint8)
        label[1:3, 1:3, 1:3] = 3
        label[5:7, 4:6, 1:3] = 2
        nib.save(nib.Nifti1Image(image, affine), images / f"amos_{identifier:04d}.nii.gz")
        nib.save(nib.Nifti1Image(label, affine), labels / f"amos_{identifier:04d}.nii.gz")
    rows = _prepare_amos(tmp_path / "raw", tmp_path / "prepared")
    assert len(rows) == 1
    assert rows[0]["subject_id"] == "AMOS_0500"
    mapped = np.asarray(nib.load(rows[0]["mask_path"]).dataobj)
    assert set(np.unique(mapped)) == {0, 1, 2}
    assert np.all(mapped[1:3, 1:3, 1:3] == 1)
    assert np.all(mapped[5:7, 4:6, 1:3] == 2)


def test_external_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        _safe_extract(archive, tmp_path / "out")


def test_external_lock_rejects_a_second_writer_and_cleans_up(tmp_path: Path) -> None:
    lock = tmp_path / "archive.lock"
    with _exclusive_lock(lock):
        assert lock.exists()
        with pytest.raises(RuntimeError, match="already locked"):
            with _exclusive_lock(lock):
                pass
    assert not lock.exists()


def test_world_side_order_is_fail_closed() -> None:
    labels = np.zeros((8, 4, 2), dtype=np.uint8)
    labels[1:3] = 1
    labels[5:7] = 2
    _validate_world_side_order(labels, np.eye(4), "correct")
    with pytest.raises(ValueError, match="disagree with RAS"):
        _validate_world_side_order(labels, np.diag([-1.0, 1.0, 1.0, 1.0]), "reversed")

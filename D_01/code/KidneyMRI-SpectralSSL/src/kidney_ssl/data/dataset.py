from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler


@dataclass
class VolumeData:
    image: torch.Tensor
    mask: torch.Tensor | None
    spacing_resampled: tuple[float, float, float]
    model_original_shape: tuple[int, int, int]
    canonical_shape: tuple[int, int, int]
    ras_slice_axis: int
    canonical_affine: np.ndarray
    original_image: nib.Nifti1Image


def _preprocess_image(array: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    nonzero = array[array != 0]
    if nonzero.size:
        low, high = np.percentile(nonzero, percentiles)
        array = np.clip(array, low, high)
        foreground = array[array != 0]
    else:
        foreground = array.reshape(-1)
    mean = float(foreground.mean()) if foreground.size else 0.0
    std = float(foreground.std()) if foreground.size else 1.0
    return (array - mean) / max(std, 1e-6)


def _resize_volume(array: np.ndarray, size: tuple[int, int], mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).unsqueeze(1).float()
    kwargs = {"size": size, "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    resized = F.interpolate(tensor, **kwargs)
    return resized[:, 0].permute(1, 2, 0).contiguous()


class VolumeStore:
    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
        clip_percentiles: tuple[float, float] = (0.5, 99.5),
        cache: bool = True,
    ) -> None:
        self.image_size = image_size
        self.clip_percentiles = clip_percentiles
        self.cache_enabled = cache
        self.cache: dict[str, VolumeData] = {}

    def load(self, record: dict[str, Any], load_mask: bool) -> VolumeData:
        key = f"{record['image_path']}|{load_mask}"
        if self.cache_enabled and key in self.cache:
            return self.cache[key]
        original = nib.load(str(record["image_path"]))
        canonical = nib.as_closest_canonical(original)
        canonical_array = np.asarray(canonical.dataobj)
        canonical_shape = tuple(int(v) for v in canonical.shape[:3])
        canonical_spacing = tuple(float(v) for v in canonical.header.get_zooms()[:3])
        slice_axis = int(np.argmax(canonical_spacing))
        if canonical_spacing[slice_axis] <= 1.2 * min(canonical_spacing):
            slice_axis = int(np.argmin(canonical_shape))
        image_np = _preprocess_image(
            np.moveaxis(canonical_array, slice_axis, -1), self.clip_percentiles
        )
        model_original_shape = tuple(int(v) for v in image_np.shape)
        spacing_list = list(canonical_spacing)
        through_plane_spacing = spacing_list.pop(slice_axis)
        model_spacing = tuple([*spacing_list, through_plane_spacing])
        image = _resize_volume(image_np, self.image_size, "bilinear")
        spacing = (
            model_spacing[0] * model_original_shape[0] / self.image_size[0],
            model_spacing[1] * model_original_shape[1] / self.image_size[1],
            model_spacing[2],
        )
        mask = None
        if load_mask:
            mask_original = nib.load(str(record["mask_path"]))
            mask_canonical = nib.as_closest_canonical(mask_original)
            mask_array = np.moveaxis(np.asarray(mask_canonical.dataobj), slice_axis, -1)
            mask = _resize_volume((mask_array > 0).astype(np.float32), self.image_size, "nearest")
        volume = VolumeData(
            image=image,
            mask=mask,
            spacing_resampled=spacing,
            model_original_shape=model_original_shape,
            canonical_shape=canonical_shape,
            ras_slice_axis=slice_axis,
            canonical_affine=canonical.affine.copy(),
            original_image=original,
        )
        if self.cache_enabled:
            self.cache[key] = volume
        return volume


def adjacent_stack(volume: torch.Tensor, slice_index: int, stack_depth: int = 3) -> torch.Tensor:
    if stack_depth < 1 or stack_depth % 2 == 0:
        raise ValueError("stack_depth must be a positive odd integer")
    depth = volume.shape[-1]
    radius = stack_depth // 2
    indices = [
        max(0, min(depth - 1, slice_index + offset))
        for offset in range(-radius, radius + 1)
    ]
    return torch.stack([volume[:, :, index] for index in indices], dim=0)


class KidneySliceDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        labeled_subjects: Iterable[str],
        store: VolumeStore,
        stack_depth: int = 3,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.labeled_subjects = set(labeled_subjects)
        self.store = store
        self.stack_depth = int(stack_depth)
        self.samples: list[tuple[int, int]] = []
        self.subject_slice_counts: Counter[str] = Counter()
        for record_index, row in self.records.iterrows():
            shape = tuple(int(value) for value in str(row["shape"]).split("x"))
            for slice_index in range(shape[2]):
                self.samples.append((record_index, slice_index))
                self.subject_slice_counts[str(row.subject_id)] += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, slice_index = self.samples[index]
        row = self.records.iloc[record_index]
        labeled = str(row.subject_id) in self.labeled_subjects
        safe_record = {
            "image_path": row.image_path,
            **({"mask_path": row.mask_path} if labeled else {}),
        }
        volume = self.store.load(safe_record, load_mask=labeled)
        item: dict[str, Any] = {
            "image": adjacent_stack(volume.image, slice_index, self.stack_depth),
            "subject_id": str(row.subject_id),
            "acquisition_id": str(row.acquisition_id),
            "slice_index": slice_index,
            "is_labeled": labeled,
        }
        if labeled:
            if volume.mask is None:
                raise AssertionError("Labeled sample has no mask")
            item["mask"] = volume.mask[:, :, slice_index].unsqueeze(0)
        return item

    def balanced_sampler(self, seed: int) -> WeightedRandomSampler:
        weights = []
        for record_index, _ in self.samples:
            subject_id = str(self.records.iloc[record_index].subject_id)
            weights.append(1.0 / self.subject_slice_counts[subject_id])
        generator = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)


def collate_slices(items: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "image": torch.stack([item["image"] for item in items]),
        "subject_id": [item["subject_id"] for item in items],
        "acquisition_id": [item["acquisition_id"] for item in items],
        "slice_index": [item["slice_index"] for item in items],
    }
    labeled_indices = [index for index, item in enumerate(items) if item["is_labeled"]]
    output["labeled_indices"] = torch.tensor(labeled_indices, dtype=torch.long)
    if labeled_indices:
        output["mask"] = torch.stack([items[index]["mask"] for index in labeled_indices])
    return output


def volume_batch(
    store: VolumeStore,
    record: dict[str, Any],
    load_mask: bool = False,
    stack_depth: int = 3,
) -> tuple[torch.Tensor, VolumeData]:
    safe = {"image_path": record["image_path"]}
    if load_mask:
        safe["mask_path"] = record["mask_path"]
    volume = store.load(safe, load_mask=load_mask)
    slices = torch.stack(
        [adjacent_stack(volume.image, index, stack_depth) for index in range(volume.image.shape[-1])]
    )
    return slices, volume

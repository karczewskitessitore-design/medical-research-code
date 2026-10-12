from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from kidney_ssl.bilateral import split_binary_kidneys
from kidney_ssl.data.dataset import VolumeStore, adjacent_stack


class BilateralKidneySliceDataset(Dataset):
    """V2 slice dataset that never loads a mask for an unlabeled subject."""

    def __init__(
        self,
        records: pd.DataFrame,
        labeled_subjects: Iterable[str],
        store: VolumeStore,
        stack_depth: int = 3,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.labeled_subjects = {str(value) for value in labeled_subjects}
        self.store = store
        self.stack_depth = int(stack_depth)
        self.samples: list[tuple[int, int]] = []
        self.subject_slice_counts: Counter[str] = Counter()
        for record_index, row in self.records.iterrows():
            depth = int(str(row["shape"]).split("x")[-1])
            for slice_index in range(depth):
                self.samples.append((record_index, slice_index))
                self.subject_slice_counts[str(row.subject_id)] += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, slice_index = self.samples[index]
        row = self.records.iloc[record_index]
        labeled = str(row.subject_id) in self.labeled_subjects
        safe_record = {"image_path": row.image_path}
        if labeled:
            safe_record["mask_path"] = row.mask_path
        volume = self.store.load(safe_record, load_mask=labeled)
        item: dict[str, Any] = {
            "image": adjacent_stack(volume.image, slice_index, self.stack_depth),
            "subject_id": str(row.subject_id),
            "acquisition_id": str(row.acquisition_id),
            "slice_index": int(slice_index),
            "is_labeled": bool(labeled),
        }
        if labeled:
            if volume.mask is None:
                raise AssertionError("A labeled sample was loaded without a mask")
            binary = volume.mask[:, :, slice_index].cpu().numpy() > 0.5
            item["labels"] = torch.from_numpy(
                split_binary_kidneys(binary, lr_axis=0).astype(np.int64)
            )
        return item

    def balanced_sampler(self, seed: int) -> WeightedRandomSampler:
        weights = []
        for record_index, _ in self.samples:
            subject = str(self.records.iloc[record_index].subject_id)
            weights.append(1.0 / self.subject_slice_counts[subject])
        return WeightedRandomSampler(
            weights,
            len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )


def collate_bilateral_slices(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = len(items)
    height, width = items[0]["image"].shape[-2:]
    labels = torch.zeros((batch, height, width), dtype=torch.long)
    labeled_indices = []
    for index, item in enumerate(items):
        if item["is_labeled"]:
            labeled_indices.append(index)
            labels[index] = item["labels"]
    return {
        "image": torch.stack([item["image"] for item in items]),
        "labels": labels,
        "labeled_indices": torch.as_tensor(labeled_indices, dtype=torch.long),
        "subject_id": [item["subject_id"] for item in items],
        "acquisition_id": [item["acquisition_id"] for item in items],
        "slice_index": [item["slice_index"] for item in items],
    }

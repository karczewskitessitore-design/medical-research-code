from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from kidney_ssl.bilateral import audit_bilateral_mask, split_binary_kidneys
from kidney_ssl.utils import atomic_json


def audit_bilateral_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    minimum_side_fraction: float = 0.20,
    minimum_two_component_fraction: float = 0.95,
) -> Path:
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    rows = []
    anomalies = []
    for record in manifest.to_dict("records"):
        mask_image = nib.as_closest_canonical(nib.load(str(record["mask_path"])))
        mask = np.asarray(mask_image.dataobj) > 0
        audit = audit_bilateral_mask(mask, lr_axis=0)
        labels = split_binary_kidneys(mask, lr_axis=0)
        left = int((labels == 1).sum())
        right = int((labels == 2).sum())
        left_world_x = float(
            nib.affines.apply_affine(mask_image.affine, np.argwhere(labels == 1)).mean(axis=0)[0]
        )
        right_world_x = float(
            nib.affines.apply_affine(mask_image.affine, np.argwhere(labels == 2)).mean(axis=0)[0]
        )
        volume_ratio = min(left, right) / max(left, right, 1)
        reasons = []
        if min(audit.left_fraction, audit.right_fraction) < minimum_side_fraction:
            reasons.append("side_fraction")
        if audit.dominant_component_fraction < minimum_two_component_fraction:
            reasons.append("fragmentation")
        if audit.crosses_midline:
            reasons.append("midline_crossing")
        if not left_world_x < right_world_x:
            reasons.append("ras_side_order")
        row = {
            "subject_id": str(record["subject_id"]),
            "acquisition_id": str(record["acquisition_id"]),
            "components": audit.components,
            "two_largest_component_fraction": audit.dominant_component_fraction,
            "left_fraction": audit.left_fraction,
            "right_fraction": audit.right_fraction,
            "smaller_to_larger_volume_ratio": volume_ratio,
            "left_centroid_world_x": left_world_x,
            "right_centroid_world_x": right_world_x,
            "crosses_midline": audit.crosses_midline,
            "status": "review" if reasons else "pass",
            "reasons": ";".join(reasons),
        }
        rows.append(row)
        if reasons:
            anomalies.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "bilateral_mask_audit.csv", index=False)
    atomic_json(
        {
            "manifest": str(manifest_path),
            "acquisitions": len(rows),
            "passed": len(rows) - len(anomalies),
            "review_required": len(anomalies),
            "thresholds": {
                "minimum_side_fraction": minimum_side_fraction,
                "minimum_two_component_fraction": minimum_two_component_fraction,
            },
            "anomalies": anomalies,
            "interpretation": (
                "Review flags do not automatically exclude a case. Bilateral classes are defined by "
                "the canonical RAS left/right axis and exclusions require a documented protocol amendment."
            ),
        },
        output_dir / "bilateral_mask_audit_summary.json",
    )
    return output_dir / "bilateral_mask_audit_summary.json"

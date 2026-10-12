#!/usr/bin/env python3
"""22_make_amos_splits.py

为 AMOS22 MRI 生成 subject 级 5 折与嵌套标签子集（10%/20%/50%/100%），
输出布局与 kidney_ssl.data.folds.make_folds 兼容（训练管线只依赖
manifest_with_folds.csv + labeled_fold{fold}.csv）。

用法:
  python scripts/autodl/22_make_amos_splits.py \
      --manifest /root/autodl-tmp/kidney/data/AMOS22/prepared/manifest.csv \
      --out /root/autodl-tmp/kidney/data/AMOS22/splits/seed_20260821 \
      --seed 20260821
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FRACTIONS = (0.10, 0.20, 0.50, 1.00)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = pd.read_csv(manifest_path)
    required = {"subject_id", "acquisition_id", "image_path", "mask_path", "shape"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise SystemExit(f"AMOS manifest missing columns: {missing}")
    subjects = manifest[["subject_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(subjects))
    # 先打乱再顺序分 5 折
    shuffled = subjects.iloc[order].reset_index(drop=True)
    shuffled["test_fold"] = shuffled.index % 5
    assignments = shuffled.sort_values("subject_id").reset_index(drop=True)
    counts = assignments.test_fold.value_counts().sort_index().to_dict()
    print(f"subjects={len(assignments)} fold_counts={counts}")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(out / "subject_folds.csv", index=False)
    manifest_with_folds = manifest.merge(assignments[["subject_id", "test_fold"]], on="subject_id")
    manifest_with_folds.to_csv(out / "manifest_with_folds.csv", index=False)

    for fold in range(5):
        training = assignments[assignments.test_fold != fold].sort_values("subject_id")
        ids = training.subject_id.tolist()
        rng2 = np.random.default_rng(args.seed + 1000 + fold)
        order2 = rng2.permutation(len(ids))
        ordered = [ids[i] for i in order2]
        n = len(ordered)
        counts_by_fraction = [int(round(n * f)) for f in FRACTIONS]
        counts_by_fraction[-1] = n
        for i in range(len(counts_by_fraction) - 1):
            counts_by_fraction[i] = min(counts_by_fraction[i], counts_by_fraction[i + 1])
        if len(set(counts_by_fraction)) != len(counts_by_fraction) or counts_by_fraction[0] < 1:
            raise SystemExit(f"Bad nested counts: {counts_by_fraction}")
        selected = {
            fraction: set(ordered[:count])
            for fraction, count in zip(FRACTIONS, counts_by_fraction)
        }
        for left, right in zip(FRACTIONS, FRACTIONS[1:]):
            if not selected[left] < selected[right]:
                raise SystemExit(f"Fold {fold}: label subsets not nested")
        rows = [
            {"subject_id": sid, "labeled_fraction": fraction}
            for fraction in FRACTIONS
            for sid in sorted(selected[fraction])
        ]
        pd.DataFrame(rows).to_csv(out / f"labeled_fold{fold}.csv", index=False)
        print(f"fold{fold}: train={n} labeled={ {f: len(s) for f, s in selected.items()} }")

    metadata = {
        "seed": args.seed,
        "n_folds": 5,
        "split_unit": "subject",
        "label_fractions": list(FRACTIONS),
        "source": "AMOS22 prepared manifest",
    }
    (out / "split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    split_files = sorted(path for path in out.iterdir() if path.is_file() and path.name != "split_files_sha256.csv")
    pd.DataFrame(
        [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in split_files]
    ).to_csv(out / "split_files_sha256.csv", index=False)
    print(f"amos splits written to {out}")


if __name__ == "__main__":
    main()

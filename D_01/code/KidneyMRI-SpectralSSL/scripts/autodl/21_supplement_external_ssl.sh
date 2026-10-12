#!/usr/bin/env bash
# 21_supplement_external_ssl.sh
# 补充实验 P1-4：把已训练好的近两年 SSL 基线（FRCNet/CMMT-Net 等）在 CHAOS/AMOS 上做
# source-only 外部推理（零训练成本）。输出目录与正式 external 同构但独立，不触碰正式结果。
#
# 用法：
#   bash scripts/autodl/21_supplement_external_ssl.sh
#   SUPP_METHODS="frcnet_5slice cmmt_5slice tcsm_5slice unetpp_5slice" bash scripts/autodl/21_supplement_external_ssl.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_2_csd_comparison_autodl_v1"
SUPP_ROOT="$WORKSPACE/experiments/supplement/external_supplement"
LOG_ROOT="$WORKSPACE/logs/supplement_external_ssl"
SUPP_METHODS="${SUPP_METHODS:-frcnet_5slice cmmt_5slice}"
DATASETS="${SUPP_DATASETS:-chaos_t2_spir amos22_mri}"
SEEDS="${SUPP_SEEDS:-20260811 20260812 20260813}"

mkdir -p "$LOG_ROOT" "$SUPP_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python - "$ROOT" "$SUPP_ROOT" "$DATASETS" "$SEEDS" $SUPP_METHODS <<'PY'
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kidney_ssl.benchmark_v2 import _aggregate_external_seeds
from kidney_ssl.external_evaluation import evaluate_external

internal_root, supp_root = Path(sys.argv[1]), Path(sys.argv[2])
datasets = sys.argv[3].split()
seeds = [int(v) for v in sys.argv[4].split()]
methods = sys.argv[5:]
runs = []
for dataset in datasets:
    for method in methods:
        for seed in seeds:
            output = supp_root / dataset / method / f"seed{seed}"
            if (output / "evaluation_summary.json").is_file():
                status = "skipped_completed"
            else:
                try:
                    evaluate_external(internal_root / "internal" / method, dataset, seed, output)
                    status = "completed"
                except Exception as exc:  # noqa: BLE001
                    status = "failed"
                    print(f"FAILED {dataset} {method} seed{seed}: {exc!r}", file=sys.stderr)
            runs.append({"dataset": dataset, "method": method, "seed": seed, "status": status})
        _aggregate_external_seeds(supp_root / dataset / method, supp_root / dataset / method)
with (supp_root / "external_supplement_runs.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["dataset", "method", "seed", "status"])
    writer.writeheader()
    writer.writerows(runs)

summary_rows = []
for dataset in datasets:
    for method in methods:
        table = supp_root / dataset / method / "metrics_subject.csv"
        if table.is_file():
            data = pd.read_csv(table)
            dice = float(np.mean(data["dice"]))
            hd95 = float(np.mean(data["hd95_mm"])) if "hd95_mm" in data else None
            tkv_err = float(np.mean(data["tkv_relative_error"])) if "tkv_relative_error" in data else None
            summary_rows.append({"dataset": dataset, "method": method, "dice": round(dice, 4),
                                 "hd95_mm": round(hd95, 4) if hd95 else None,
                                 "tkv_relative_error": round(tkv_err, 4) if tkv_err else None})
            print(f"{dataset} {method}: dice={dice:.4f} hd95={hd95} tkv_err={tkv_err}")
with (supp_root / "external_supplement_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["dataset", "method", "dice", "hd95_mm", "tkv_relative_error"])
    writer.writeheader()
    writer.writerows(summary_rows)
PY
echo "external-ssl supplement finished; outputs under $SUPP_ROOT"

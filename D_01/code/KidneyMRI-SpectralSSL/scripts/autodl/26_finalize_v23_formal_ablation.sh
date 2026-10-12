#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_3_formal_ablation_autodl_v1"
INTERNAL="$ROOT/internal"
PRIOR="$WORKSPACE/backups/comparison_v22_csd_internal_compact_20260822T221119Z.tar.gz"
EXPECTED_PRIOR_SHA="369f6788b5493c5adb4dd76cc8c657e683b65144ac3cd939f7ba9832fa2fec96"
ANALYSIS="$ROOT/analysis"
LOG_ROOT="$WORKSPACE/logs/v23_formal_ablation_finalize"
METHODS=(V23_BinarySup_5slice V23_MeanTeacher_5slice V23_M1_CSD_Global_5slice)

mkdir -p "$LOG_ROOT" "$ANALYSIS"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

[[ -f "$PRIOR" ]] || { echo "Locked v2.2 archive is missing" >&2; exit 2; }
actual_sha="$(sha256sum "$PRIOR" | awk '{print $1}')"
[[ "$actual_sha" == "$EXPECTED_PRIOR_SHA" ]] || { echo "Locked v2.2 archive hash mismatch" >&2; exit 2; }

python - "$INTERNAL" "${METHODS[@]}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
for method in sys.argv[2:]:
    method_root = root / method
    runs = sorted(method_root.glob("fold*_seed*"))
    summaries = list(method_root.glob("fold*_seed*/evaluation_summary.json"))
    statuses = list(method_root.glob("fold*_seed*/benchmark_status.json"))
    acquisitions = list(method_root.glob("fold*_seed*/metrics_acquisition.csv"))
    if not (len(runs) == len(summaries) == len(statuses) == len(acquisitions) == 15):
        raise SystemExit(
            f"{method}: runs={len(runs)}, summaries={len(summaries)}, "
            f"statuses={len(statuses)}, acquisition_tables={len(acquisitions)}"
        )
    for path in statuses:
        if json.loads(path.read_text(encoding="utf-8")).get("status") != "completed":
            raise SystemExit(f"Incomplete status: {path}")
    tables = []
    for path in acquisitions:
        table = pd.read_csv(path)
        seed = int(path.parent.name.split("_seed")[-1])
        table["seed"] = seed
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    if combined[["dice", "hd95_mm", "pred_tkv_ml", "true_tkv_ml"]].isna().any().any():
        raise SystemExit(f"{method}: NaN in required metrics")
    coverage = combined.groupby("seed").subject_id.nunique().to_dict()
    if set(coverage.values()) != {60}:
        raise SystemExit(f"{method}: incomplete 60-subject coverage: {coverage}")
    print(method, "runs=15 subjects_per_seed=60 metrics=finite")
PY

python scripts/analysis/analyze_v23_ablation_tkv.py \
    --prior-archive "$PRIOR" \
    --ablation-root "$INTERNAL" \
    --out-dir "$ANALYSIS" \
    --bootstrap-draws 10000 \
    --permutation-draws 100000 \
    --seed 20260803 \
    2>&1 | tee "$LOG_ROOT/analysis_console.log"

echo "v2.3 formal ablation and TKV analysis completed without modifying the v2.2 gate."


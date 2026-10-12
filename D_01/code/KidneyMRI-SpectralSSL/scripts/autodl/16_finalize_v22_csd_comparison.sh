#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SUITE="configs/autodl/q2_v22_csd_comparison.yaml"
ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_2_csd_comparison_autodl_v1"
INTERNAL="$ROOT/internal"
LOG_ROOT="$WORKSPACE/logs/comparison_v22_csd_finalize"
METHODS=(V22_B0plus_5slice KidneyBand_ReliableSSL_v2_2_CSD daniel_unet damiano_mcd_plus nnunet_3d_default unetpp_5slice tcsm_5slice cmmt_5slice frcnet_5slice semi_urf_5slice randdg_5slice)
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python - "$INTERNAL" "${METHODS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for method in sys.argv[2:]:
    method_root = root / method
    summaries = list(method_root.glob("fold*_seed*/evaluation_summary.json"))
    statuses = list(method_root.glob("fold*_seed*/benchmark_status.json"))
    if len(summaries) != 15 or len(statuses) != 15:
        raise SystemExit(
            f"{method}: summaries={len(summaries)}, statuses={len(statuses)}; expected 15 each"
        )
    bad = [
        str(path)
        for path in statuses
        if json.loads(path.read_text(encoding="utf-8")).get("status") != "completed"
    ]
    if bad:
        raise SystemExit(f"{method}: incomplete statuses={bad}")
    print(method, "complete")
PY

kidney-seg benchmark --suite "$SUITE" --phase internal --resume \
    2>&1 | tee "$LOG_ROOT/aggregate_console.log"

kidney-seg compare \
    --benchmark-dir "$INTERNAL" \
    --suite "$SUITE" \
    --draws 10000 \
    --seed 20260803 \
    2>&1 | tee "$LOG_ROOT/compare_console.log"

echo "CSD-only v2.2 internal comparison finalized. External and label-efficiency gates remain pending."


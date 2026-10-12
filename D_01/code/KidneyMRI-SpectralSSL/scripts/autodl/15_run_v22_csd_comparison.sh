#!/usr/bin/env bash
set -euo pipefail

[[ "$#" -eq 1 ]] || {
    echo "Usage: bash $0 primary|ssl|nnunet" >&2
    exit 2
}

SHARD="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SUITE="configs/autodl/q2_v22_csd_comparison.yaml"
ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_2_csd_comparison_autodl_v1"
LOG_ROOT="$WORKSPACE/logs/comparison_v22_csd_${SHARD}"

case "$SHARD" in
    primary)
        METHODS=(V22_B0plus_5slice KidneyBand_ReliableSSL_v2_2_CSD daniel_unet damiano_mcd_plus unetpp_5slice)
        ;;
    ssl)
        METHODS=(tcsm_5slice cmmt_5slice frcnet_5slice semi_urf_5slice randdg_5slice)
        ;;
    nnunet)
        METHODS=(nnunet_3d_default)
        ;;
    *)
        echo "Unknown shard: $SHARD; use primary, ssl, or nnunet" >&2
        exit 2
        ;;
esac

mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

[[ -f "$ROOT/protocol_lock_v2.json" ]] || {
    echo "Formal v2.2 protocol lock is missing; run 14_prepare_v22_csd_comparison.sh first" >&2
    exit 2
}
for method in "${METHODS[@]}"; do
    [[ ! -e "$ROOT/internal/$method" ]] || {
        echo "Refusing to overwrite existing output: $ROOT/internal/$method" >&2
        echo "After auditing an interrupted run, use 17_resume_v22_csd_comparison.sh" >&2
        exit 2
    }
done

kidney-seg benchmark --suite "$SUITE" --phase internal --methods "${METHODS[@]}" \
    2>&1 | tee "$LOG_ROOT/benchmark_console.log"

python - "$ROOT/internal" "${METHODS[@]}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
for method in sys.argv[2:]:
    count = len(list((root / method).glob("fold*_seed*/evaluation_summary.json")))
    if count != 15:
        raise SystemExit(f"{method}: expected 15 evaluations, found {count}")
    print(method, "15/15")
PY


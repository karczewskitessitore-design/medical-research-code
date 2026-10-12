#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SUITE="configs/autodl/q2_v21_comparison.yaml"
ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_1_comparison_autodl_v1"
LOG_ROOT="$WORKSPACE/logs/comparison_v21_primary"
METHODS=(V21_B0plus_5slice KidneyBand_ReliableSSL_v2_1 daniel_unet damiano_mcd_plus unetpp_5slice)
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

[[ -f "$ROOT/protocol_lock_v2.json" ]] || { echo "Formal protocol lock is missing" >&2; exit 2; }
for method in "${METHODS[@]}"; do
    [[ ! -e "$ROOT/internal/$method" ]] || {
        echo "Refusing to overwrite existing method output: $ROOT/internal/$method" >&2
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

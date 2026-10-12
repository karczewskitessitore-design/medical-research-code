#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SUITE="configs/autodl/q2_v21_comparison.yaml"
ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_1_comparison_autodl_v1"
LOG_ROOT="$WORKSPACE/logs/comparison_v21_nnunet"
METHOD="nnunet_3d_default"
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

[[ -f "$ROOT/protocol_lock_v2.json" ]] || { echo "Formal protocol lock is missing" >&2; exit 2; }
[[ ! -e "$ROOT/internal/$METHOD" ]] || {
    echo "Refusing to overwrite existing nnU-Net output: $ROOT/internal/$METHOD" >&2
    exit 2
}

kidney-seg benchmark --suite "$SUITE" --phase internal --methods "$METHOD" \
    2>&1 | tee "$LOG_ROOT/benchmark_console.log"

count="$(find "$ROOT/internal/$METHOD" -name evaluation_summary.json | wc -l)"
[[ "$count" -eq 15 ]] || { echo "nnU-Net expected 15 evaluations, found $count" >&2; exit 2; }
echo "nnunet_3d_default 15/15"

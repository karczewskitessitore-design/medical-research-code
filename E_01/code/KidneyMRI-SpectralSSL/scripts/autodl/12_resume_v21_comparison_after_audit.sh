#!/usr/bin/env bash
set -euo pipefail

[[ "${CONFIRM_AUDITED_RESUME:-}" == "YES" ]] || {
    echo "Set CONFIRM_AUDITED_RESUME=YES only after reviewing the failed run logs." >&2
    exit 2
}
[[ "$#" -ge 1 ]] || {
    echo "Usage: CONFIRM_AUDITED_RESUME=YES bash $0 <method_id> [method_id ...]" >&2
    exit 2
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SUITE="configs/autodl/q2_v21_comparison.yaml"
LOG_ROOT="$WORKSPACE/logs/comparison_v21_resume"
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
kidney-seg benchmark --suite "$SUITE" --phase internal --resume --methods "$@" \
    2>&1 | tee "$LOG_ROOT/resume_${stamp}.log"

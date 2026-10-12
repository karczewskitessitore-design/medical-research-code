#!/usr/bin/env bash
set -euo pipefail

[[ "${CONFIRM_AUDITED_RESUME:-}" == "YES" ]] || {
    echo "Set CONFIRM_AUDITED_RESUME=YES only after auditing the failed run and confirming unchanged configuration." >&2
    exit 2
}
[[ "$#" -ge 1 ]] || {
    echo "Usage: CONFIRM_AUDITED_RESUME=YES bash $0 METHOD [METHOD ...]" >&2
    exit 2
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SUITE="configs/autodl/q2_v23_formal_ablation.yaml"
LOG_ROOT="$WORKSPACE/logs/v23_formal_ablation_manual_resume"
ALLOWED=" V23_BinarySup_5slice V23_MeanTeacher_5slice V23_M1_CSD_Global_5slice "
for method in "$@"; do
    [[ "$ALLOWED" == *" $method "* ]] || { echo "Unknown v2.3 method: $method" >&2; exit 2; }
done

mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

kidney-seg benchmark --suite "$SUITE" --phase internal --resume --methods "$@" \
    2>&1 | tee "$LOG_ROOT/resume_console.log"


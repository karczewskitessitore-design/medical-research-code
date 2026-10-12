#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SUITE="configs/autodl/q2_v23_formal_ablation.yaml"
ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_3_formal_ablation_autodl_v1"
LOG_ROOT="$WORKSPACE/logs/v23_formal_ablation_run"
METHODS=(V23_BinarySup_5slice V23_MeanTeacher_5slice V23_M1_CSD_Global_5slice)

mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

[[ -f "$ROOT/protocol_lock_v2.json" ]] || {
    echo "Formal v2.3 protocol lock is missing; run 24_prepare_v23_formal_ablation.sh" >&2
    exit 2
}
for method in "${METHODS[@]}"; do
    [[ ! -e "$ROOT/internal/$method" ]] || {
        echo "Refusing to overwrite existing output: $ROOT/internal/$method" >&2
        echo "Audit the interruption before using 27_resume_v23_formal_ablation.sh" >&2
        exit 2
    }
done

kidney-seg benchmark --suite "$SUITE" --phase internal --methods "${METHODS[@]}" \
    2>&1 | tee "$LOG_ROOT/benchmark_console.log"

python - "$ROOT/internal" "${METHODS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for method in sys.argv[2:]:
    method_root = root / method
    summaries = list(method_root.glob("fold*_seed*/evaluation_summary.json"))
    statuses = list(method_root.glob("fold*_seed*/benchmark_status.json"))
    if len(summaries) != 15 or len(statuses) != 15:
        raise SystemExit(f"{method}: summaries={len(summaries)}, statuses={len(statuses)}; expected 15")
    bad = [
        str(path) for path in statuses
        if json.loads(path.read_text(encoding="utf-8")).get("status") != "completed"
    ]
    if bad:
        raise SystemExit(f"{method}: incomplete statuses={bad}")
    print(method, "15/15 complete")
PY

echo "All 45 new v2.3 runs completed. Run 26_finalize_v23_formal_ablation.sh next."


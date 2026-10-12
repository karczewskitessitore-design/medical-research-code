#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SCREEN="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_1_5slice_screen_autodl_fresh_v1/screen"
SMOKE_SUITE="configs/autodl/q2_v21_comparison_smoke.yaml"
FORMAL_SUITE="configs/autodl/q2_v21_comparison.yaml"
SMOKE_ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_1_comparison_smoke_autodl_v1"
FORMAL_ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_1_comparison_autodl_v1"
LOG_ROOT="$WORKSPACE/logs/comparison_v21_prepare"
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python - "$SCREEN" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
decision_path = root / "screening_decision.json"
if not decision_path.is_file():
    raise SystemExit(f"Missing v2.1 screening decision: {decision_path}")
decision = json.loads(decision_path.read_text(encoding="utf-8"))
summaries = list(root.glob("*/fold*_seed*/evaluation_summary.json"))
if len(summaries) != 8:
    raise SystemExit(f"Expected 8 v2.1 screening summaries, found {len(summaries)}")
if decision.get("status") != "complete" or decision.get("formal_method_ready") is not True:
    raise SystemExit(
        "The v2.1 proposed method did not pass the development gate. "
        "Do not run the formal comparison suite."
    )
print(json.dumps(decision, indent=2))
PY

if [[ -e "$SMOKE_ROOT/internal" ]]; then
    echo "Refusing to overwrite an existing comparison smoke run: $SMOKE_ROOT/internal" >&2
    exit 2
fi
if [[ -e "$FORMAL_ROOT/internal" ]]; then
    echo "Formal comparison output already exists: $FORMAL_ROOT/internal" >&2
    exit 2
fi

pytest -q \
    tests/test_benchmark_models.py \
    tests/test_nnunet_adapter.py \
    tests/test_comparison_suite_v21.py \
    2>&1 | tee "$LOG_ROOT/pytest.log"

for command in nnUNetv2_plan_and_preprocess nnUNetv2_predict; do
    command -v "$command" >/dev/null || {
        echo "Missing locked nnU-Net command: $command" >&2
        exit 2
    }
done

kidney-seg lock-protocol --suite "$SMOKE_SUITE" \
    2>&1 | tee "$LOG_ROOT/smoke_lock.log"
kidney-seg benchmark --suite "$SMOKE_SUITE" --phase internal \
    2>&1 | tee "$LOG_ROOT/smoke_console.log"

python - "$SMOKE_ROOT/internal" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = list(root.glob("*/fold*_seed*/evaluation_summary.json"))
statuses = list(root.glob("*/fold*_seed*/benchmark_status.json"))
if len(summaries) != 10 or len(statuses) != 10:
    raise SystemExit(f"Comparison smoke incomplete: summaries={len(summaries)}, statuses={len(statuses)}")
bad = [
    str(path)
    for path in statuses
    if json.loads(path.read_text(encoding="utf-8")).get("status") != "completed"
]
if bad:
    raise SystemExit(f"Comparison smoke failures: {bad}")
print("comparison_smoke=passed methods=10")
PY

kidney-seg lock-protocol --suite "$FORMAL_SUITE" \
    2>&1 | tee "$LOG_ROOT/formal_lock.log"

echo "Comparison smoke passed and the formal protocol is locked. No formal training was started."

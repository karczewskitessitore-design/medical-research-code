#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SCREEN="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_1_5slice_screen_autodl_fresh_v1/screen"
SMOKE_SUITE="configs/autodl/q2_v22_csd_comparison_smoke.yaml"
FORMAL_SUITE="configs/autodl/q2_v22_csd_comparison.yaml"
SMOKE_ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_2_csd_comparison_smoke_autodl_v1"
FORMAL_ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_2_csd_comparison_autodl_v1"
LOG_ROOT="$WORKSPACE/logs/comparison_v22_csd_prepare"
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
summary_path = root / "screening_summary.csv"
if not decision_path.is_file() or not summary_path.is_file():
    raise SystemExit("Missing the completed v2.1 screening decision or summary")
decision = json.loads(decision_path.read_text(encoding="utf-8"))
summaries = list(root.glob("*/fold*_seed*/evaluation_summary.json"))
if decision.get("status") != "complete" or len(summaries) != 8:
    raise SystemExit(
        f"The v2.1 screen is incomplete: status={decision.get('status')}, summaries={len(summaries)}"
    )
modules = decision.get("module_decisions", {})
if modules.get("CSD", {}).get("keep") is not True:
    raise SystemExit("CSD did not pass its preregistered development gate")
if modules.get("UABand", {}).get("keep") is not False:
    raise SystemExit("Expected UABand to be rejected by the recorded development gate")
if decision.get("formal_method_ready") is not False:
    raise SystemExit("Expected the v2.1 full method to be rejected before selecting CSD-only v2.2")
print("development_gate=verified selected=CSD-only rejected=UABand full_v2.1")
print(json.dumps(decision, indent=2))
PY

if [[ -e "$SMOKE_ROOT/internal" ]]; then
    echo "Refusing to overwrite an existing v2.2 smoke run: $SMOKE_ROOT/internal" >&2
    exit 2
fi
if [[ -e "$FORMAL_ROOT/internal" ]]; then
    echo "Formal v2.2 comparison output already exists: $FORMAL_ROOT/internal" >&2
    exit 2
fi

pytest -q \
    tests/test_benchmark_models.py \
    tests/test_nnunet_adapter.py \
    tests/test_comparison_suite_v22.py \
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
    raise SystemExit(f"v2.2 smoke incomplete: summaries={len(summaries)}, statuses={len(statuses)}")
bad = [
    str(path)
    for path in statuses
    if json.loads(path.read_text(encoding="utf-8")).get("status") != "completed"
]
if bad:
    raise SystemExit(f"v2.2 smoke failures: {bad}")
print("v22_csd_comparison_smoke=passed methods=10")
PY

kidney-seg lock-protocol --suite "$FORMAL_SUITE" \
    2>&1 | tee "$LOG_ROOT/formal_lock.log"

echo "CSD-only v2.2 smoke passed and the formal protocol is locked. No formal training was started."


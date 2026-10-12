#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SUITE="configs/autodl/q2_v2_1_5slice_screen.yaml"
SUITE_ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_1_5slice_screen_autodl_fresh_v1"
SCREEN_ROOT="$SUITE_ROOT/screen"
LOG_ROOT="$WORKSPACE/logs/screen_v21_5slice"
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$WORKSPACE/logs/preflight/preflight_passed.json" ]] || {
    echo "GPU preflight gate is missing. Run 02_preflight_gpu.sh first." >&2
    exit 2
}
if [[ -e "$SCREEN_ROOT" ]]; then
    echo "Refusing to reuse or overwrite an existing v2.1 screen: $SCREEN_ROOT" >&2
    exit 2
fi

{
    date --iso-8601=seconds
    python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
    python - <<'PY'
from kidney_ssl.config import load_config

for name in (
    "configs/v2/B0plus_5slice.yaml",
    "configs/v2/UABand_5slice.yaml",
    "configs/v2/CSD_5slice.yaml",
    "configs/v2/UABand_CSD_5slice.yaml",
):
    config = load_config(name)
    assert config["model"]["in_channels"] == 5, name
    assert config["repeat_morphometry"]["enabled"] is False, name
    print(name, "validated")
PY
} > "$LOG_ROOT/run_identity.txt"

# Eight fresh runs: four matched five-slice methods x two development folds.
kidney-seg benchmark --suite "$SUITE" --phase screen \
    2>&1 | tee "$LOG_ROOT/benchmark_console.log"

python - "$SCREEN_ROOT/screening_decision.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
decision = json.loads(path.read_text(encoding="utf-8"))
if decision.get("status") != "complete":
    raise SystemExit(f"v2.1 screening did not complete: {decision}")
print(json.dumps(decision, indent=2))
PY

echo "v2.1 matched five-slice screening completed. Review the decision before any protocol lock."

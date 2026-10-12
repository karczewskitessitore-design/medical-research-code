#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
DATA_ROOT="$WORKSPACE/data/T2KidneyMRI"
LOG_ROOT="$WORKSPACE/logs/preflight"
SMOKE_ROOT="$WORKSPACE/experiments/engineering_fresh/V2_DZ_GPU_BF16_autodl_fresh_smoke/fold0_seed20260803"
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

[[ -f "$DATA_ROOT/audit/autodl_data_ready.json" ]] || {
    echo "Fresh data gate is missing. Run 01_prepare_data_fresh.sh first." >&2
    exit 2
}
if [[ -e "$SMOKE_ROOT" ]]; then
    echo "Refusing to overwrite an existing smoke run: $SMOKE_ROOT" >&2
    exit 2
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    | tee "$LOG_ROOT/nvidia_smi.txt"
pytest -q 2>&1 | tee "$LOG_ROOT/pytest.log"

kidney-seg train \
    --config configs/autodl/engineering_dz_gpu_smoke.yaml \
    --fold 0 --seed 20260803 \
    2>&1 | tee "$LOG_ROOT/gpu_smoke_train.log"
kidney-seg evaluate --run-dir "$SMOKE_ROOT" \
    2>&1 | tee "$LOG_ROOT/gpu_smoke_evaluate.log"

python - "$SMOKE_ROOT" "$LOG_ROOT/preflight_passed.json" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
destination = Path(sys.argv[2])
rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
result = json.loads((run / "training_result.json").read_text(encoding="utf-8"))
if not rows or rows[-1].get("repeat_update_applied") is not True:
    raise SystemExit("DZ-BiMorph repeat update was not exercised")
if rows[-1].get("precision") != "bf16" or rows[-1].get("amp_enabled") is not True:
    raise SystemExit(f"Unexpected precision state: {rows[-1]}")
if result.get("resumed") is not False or result.get("device") != "cuda":
    raise SystemExit(f"Smoke run was not a fresh CUDA run: {result}")
if not (run / "evaluation_summary.json").is_file():
    raise SystemExit("Smoke evaluation output is missing")
payload = {
    "status": "passed",
    "run_dir": str(run),
    "repeat_update_applied": True,
    "precision": "bf16",
    "peak_gpu_memory_mb": result.get("peak_gpu_memory_mb"),
    "duration_seconds": result.get("duration_seconds"),
}
destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "GPU preflight passed. The full screen has not been started."

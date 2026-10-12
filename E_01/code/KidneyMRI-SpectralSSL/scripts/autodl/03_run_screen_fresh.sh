#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SUITE="configs/autodl/q2_v2_fresh.yaml"
SUITE_ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_autodl_fresh_v1"
SCREEN_ROOT="$SUITE_ROOT/screen"
LOG_ROOT="$WORKSPACE/logs/screen_fresh"
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

[[ -f "$WORKSPACE/logs/preflight/preflight_passed.json" ]] || {
    echo "GPU preflight gate is missing. Run 02_preflight_gpu.sh first." >&2
    exit 2
}
if [[ -e "$SCREEN_ROOT" ]]; then
    echo "Refusing to reuse or overwrite an existing screen directory: $SCREEN_ROOT" >&2
    echo "For a new experiment, choose a new KIDNEY_WORKSPACE or a new versioned suite name." >&2
    exit 2
fi

{
    date --iso-8601=seconds
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        git rev-parse HEAD
        git status --short
    elif [[ -f "$WORKSPACE/upload/bundle_manifest.txt" ]]; then
        echo "git_metadata_from_upload_bundle"
        cat "$WORKSPACE/upload/bundle_manifest.txt"
    else
        echo "git_metadata_unavailable"
    fi
    python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
} > "$LOG_ROOT/run_identity.txt"

# Deliberately omit --resume: every one of the 18 screening runs starts at epoch 0.
kidney-seg benchmark --suite "$SUITE" --phase screen \
    2>&1 | tee "$LOG_ROOT/benchmark_console.log"

python - "$SCREEN_ROOT/screening_decision.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
decision = json.loads(path.read_text(encoding="utf-8"))
if decision.get("status") != "complete":
    raise SystemExit(f"Screening did not complete: {decision}")
print(json.dumps(decision, indent=2))
PY

echo "All preregistered fresh screening runs completed. Review screening_decision.json before locking the protocol."

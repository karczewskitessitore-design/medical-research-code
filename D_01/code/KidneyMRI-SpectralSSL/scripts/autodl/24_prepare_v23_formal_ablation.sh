#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
PRIOR="$WORKSPACE/backups/comparison_v22_csd_internal_compact_20260822T221119Z.tar.gz"
EXPECTED_PRIOR_SHA="369f6788b5493c5adb4dd76cc8c657e683b65144ac3cd939f7ba9832fa2fec96"
SMOKE_SUITE="configs/autodl/q2_v23_formal_ablation_smoke.yaml"
FORMAL_SUITE="configs/autodl/q2_v23_formal_ablation.yaml"
SMOKE_ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_3_formal_ablation_smoke_autodl_v1"
FORMAL_ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_3_formal_ablation_autodl_v1"
LOG_ROOT="$WORKSPACE/logs/v23_formal_ablation_prepare"
METHODS=(V23_BinarySup_5slice V23_MeanTeacher_5slice V23_M1_CSD_Global_5slice)

mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python - <<'PY' | tee "$LOG_ROOT/gpu_preflight.log"
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; the formal ablation must not start")
name = torch.cuda.get_device_name(0)
if "4090" not in name:
    raise SystemExit(f"Expected an RTX 4090, found: {name}")
if not torch.cuda.is_bf16_supported():
    raise SystemExit(f"BF16 is unsupported on: {name}")
properties = torch.cuda.get_device_properties(0)
print(f"gpu={name}")
print(f"total_memory_gib={properties.total_memory / 1024**3:.2f}")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print("bf16_supported=True")
PY

[[ -f "$PRIOR" ]] || { echo "Locked v2.2 archive is missing: $PRIOR" >&2; exit 2; }
actual_sha="$(sha256sum "$PRIOR" | awk '{print $1}')"
[[ "$actual_sha" == "$EXPECTED_PRIOR_SHA" ]] || {
    echo "Locked v2.2 archive hash mismatch: $actual_sha" >&2
    exit 2
}
[[ ! -e "$SMOKE_ROOT/internal" ]] || {
    echo "Refusing to overwrite an existing v2.3 smoke run: $SMOKE_ROOT/internal" >&2
    exit 2
}
[[ ! -e "$FORMAL_ROOT/internal" ]] || {
    echo "Refusing to overwrite an existing v2.3 formal run: $FORMAL_ROOT/internal" >&2
    exit 2
}

pytest -q \
    tests/test_v2_losses.py \
    tests/test_evaluation_v2.py \
    tests/test_formal_ablation_v23.py \
    2>&1 | tee "$LOG_ROOT/pytest.log"

kidney-seg lock-protocol --suite "$SMOKE_SUITE" \
    2>&1 | tee "$LOG_ROOT/smoke_lock.log"
kidney-seg benchmark --suite "$SMOKE_SUITE" --phase internal --methods "${METHODS[@]}" \
    2>&1 | tee "$LOG_ROOT/smoke_console.log"

python - "$SMOKE_ROOT/internal" "${METHODS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for method in sys.argv[2:]:
    method_root = root / method
    summaries = list(method_root.glob("fold*_seed*/evaluation_summary.json"))
    statuses = list(method_root.glob("fold*_seed*/benchmark_status.json"))
    if len(summaries) != 1 or len(statuses) != 1:
        raise SystemExit(f"{method}: smoke summaries={len(summaries)}, statuses={len(statuses)}")
    status = json.loads(statuses[0].read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise SystemExit(f"{method}: smoke status={status.get('status')}")
    print(method, "smoke=passed")
PY

kidney-seg lock-protocol --suite "$FORMAL_SUITE" \
    2>&1 | tee "$LOG_ROOT/formal_lock.log"

echo "v2.3 smoke passed and the formal protocol is locked. No formal training was started."

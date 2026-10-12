#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
DATA_ROOT="$WORKSPACE/data/T2KidneyMRI"
LOG_ROOT="$WORKSPACE/logs/data_prepare"
mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate

for directory in raw audit splits; do
    candidate="$DATA_ROOT/$directory"
    if [[ -d "$candidate" ]] && [[ -n "$(find "$candidate" -mindepth 1 -print -quit)" ]]; then
        echo "Refusing to reuse non-empty data state: $candidate" >&2
        echo "Choose a new KIDNEY_WORKSPACE for a genuinely fresh run." >&2
        exit 2
    fi
done

declare -A expected_md5=(
    [CKD.zip]=8fe112954083212cbee80a6087db84c5
    [Healthy_Control.zip]=561badbc875874c5e69cf725a8545330
    [subject_information.csv]=3bc3d595339cb7c1f9208a07f7335ef5
)
for name in CKD.zip Healthy_Control.zip subject_information.csv; do
    file="$DATA_ROOT/downloads/$name"
    [[ -f "$file" ]] || { echo "Missing upload: $file" >&2; exit 2; }
    actual="$(md5sum "$file" | awk '{print $1}')"
    [[ "$actual" == "${expected_md5[$name]}" ]] || {
        echo "MD5 mismatch for $file: $actual" >&2
        exit 2
    }
done

kidney-seg download --data-root "$DATA_ROOT" 2>&1 | tee "$LOG_ROOT/download_verify_extract.log"
kidney-seg audit --data-root "$DATA_ROOT" 2>&1 | tee "$LOG_ROOT/audit.log"
kidney-seg make-folds --manifest "$DATA_ROOT/audit/manifest.csv" --seed 20260802 \
    2>&1 | tee "$LOG_ROOT/folds_20260802.log"
kidney-seg make-folds --manifest "$DATA_ROOT/audit/manifest.csv" --seed 20260803 \
    2>&1 | tee "$LOG_ROOT/folds_20260803.log"
kidney-seg audit-bilateral \
    --manifest "$DATA_ROOT/audit/manifest.csv" \
    --output-dir "$DATA_ROOT/audit/bilateral" \
    2>&1 | tee "$LOG_ROOT/bilateral.log"

python - "$DATA_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
summary = pd.read_csv(root / "audit" / "audit_summary.csv").iloc[0].to_dict()
expected = {
    "subjects": 60,
    "acquisitions": 100,
    "hc_subjects": 30,
    "ckd_subjects": 30,
    "hc_repeat_subjects": 5,
    "ckd_repeat_subjects": 5,
    "anomalies": 0,
}
actual = {key: int(summary[key]) for key in expected}
if actual != expected:
    raise SystemExit(f"Dataset audit mismatch: {actual}")
for seed in (20260802, 20260803):
    split = root / "splits" / f"seed_{seed}"
    required = [split / "subject_folds.csv", split / "manifest_with_folds.csv"]
    required.extend(split / f"labeled_fold{fold}.csv" for fold in range(5))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing split artifacts for {seed}: {missing}")
(root / "audit" / "autodl_data_ready.json").write_text(
    json.dumps({"status": "ready", **actual}, indent=2), encoding="utf-8"
)
print(json.dumps(actual, indent=2))
PY

echo "Fresh data preparation completed: $DATA_ROOT"

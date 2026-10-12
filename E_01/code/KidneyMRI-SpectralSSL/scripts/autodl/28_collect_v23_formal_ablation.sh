#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_3_formal_ablation_autodl_v1"
INTERNAL="$ROOT/internal"
BACKUP_ROOT="$WORKSPACE/backups"
mkdir -p "$BACKUP_ROOT"

[[ -f "$ROOT/analysis/analysis_status.json" ]] || {
    echo "Formal analysis is missing; run 26_finalize_v23_formal_ablation.sh first" >&2
    exit 2
}
python - "$ROOT/analysis/analysis_status.json" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if status.get("formal_ablation", {}).get("status") != "complete":
    raise SystemExit("Formal ablation analysis is not complete")
if status.get("locked_gate_unchanged") != "Q2_GATE_NOT_MET":
    raise SystemExit("Locked v2.2 gate provenance is missing")
PY

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="$BACKUP_ROOT/formal_ablation_v23_compact_${stamp}.tar.gz"
tar --exclude='*.pt' --exclude='predictions' -czf "$archive" -C "$ROOT" \
    protocol_lock_v2.json internal analysis
sha256sum "$archive" > "$archive.sha256"
echo "$archive"
echo "$archive.sha256"

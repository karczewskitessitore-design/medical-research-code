#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_1_comparison_autodl_v1"
INTERNAL="$ROOT/internal"
BACKUP_ROOT="$WORKSPACE/backups"
mkdir -p "$BACKUP_ROOT"

[[ -f "$INTERNAL/comparison_summary.json" ]] || {
    echo "Comparison summary is missing; run 11_finalize_v21_comparison.sh first." >&2
    exit 2
}

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="$BACKUP_ROOT/comparison_v21_internal_compact_${stamp}.tar.gz"
tar --exclude='*.pt' --exclude='nnunet_workspace' -czf "$archive" -C "$ROOT" \
    protocol_lock_v2.json internal
sha256sum "$archive" > "$archive.sha256"
echo "$archive"
echo "$archive.sha256"

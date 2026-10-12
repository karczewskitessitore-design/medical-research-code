#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SUITE_ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_1_5slice_screen_autodl_fresh_v1"
SCREEN_ROOT="$SUITE_ROOT/screen"
BACKUP_ROOT="$WORKSPACE/backups"
mkdir -p "$BACKUP_ROOT"

[[ -f "$SCREEN_ROOT/screening_decision.json" ]] || {
    echo "v2.1 screening decision is missing: $SCREEN_ROOT" >&2
    exit 2
}

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="$BACKUP_ROOT/screen_v21_5slice_compact_${stamp}.tar.gz"
tar --exclude='*.pt' -czf "$archive" -C "$SUITE_ROOT" screen
sha256sum "$archive" > "$archive.sha256"
echo "$archive"
echo "$archive.sha256"

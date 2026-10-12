#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SUITE_ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_autodl_fresh_v1"
SCREEN_ROOT="$SUITE_ROOT/screen"
BACKUP_ROOT="$WORKSPACE/backups"
mkdir -p "$BACKUP_ROOT"

[[ -f "$SCREEN_ROOT/screening_decision.json" ]] || {
    echo "Screening decision is missing: $SCREEN_ROOT" >&2
    exit 2
}

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="$BACKUP_ROOT/screen_compact_${stamp}.tar.gz"

# Compact backup intentionally excludes large model checkpoints. Keep the
# original experiment directory until a separate checkpoint backup is verified.
tar --exclude='*.pt' -czf "$archive" -C "$SUITE_ROOT" screen
sha256sum "$archive" > "$archive.sha256"
echo "$archive"
echo "$archive.sha256"

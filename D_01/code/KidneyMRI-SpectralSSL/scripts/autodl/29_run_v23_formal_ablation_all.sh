#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
MASTER_LOG="$WORKSPACE/logs/v23_formal_ablation_master.log"
mkdir -p "$(dirname "$MASTER_LOG")"
cd "$PROJECT_ROOT"

{
    echo "v2.3 formal ablation started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    bash scripts/autodl/24_prepare_v23_formal_ablation.sh
    bash scripts/autodl/25_run_v23_formal_ablation.sh
    bash scripts/autodl/26_finalize_v23_formal_ablation.sh
    bash scripts/autodl/28_collect_v23_formal_ablation.sh
    echo "v2.3 formal ablation completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$MASTER_LOG"

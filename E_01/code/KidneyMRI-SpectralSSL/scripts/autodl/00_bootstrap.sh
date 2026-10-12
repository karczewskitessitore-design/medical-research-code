#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SETUP_LOG="$WORKSPACE/logs/setup"
mkdir -p "$SETUP_LOG"
cd "$PROJECT_ROOT"

python -m pip install --upgrade pip uv 2>&1 | tee "$SETUP_LOG/pip_uv.log"
uv python install 3.11 2>&1 | tee "$SETUP_LOG/uv_python.log"
uv venv --python 3.11 .venv 2>&1 | tee "$SETUP_LOG/uv_venv.log"
source .venv/bin/activate

# --frozen prevents the server from silently changing the committed dependency lock.
UV_LINK_MODE=copy uv sync --frozen --extra benchmark --group dev --no-editable \
    2>&1 | tee "$SETUP_LOG/uv_sync.log"

python - <<'PY' | tee "$SETUP_LOG/environment.txt"
import platform
import torch

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"bf16_supported={torch.cuda.is_bf16_supported()}")
PY

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; do not start the experiment")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The selected GPU does not support BF16; do not change precision silently")
PY

echo "Bootstrap completed. Activate with: source $PROJECT_ROOT/.venv/bin/activate"

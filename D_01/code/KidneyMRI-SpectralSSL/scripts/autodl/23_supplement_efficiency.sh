#!/usr/bin/env bash
# 23_supplement_efficiency.sh
# 补充实验 P2-7：效率表（参数量 + 单批推理延迟 + 单卷推理估计），
# 用于论文 Discussion 中与 nnU-Net 3D 的复杂度对比。
# 用法：bash scripts/autodl/23_supplement_efficiency.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
SUPP_ROOT="$WORKSPACE/experiments/supplement/efficiency"
LOG_ROOT="$WORKSPACE/logs/supplement_efficiency"
# id=config 对（全部 2.5D 方法）
METHODS="${METHODS:-V22_B0plus_5slice=configs/v2/B0plus_5slice.yaml KidneyBand_ReliableSSL_v2_2_CSD=configs/v2/CSD_v22_5slice.yaml daniel_unet=configs/v2/baselines/daniel_unet.yaml damiano_mcd_plus=configs/v2/baselines/damiano_mcd_plus.yaml unetpp_5slice=configs/v2/baselines/unetpp_5slice.yaml tcsm_5slice=configs/v2/baselines/tcsm_5slice.yaml cmmt_5slice=configs/v2/baselines/cmmt_5slice.yaml frcnet_5slice=configs/v2/baselines/frcnet_5slice.yaml semi_urf_5slice=configs/v2/baselines/semi_urf_5slice.yaml randdg_5slice=configs/v2/baselines/randdg_5slice.yaml}"

mkdir -p "$LOG_ROOT" "$SUPP_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python - "$SUPP_ROOT" $METHODS <<'PY'
import csv
import sys
import time
from pathlib import Path

import torch

from kidney_ssl.config import load_config
from kidney_ssl.model import build_model

out_root = Path(sys.argv[1])
specs = [(item.split("=", 1)[0], Path(item.split("=", 1)[1]).resolve()) for item in sys.argv[2:]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rows = []
with torch.no_grad():
    for method_id, config_path in specs:
        config = load_config(config_path)
        model = build_model(config["model"]).to(device).eval()
        params = sum(parameter.numel() for parameter in model.parameters())
        batch = torch.randn(8, int(config["model"]["in_channels"]), 256, 256, device=device)
        for _ in range(5):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) / 20 * 1000
        # 30 层卷 ≈ 30 次切片前向（2.5D 单卷全量推理的粗略估计）
        volume_estimate_s = latency_ms / 1000 * 30
        rows.append({"method": method_id, "params": int(params),
                     "batch8_latency_ms": round(latency_ms, 2),
                     "est_30slice_volume_s": round(volume_estimate_s, 2)})
        print(rows[-1])
with (out_root / "efficiency_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["method", "params", "batch8_latency_ms", "est_30slice_volume_s"])
    writer.writeheader()
    writer.writerows(rows)
PY
echo "efficiency supplement finished; see $SUPP_ROOT/efficiency_summary.csv"
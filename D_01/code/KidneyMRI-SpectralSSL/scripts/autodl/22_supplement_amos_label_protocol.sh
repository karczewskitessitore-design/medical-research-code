#!/usr/bin/env bash
# 22_supplement_amos_label_protocol.sh
# 补充实验 P0-2：公共基准 AMOS22 MRI 上的 10%/20% 标签半监督协议，
# 与近两年论文（BoCLIS/VerSemi/FRCNet 等）同表对比。
# 流程：生成 AMOS 5 折 splits → 1 epoch 冒烟校准 → 全量训练（3 方法 × 2 比例 × 5 折）→ 汇总。
#
# 用法：
#   bash scripts/autodl/22_supplement_amos_label_protocol.sh
#   AMOS_SEEDS="20260811 20260812 20260813" bash scripts/autodl/22_supplement_amos_label_protocol.sh
#   AMOS_METHODS="KidneyBand_ReliableSSL_v2_2_CSD=configs/v2/CSD_v22_5slice.yaml V22_B0plus_5slice=configs/v2/B0plus_5slice.yaml" bash ...
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
AMOS_DATA="$WORKSPACE/data/AMOS22/prepared"
AMOS_MANIFEST="$AMOS_DATA/manifest.csv"
AMOS_SPLIT="$WORKSPACE/data/AMOS22/splits/seed_20260821"
SUPP_ROOT="$WORKSPACE/experiments/supplement/amos_label_protocol"
LOG_ROOT="$WORKSPACE/logs/supplement_amos_label_protocol"
AMOS_FRACTIONS="${AMOS_FRACTIONS:-0.10 0.20}"
AMOS_SEEDS="${AMOS_SEEDS:-20260811}"
# id=config 对
AMOS_METHODS="${AMOS_METHODS:-KidneyBand_ReliableSSL_v2_2_CSD=configs/v2/CSD_v22_5slice.yaml V22_B0plus_5slice=configs/v2/B0plus_5slice.yaml frcnet_5slice=configs/v2/baselines/frcnet_5slice.yaml}"

mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

[[ -f "$AMOS_MANIFEST" ]] || { echo "AMOS prepared manifest missing; run: kidney-seg prepare-external --dataset amos --data-root $WORKSPACE/data/AMOS22" >&2; exit 2; }

# 1) 生成 AMOS splits（10/20/50/100% 嵌套 5 折）
python scripts/autodl/22_make_amos_splits.py \
    --manifest "$AMOS_MANIFEST" --out "$AMOS_SPLIT" --seed 20260821 \
    2>&1 | tee "$LOG_ROOT/make_splits.log"

# 2) 生成解析后配置（每 方法 × 比例）
python - "$SUPP_ROOT" "$AMOS_SPLIT" "$AMOS_FRACTIONS" $AMOS_METHODS <<'PY'
import sys
from pathlib import Path
from kidney_ssl.config import load_config, dump_config

out_root, split_dir = Path(sys.argv[1]), Path(sys.argv[2])
fractions = [float(v) for v in sys.argv[3].split()]
specs = [(item.split("=", 1)[0], Path(item.split("=", 1)[1]).resolve()) for item in sys.argv[4:]]
overrides = {
    "data": {
        "root": "/root/autodl-tmp/kidney/data/AMOS22/prepared",
        "manifest": "/root/autodl-tmp/kidney/data/AMOS22/prepared/manifest.csv",
        "folds": str(split_dir),
    },
    "external_validation": {
        "datasets": [],
        "manifests": {
            "chaos": "/root/autodl-tmp/kidney/data/CHAOS/prepared/manifest.csv",
            "amos": "/root/autodl-tmp/kidney/data/AMOS22/prepared/manifest.csv",
        },
    },
    "runtime": {"num_workers": 0, "resume": False, "phase": "internal"},
}
resolved = out_root / "resolved_configs"
resolved.mkdir(parents=True, exist_ok=True)
for fraction in fractions:
    token = str(fraction).replace(".", "p")
    for method_id, config_path in specs:
        config = load_config(config_path, overrides=overrides)
        config["split"]["labeled_fraction"] = fraction
        config["runtime"]["output_root"] = str(out_root / token)
        config["runtime"]["run_name"] = method_id
        target = resolved / f"{method_id}_{token}.yaml"
        dump_config(config, target)
        print(f"{method_id}|{token}|{target}")
PY

# 3) 冒烟：1 epoch、每 epoch 4 步，验证 AMOS 数据与管线兼容并校准单轮耗时
SMOKE_METHOD="${AMOS_METHODS%%=*}"
SMOKE_TOKEN="$(python -c "import sys; print(str(float(sys.argv[1])).replace('.', 'p'))" "${AMOS_FRACTIONS%% *}")"
SMOKE_CFG="$SUPP_ROOT/resolved_configs/${SMOKE_METHOD}_${SMOKE_TOKEN}.yaml"
SMOKE_DIR="$SUPP_ROOT/smoke"
python - "$SMOKE_CFG" "$SMOKE_DIR" <<'PY'
import sys
from pathlib import Path
from kidney_ssl.config import load_config, dump_config
config = load_config(sys.argv[1])
config["optimizer"]["epochs"] = 1
config["runtime"]["max_steps_per_epoch"] = 4
config["runtime"]["output_root"] = str(Path(sys.argv[2]))
dump_config(config, Path(sys.argv[2]) / "smoke.yaml")
PY
echo "smoke: 1 epoch on fold0/seed20260811 (AMOS $SMOKE_TOKEN labels)"
timeout 3600 python -m kidney_ssl.cli train --config "$SMOKE_DIR/smoke.yaml" --fold 0 --seed 20260811 \
    > "$LOG_ROOT/smoke_train.log" 2>&1 || { echo "SMOKE FAILED — AMOS 训练管线不兼容，请检查 $LOG_ROOT/smoke_train.log" >&2; tail -n 30 "$LOG_ROOT/smoke_train.log" >&2; exit 2; }
smoke_seconds=$(tail -n 1 "$SMOKE_DIR/$SMOKE_METHOD/fold0_seed20260811/metrics.jsonl" 2>/dev/null \
    | python -c "import json,sys; print(json.loads(sys.stdin.read())['elapsed_seconds'])" 2>/dev/null || echo "")
if [[ -n "$smoke_seconds" ]]; then
    per_epoch=$(python -c "print(round(float('$smoke_seconds')))")
    echo "smoke_ok 1_epoch_elapsed_s=$per_epoch; full-150-epoch estimate ≈ $((per_epoch * 150 / 60)) min/run"
fi

# 4) 全量训练 + 评估（已有结果自动跳过）
for fraction in $AMOS_FRACTIONS; do
  token="$(python -c "import sys; print(str(float(sys.argv[1])).replace('.', 'p'))" "$fraction")"
  for spec in $AMOS_METHODS; do
    method_id="${spec%%=*}"
    cfg="$SUPP_ROOT/resolved_configs/${method_id}_${token}.yaml"
    [[ -f "$cfg" ]] || { echo "Missing resolved config: $cfg" >&2; exit 2; }
    for seed in $AMOS_SEEDS; do
      for fold in 0 1 2 3 4; do
        run_dir="$SUPP_ROOT/$token/$method_id/fold${fold}_seed${seed}"
        [[ -f "$run_dir/evaluation_summary.json" ]] && { echo "skip completed: $run_dir"; continue; }
        mkdir -p "$run_dir"
        echo "train $method_id fraction=$fraction fold=$fold seed=$seed"
        timeout 28800 python -m kidney_ssl.cli train --config "$cfg" --fold "$fold" --seed "$seed" \
          > "$run_dir/train.log" 2>&1 || { echo "FAILED train: $run_dir" >&2; tail -n 20 "$run_dir/train.log" >&2; exit 2; }
        python -m kidney_ssl.cli evaluate --run-dir "$run_dir" \
          > "$run_dir/evaluate.log" 2>&1 || { echo "FAILED evaluate: $run_dir" >&2; tail -n 20 "$run_dir/evaluate.log" >&2; exit 2; }
        echo "  done: $run_dir"
      done
    done
  done
done

# 5) 汇总 AMOS 10%/20% Dice
python - "$SUPP_ROOT" "$AMOS_FRACTIONS" $AMOS_METHODS <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
fractions = [float(v) for v in sys.argv[2].split()]
rows = []
for fraction in fractions:
    token = str(fraction).replace(".", "p")
    for item in sys.argv[3:]:
        method_id = item.split("=", 1)[0]
        dice = []
        for path in (root / token / method_id).glob("fold*_seed*/evaluation_summary.json"):
            dice.append(float(json.loads(path.read_text(encoding="utf-8"))["overall"]["dice"]))
        rows.append({"dataset": "amos22_mri", "labeled_fraction": fraction, "method": method_id,
                     "mean_dice": round(float(sum(dice) / len(dice)), 6) if dice else None,
                     "n_runs": len(dice)})
        print(rows[-1])
with (root / "amos_label_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["dataset", "labeled_fraction", "method", "mean_dice", "n_runs"])
    writer.writeheader()
    writer.writerows(rows)
PY
echo "amos label protocol finished; outputs under $SUPP_ROOT"

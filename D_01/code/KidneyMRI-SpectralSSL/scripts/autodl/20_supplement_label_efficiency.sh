#!/usr/bin/env bash
# 20_supplement_label_efficiency.sh
# 补充实验 P0-1/P1-6：内部标签效率曲线（12.5%/50%/100%）+ 100% 标签 sanity。
# 使用正式 split（seed_20260803），直接调用 kidney-seg train/evaluate，不触碰已锁定的正式协议。
# 完成后把 100% 结果写入正式 benchmark 根的 label_efficiency/，并重新生成 q2 gate 汇总。
#
# 用法：
#   bash scripts/autodl/20_supplement_label_efficiency.sh            # 1 seed，全曲线
#   LABEL_SEEDS="20260811 20260812 20260813" bash scripts/autodl/20_supplement_label_efficiency.sh   # 100% 也跑 3 seeds
#   SKIP_GATE=1 bash scripts/autodl/20_supplement_label_efficiency.sh # 只跑实验，不重新生成 gate
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${KIDNEY_WORKSPACE:-/root/autodl-tmp/kidney}"
ROOT="$WORKSPACE/experiments/benchmarks/kidneyband_reliable_ssl_q2_v2_2_csd_comparison_autodl_v1"
SUPP_ROOT="$WORKSPACE/experiments/supplement"
LOG_ROOT="$WORKSPACE/logs/supplement_label_efficiency"
FORMAL_SUITE="configs/autodl/q2_v22_csd_comparison.yaml"
FORMAL_SPLIT="$WORKSPACE/data/T2KidneyMRI/splits/seed_20260803"
LABEL_SEEDS="${LABEL_SEEDS:-20260811}"
LABEL_FRACTIONS="${LABEL_FRACTIONS:-0.125 0.5 1.0}"
# id=config 对；修改 METHODS 可增减方法
METHODS="${METHODS:-KidneyBand_ReliableSSL_v2_2_CSD=configs/v2/CSD_v22_5slice.yaml V22_B0plus_5slice=configs/v2/B0plus_5slice.yaml}"

mkdir -p "$LOG_ROOT" "$SUPP_ROOT/label_efficiency"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# 1) 生成各 fraction 的解析后配置（与 benchmark 的 resolved_config 同构）
python - "$SUPP_ROOT/label_efficiency" "$FORMAL_SPLIT" "$LABEL_FRACTIONS" $METHODS <<'PY'
import sys
from pathlib import Path
from kidney_ssl.config import load_config, dump_config

out_root, split_dir = Path(sys.argv[1]), Path(sys.argv[2])
fractions = [float(v) for v in sys.argv[3].split()]
specs = []
for item in sys.argv[4:]:
    method_id, config_rel = item.split("=", 1)
    specs.append((method_id, Path(config_rel).resolve()))
overrides = {
    "data": {
        "root": "/root/autodl-tmp/kidney/data/T2KidneyMRI",
        "manifest": "/root/autodl-tmp/kidney/data/T2KidneyMRI/audit/manifest.csv",
        "folds": str(split_dir),
    },
    "external_validation": {
        "manifests": {
            "chaos": "/root/autodl-tmp/kidney/data/CHAOS/prepared/manifest.csv",
            "amos": "/root/autodl-tmp/kidney/data/AMOS22/prepared/manifest.csv",
        }
    },
    "runtime": {"num_workers": 2, "resume": False, "phase": "internal"},
}
token_map = {0.125: "12p5", 0.5: "50", 1.0: "100"}
resolved = (out_root / "resolved_configs")
resolved.mkdir(parents=True, exist_ok=True)
for fraction in fractions:
    token = token_map.get(fraction, str(fraction).replace(".", "p"))
    for method_id, config_path in specs:
        config = load_config(config_path, overrides=overrides)
        config["split"]["labeled_fraction"] = fraction
        config["runtime"]["output_root"] = str(out_root / token)
        config["runtime"]["run_name"] = method_id
        target = resolved / f"{method_id}_{token}.yaml"
        dump_config(config, target)
        print(f"{method_id}|{token}|{target}")
PY

# 2) 逐 fold/seed 训练 + 评估（已有结果的自动跳过）
for fraction in $LABEL_FRACTIONS; do
  case "$fraction" in
    0.125) token="12p5" ;;
    0.5) token="50" ;;
    1.0) token="100" ;;
    *) token="${fraction//./p}" ;;
  esac
  for spec in $METHODS; do
    method_id="${spec%%=*}"
    cfg="$SUPP_ROOT/label_efficiency/resolved_configs/${method_id}_${token}.yaml"
    [[ -f "$cfg" ]] || { echo "Missing resolved config: $cfg" >&2; exit 2; }
    for seed in $LABEL_SEEDS; do
      for fold in 0 1 2 3 4; do
        run_dir="$SUPP_ROOT/label_efficiency/$token/$method_id/fold${fold}_seed${seed}"
        [[ -f "$run_dir/evaluation_summary.json" ]] && { echo "skip completed: $run_dir"; continue; }
        mkdir -p "$run_dir"
        echo "train $method_id fraction=$fraction fold=$fold seed=$seed"
        timeout 28800 python -m kidney_ssl.cli train --config "$cfg" --fold "$fold" --seed "$seed" \
          > "$run_dir/train.log" 2>&1 || { echo "FAILED train: $run_dir" >&2; tail -n 20 "$run_dir/train.log" >&2; exit 2; }
        python -m kidney_ssl.cli evaluate --run-dir "$run_dir" \
          > "$run_dir/evaluate.log" 2>&1 || { echo "FAILED evaluate: $run_dir" >&2; tail -n 20 "$run_dir/evaluate.log" >&2; exit 2; }
        seconds_int=$(tail -n 1 "$run_dir/metrics.jsonl" 2>/dev/null | python -c "import json,sys; print(int(round(float(json.loads(sys.stdin.read())['elapsed_seconds']))))" 2>/dev/null || echo 0)
        echo "  done in ~$((seconds_int / 60)) min"
      done
    done
  done
done

# 3) 汇总：label_efficiency_summary.csv + 100pct_summary.json（写入正式 benchmark 根，供 q2 gate 读取）
python - "$SUPP_ROOT/label_efficiency" "$ROOT" "$LABEL_FRACTIONS" $METHODS <<'PY'
import csv
import json
import sys
from pathlib import Path

supp, formal_root = Path(sys.argv[1]), Path(sys.argv[2])
fractions = [float(v) for v in sys.argv[3].split()]
token_map = {0.125: "12p5", 0.5: "50", 1.0: "100"}
rows = []
for fraction in fractions:
    token = token_map.get(fraction, str(fraction).replace(".", "p"))
    for item in sys.argv[4:]:
        method_id = item.split("=", 1)[0]
        dice = []
        for path in (supp / token / method_id).glob("fold*_seed*/evaluation_summary.json"):
            dice.append(float(json.loads(path.read_text(encoding="utf-8"))["overall"]["dice"]))
        rows.append({"labeled_fraction": fraction, "method": method_id,
                     "mean_dice": round(float(sum(dice) / len(dice)), 6) if dice else None,
                     "n_runs": len(dice)})
with (supp / "label_efficiency_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["labeled_fraction", "method", "mean_dice", "n_runs"])
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(row)

# 总训练耗时（取自所有 metrics.jsonl 最后一个 epoch 的累计 elapsed_seconds）
total_seconds = 0.0
for path in supp.rglob("metrics.jsonl"):
    try:
        last_line = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][-1]
        total_seconds += float(json.loads(last_line)["elapsed_seconds"])
    except Exception:  # noqa: BLE001
        pass
print(f"measured_training_seconds_total={total_seconds:.0f} (~{total_seconds / 3600:.2f} h)")

# 100% 汇总：仅当存在有效结果时写入正式 benchmark 根；否则清理旧文件，避免 q2 gate 读 None 崩溃
gate_dir = formal_root / "label_efficiency"
gate_dir.mkdir(parents=True, exist_ok=True)
gate_file = gate_dir / "100pct_summary.json"
full_rows = [row for row in rows if abs(row["labeled_fraction"] - 1.0) < 1e-8]
valid = [row for row in full_rows if row["mean_dice"] is not None]
if valid:
    proposed = [row for row in valid if row["method"] == "KidneyBand_ReliableSSL_v2_2_CSD"]
    source = proposed[0] if proposed else valid[0]
    summary = {"labeled_fraction": 1.0, "proposed_mean_dice": source["mean_dice"], "n_runs": source["n_runs"]}
    gate_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", gate_file, summary)
else:
    if gate_file.exists():
        gate_file.unlink()
    print("100% label runs not finished yet; removed stale", gate_file)
PY

# 4) 重新生成 q2 gate 汇总（full_label_sanity 应由 false 变为 true，若 Dice>=0.90）
if [[ "${SKIP_GATE:-0}" != "1" ]]; then
  if python -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('proposed_mean_dice') is not None else 1)" "$ROOT/label_efficiency/100pct_summary.json" 2>/dev/null; then
    kidney-seg compare --benchmark-dir "$ROOT" --suite "$FORMAL_SUITE" --q2-gate \
      2>&1 | tee "$LOG_ROOT/q2_gate_summary.log"
  else
    echo "100% label runs incomplete; skipping q2 gate regeneration (rerun after all runs finish)"
  fi
fi
echo "label-efficiency supplement finished; outputs under $SUPP_ROOT/label_efficiency"
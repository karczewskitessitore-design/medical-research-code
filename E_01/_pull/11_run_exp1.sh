#!/bin/bash
# Exp1 驱动: 逐细胞类型并行跑 scTenifoldKnk (每个任务 1 核)
set -u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/vko/env
cd /root/autodl-tmp/vko
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=${OUT:-results/exp1}
BASE=${BASE:-data/proc/gse183276/celltype_counts}
NGENE=${NGENE:-2500}
NCELL=${NCELL:-1500}
NNET=${NNET:-10}
NCELLS=${NCELLS:-500}
SEED=${SEED:-1}

mkdir -p "$OUT" logs/exp1
CTS=${CTS:-"PCT PT_injured TAL DTL DCT CNT CD_PC CD_IC Podocyte GEnC PEC Fibroblast VSMC_P Mesangium T_cell NK_cell B_Plasma Myeloid Mast"}

echo "[start] $(date)  OUT=$OUT  G=$NGENE cells=$NCELL nNet=$NNET"
for CT in $CTS; do
  if [ ! -f "$BASE/$CT.mtx" ]; then echo "[skip] $CT (no matrix)"; continue; fi
  Rscript scripts/11_run_exp1.R --ct "$CT" --base "$BASE" --out "$OUT" \
      --ngene "$NGENE" --ncell "$NCELL" --nNet "$NNET" --nCells "$NCELLS" \
      --nCores 1 --seed "$SEED" > "logs/exp1/$CT.log" 2>&1 &
  echo "  launched $CT  pid=$!"
done
wait
echo "[done] $(date)"
grep -h "\[TIME\]" logs/exp1/*.log | sort
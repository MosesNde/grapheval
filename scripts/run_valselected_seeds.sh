#!/bin/bash
# Five seeds of the VALIDATION-SELECTED configuration (epochs 30, effective
# batch 128, lr 2e-4). Seed 1 already exists from the sweep, so this adds 0,2,3,4.
# The configuration was chosen on best_val_auc across a 12-cell grid without
# consulting test; this run establishes its seed variance for reporting.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
GPUS=(0 2 3)
TAG=valsel
run(){ local s=$1 G=$2
  [ -f "$R/repo-gat-rev-s${s}-${TAG}_single.json" ] && { echo "SKIP  s$s"; return 0; }
  echo "START valsel s$s -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u train_single.py --mode repo --backbone gat --reverse-edges \
      --seed "$s" --epochs 30 --batch-size 16 --accum 8 --lr 2e-4 \
      --tag "$TAG" > "$R/valsel_s${s}.log" 2>&1
  echo "DONE  valsel s$s exit=$?"; }
i=0
for s in 0 2 3 4; do
  run "$s" "${GPUS[$((i % 3))]}" &
  i=$((i+1)); if [ $((i % 3)) -eq 0 ]; then wait; else sleep 20; fi
done
wait; echo "VALSEL_DONE"

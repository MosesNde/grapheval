#!/bin/bash
# Recipe sensitivity, selected on VALIDATION ONLY.
#
# The two recipes compared so far are not symmetric: train_cwe's CONFIG was
# tuned for a corpus ~10x smaller and had its batch raised for throughput, while
# train_single's schedule "follows the runs that produced 0.652" -- a TEST score.
# Selecting on test is exactly what this paper criticises, so the comparison
# cannot stand as evidence of recipe sensitivity.
#
# This sweep varies epochs, effective batch and learning rate within ONE
# implementation, so the only thing changing is the recipe. Selection is by
# best_val_auc, which train_single records separately from the test metrics.
# Test is not consulted until a configuration has been chosen.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
GPUS=(0 2 3)
SEED=1

run(){ local ep=$1 accum=$2 lr=$3 G=$4
  local tag="sw_e${ep}_a${accum}_lr${lr}"
  [ -f "$R/repo-gat-rev-s${SEED}-${tag}_single.json" ] && { echo "SKIP  $tag"; return 0; }
  echo "START $tag -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u train_single.py --mode repo --backbone gat --reverse-edges \
      --seed $SEED --epochs "$ep" --batch-size 16 --accum "$accum" --lr "$lr" \
      --tag "$tag" > "$R/sweep_${tag}.log" 2>&1
  echo "DONE  $tag exit=$?"; }

JOBS=()
for ep in 15 30; do for accum in 8 16; do for lr in 1e-3 5e-4 2e-4; do
  JOBS+=("$ep:$accum:$lr"); done; done; done
echo "${#JOBS[@]} configurations"
i=0
for j in "${JOBS[@]}"; do
  IFS=':' read -r ep accum lr <<< "$j"
  run "$ep" "$accum" "$lr" "${GPUS[$((i % 3))]}" &
  i=$((i+1)); if [ $((i % 3)) -eq 0 ]; then wait; else sleep 20; fi
done
wait; echo "RECIPE_SWEEP_DONE"

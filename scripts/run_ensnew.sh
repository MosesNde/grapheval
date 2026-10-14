#!/bin/bash
# Second controlled arm: train_single.py's recipe applied to a five-fold ensemble.
# Completes the 2x2 with run_stage1 (old recipe ensemble), train_fullsingle (old
# recipe single 5/5) and train_single.py (new recipe single 5/5).
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
GPUS=(0 2 3)
run(){ local rev=$1 s=$2 G=$3
  local sfx="" revarg=""
  [ "$rev" = "rev" ] && { revarg="--reverse-edges"; sfx="-rev"; }
  local tag="repo-gat${sfx}-s${s}"
  [ -f "$R/${tag}_ensnew.json" ] && { echo "SKIP  $tag"; return 0; }
  echo "START $tag -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u train_ensemble_newrecipe.py --mode repo --backbone gat $revarg --seed "$s" \
    > "$R/ensnew_gat${sfx}_s${s}.log" 2>&1
  echo "DONE  $tag exit=$?"; }
JOBS=(); for s in 0 1 2 3 4; do JOBS+=("rev:$s"); done
i=0
for j in "${JOBS[@]}"; do
  run "${j%%:*}" "${j#*:}" "${GPUS[$((i % 3))]}" &
  i=$((i+1)); if [ $((i % 3)) -eq 0 ]; then wait; else sleep 20; fi
done
wait; echo "ENSNEW_DONE"

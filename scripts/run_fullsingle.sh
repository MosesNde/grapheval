#!/bin/bash
# Controlled ensemble-vs-single arm: one model on 5/5 with the IDENTICAL recipe
# run_stage1 uses per fold (batch 256, 120 epochs, patience 20, same cfg, same
# held-out validation partition). Five seeds, directed and bidirectional.
#
# Round-robin GPU by job index; polling memory.used races because allocation lags.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
GPUS=(0 2 3)

run(){ local rev=$1 s=$2 G=$3
  local sfx=""; local revarg=""
  [ "$rev" = "rev" ] && { revarg="--reverse-edges"; sfx="-rev"; }
  local tag="repo-gat${sfx}-s${s}"
  [ -f "$R/${tag}_fullsingle.json" ] && { echo "SKIP  $tag"; return 0; }
  echo "START $tag -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u train_fullsingle.py --mode repo --backbone gat $revarg --seed "$s" \
    > "$R/fullsingle_gat${sfx}_s${s}.log" 2>&1
  echo "DONE  $tag exit=$?"; }

JOBS=(); for rev in dir rev; do for s in 0 1 2 3 4; do JOBS+=("$rev:$s"); done; done
i=0
for j in "${JOBS[@]}"; do
  run "${j%%:*}" "${j#*:}" "${GPUS[$((i % 3))]}" &
  i=$((i+1))
  if [ $((i % 3)) -eq 0 ]; then wait; else sleep 20; fi
done
wait
echo "FULLSINGLE_DONE"

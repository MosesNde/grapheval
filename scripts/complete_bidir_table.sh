#!/bin/bash
# Complete the protocol x encoding table: group bidirectional (5 seeds) and
# random bidirectional seeds 3-4. All 5-fold, matching every other cell.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
JOBS=("group|" "group|--seed 1" "group|--seed 2" "group|--seed 3" "group|--seed 4"
      "random|--seed 3" "random|--seed 4")
Q=/tmp/bidq.$$; L=$Q.lock; echo 0 > $Q
worker(){ local gpu=$1
  while true; do
    local i; i=$( { flock 9; v=$(cat $Q); echo $((v+1)) > $Q; echo "$v"; } 9>$L )
    [ "$i" -ge "${#JOBS[@]}" ] && break
    IFS='|' read -r m s <<< "${JOBS[$i]}"
    local nm="$m-gat-rev$(echo "$s" | grep -o '[0-9]*' | sed 's/^/-s/')"
    echo "[gpu $gpu] $nm"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u train_cwe.py --stage 1 --mode "$m" \
      --reverse-edges --workers 6 $s > "$R/bid_$nm.log" 2>&1
    rc=$?; [ $rc -ne 0 ] && echo "[gpu $gpu] FAILED rc=$rc : $nm" || echo "[gpu $gpu] ok $nm"
  done
}
for g in 1 2 3; do worker $g & done
wait; rm -f $Q $L; echo "BIDIR_TABLE done"

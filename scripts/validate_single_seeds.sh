#!/bin/bash
# Compare like with like: 5 seeds of train_single against the 5-seed frozen arm
# from joint_finetune (0.6519 +- 0.0096). Single-seed differences of ~0.008 are
# inside that spread, so only the means are informative.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q=/tmp/vsq.$$; L=$Q.lock; echo 0 > $Q
JOBS=(42 1 2 3 4)
worker(){ local gpu=$1
  while true; do
    local i; i=$( { flock 9; v=$(cat $Q); echo $((v+1)) > $Q; echo "$v"; } 9>$L )
    [ "$i" -ge "${#JOBS[@]}" ] && break
    local s=${JOBS[$i]}
    echo "[gpu $gpu] seed $s"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u train_single.py --mode repo --reverse-edges \
      --seed $s --tag vseed > "$R/vseed_s$s.log" 2>&1
    echo "[gpu $gpu] seed $s exit=$?"
  done
}
for g in 0 1 2; do worker $g & done
wait; rm -f $Q $L; echo "VALIDATE_SEEDS done"

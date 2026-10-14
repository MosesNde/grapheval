#!/bin/bash
# repo seeds 2 and 4 died of CUDA OOM: doubling the edge set doubles the
# attention tensors, and both landed on a GPU another job had grown into.
# Rerun them one at a time, on the emptiest GPU, with expandable segments.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# wait for the in-flight matrix to clear so nothing competes
while pgrep -u "$(id -u)" -f "train_cwe.py.*reverse-edges" > /dev/null; do sleep 60; done
echo "matrix clear; starting reruns"

emptiest(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }

for s in 2 4; do
  G=$(emptiest)
  echo "rerun repo seed $s -> gpu $G (used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $G))"
  CUDA_VISIBLE_DEVICES=$G $PY -u train_cwe.py --stage 1 --mode repo --backbone gat \
    --reverse-edges --seed $s > $R/rev_repo-s$s.log 2>&1
  echo "repo-s$s exit=$?"
done
echo "REV_RERUN done"

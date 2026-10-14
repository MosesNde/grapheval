#!/bin/bash
# Matched baseline: identical data, loop, schedule and architecture to the joint
# runs, encoder frozen. The 5-fold frozen numbers train on 80% of the partition
# and so cannot isolate joint training from training-set size.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
emptiest(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }
for s in 42 1 2 3 4; do
  G=$(emptiest); echo "frozen matched seed $s -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u joint_finetune.py --mode repo --reverse-edges \
    --epochs 15 --batch-size 16 --accum 8 --freeze-encoder --seed $s \
    --tag frozenmatched > $R/frozmatch_s$s.log 2>&1
  echo "seed $s exit=$?"
done
echo "FROZEN_MATCHED done"

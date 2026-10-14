#!/bin/bash
# Positive control: can this pipeline learn a label that lives ONLY in the edges?
# Waits for the backbone-seed matrix to clear first.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
emptiest(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }
while pgrep -u "$(id -u)" -f "train_cwe.py --stage 1 --mode repo --backbone" > /dev/null; do sleep 60; done
echo "backbone matrix clear; starting structural control"
for task in cfg-cycle ddg-density; do
  for s in 1 2 3; do
    G=$(emptiest); echo "start $task seed $s -> gpu $G"
    CUDA_VISIBLE_DEVICES=$G $PY -u structural_control.py --task $task --mode repo --seed $s \
      > $R/struct_${task}_s$s.log 2>&1
    echo "$task s$s exit=$?"
  done
done
echo "STRUCTURAL_CONTROL done"

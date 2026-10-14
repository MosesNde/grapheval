#!/bin/bash
# Fourth attempt. The group training partition holds a 14,947-node graph
# (median 48); GAT attention over its 30,757 edges produces a kernel long
# enough to trip the workstation driver watchdog. Batch 32 shortens each
# launch. Nothing else about the run changes, so the result stays comparable
# with the other four group seeds.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -u "$(id -u)" -f "train_cwe.py --stage 1" > /dev/null; do sleep 60; done
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
echo "group-gat-s2, attempt 4: gpu $G, batch 32"
CUDA_VISIBLE_DEVICES=$G $PY -u train_cwe.py --stage 1 --mode group --seed 2 \
  --workers 4 --batch-size 32 > $R/rp2_group-gat-s2.log 2>&1
echo "exit=$?"

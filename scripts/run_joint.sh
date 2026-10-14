#!/bin/bash
# Joint encoder+graph fine-tuning. Resolves the frozen-vs-fine-tuned confound
# that the paper currently reports as unresolved.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
emptiest(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }
while pgrep -u "$(id -u)" -f "fuse_graph_token.py" > /dev/null; do sleep 60; done
echo "fusion clear; starting joint fine-tuning"
G=$(emptiest); echo "-> gpu $G"
CUDA_VISIBLE_DEVICES=$G $PY -u joint_finetune.py --mode repo --reverse-edges \
  --epochs 6 --batch-size 8 > $R/joint_repo.log 2>&1
echo "joint exit=$?"
echo "JOINT done"

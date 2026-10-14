#!/bin/bash
cd /home/moses/TSE_Journal/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -u "$(id -u)" -f "complete_bidir_table.sh" > /dev/null; do sleep 60; done
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
echo "validation: repo bidirectional single-model on gpu $G (expect ~0.652)"
CUDA_VISIBLE_DEVICES=$G /home/moses/venv/bin/python -u train_single.py --mode repo \
  --reverse-edges --seed 42 --tag validate > /home/moses/TSE_Journal/runs/validate_single.log 2>&1
echo "exit=$?"
grep "single-model test" /home/moses/TSE_Journal/runs/validate_single.log

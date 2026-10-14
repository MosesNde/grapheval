#!/bin/bash
cd /home/moses/TSE_Journal/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
echo "repo bidirectional, single model, node-cap 400 -> gpu $G"
CUDA_VISIBLE_DEVICES=$G /home/moses/venv/bin/python -u train_single.py --mode repo \
  --reverse-edges --seed 42 --node-cap 400 --tag validatecap \
  > /home/moses/TSE_Journal/runs/validate_cap.log 2>&1
echo "exit=$?"; grep -E "node-cap|single-model test" /home/moses/TSE_Journal/runs/validate_cap.log

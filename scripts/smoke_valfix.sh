#!/bin/bash
cd /home/moses/TSE_Journal/scripts
for g in 0 1 2 3; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$u" -lt 25000 ] && { G=$g; break; }; done
CUDA_VISIBLE_DEVICES=$G /home/moses/venv/bin/python -u train_cwe.py --stage 1 --mode repo \
  --backbone gat --reverse-edges --limit 1500 --epochs 12 --patience 5 \
  > /home/moses/TSE_Journal/runs/smoke_valfix.log 2>&1
echo "exit=$?"

#!/bin/bash
cd /home/moses/TSE_Journal/scripts
for g in 0 1 2 3; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$u" -lt 25000 ] && { G=$g; break; }
done
[ -z "$G" ] && { echo NO_FREE_GPU; exit 1; }
echo "residual analysis -> gpu $G"
CUDA_VISIBLE_DEVICES=$G /home/moses/venv/bin/python -u analyze_residual.py \
  --mode repo --epochs 60 > /home/moses/TSE_Journal/runs/residual.log 2>&1
echo "RESID exit=$?"

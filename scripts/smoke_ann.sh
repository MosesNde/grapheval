#!/bin/bash
cd /home/moses/TSE_Journal/scripts
for g in 0 1 2 3; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$u" -lt 25000 ] && { G=$g; break; }; done
for m in structural hashed; do
  echo "=== $m ==="
  CUDA_VISIBLE_DEVICES=$G /home/moses/venv/bin/python -u train_cwe.py --stage 1 --mode repo \
    --backbone gat --edge-ann $m --limit 800 > /home/moses/TSE_Journal/runs/smoke_ann_$m.log 2>&1
  echo "exit=$?"
done
echo SMOKE_DONE

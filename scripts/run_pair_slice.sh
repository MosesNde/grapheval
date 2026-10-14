#!/bin/bash
cd /home/moses/TSE_Journal/scripts
for g in 0 1 2 3; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$u" -lt 25000 ] && { G=$g; break; }; done
[ -z "${G:-}" ] && { echo NO_FREE_GPU; exit 1; }
echo "pair-slice -> gpu $G"
CUDA_VISIBLE_DEVICES=$G /home/moses/venv/bin/python -u pair_slice_test.py --mode repo \
  > /home/moses/TSE_Journal/runs/pair_slice.log 2>&1
echo "PAIRSLICE exit=$?"

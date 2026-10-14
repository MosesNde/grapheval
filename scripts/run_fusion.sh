#!/bin/bash
cd /home/moses/TSE_Journal/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for g in 0 1 2 3; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$u" -lt 25000 ] && { G=$g; break; }; done
[ -z "$G" ] && { echo NO_FREE_GPU; exit 1; }
echo "fusion -> gpu $G"
CUDA_VISIBLE_DEVICES=$G /home/moses/venv/bin/python -u fuse_graph_token.py --mode repo \
  > /home/moses/TSE_Journal/runs/fusion.log 2>&1
echo "FUSION exit=$?"

#!/bin/bash
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
emptiest(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }

G=$(emptiest); echo "joint v2 -> gpu $G"
CUDA_VISIBLE_DEVICES=$G $PY -u joint_finetune.py --mode repo --reverse-edges \
  --epochs 15 --batch-size 16 --accum 8 > $R/joint_v2.log 2>&1
echo "joint_v2 exit=$?"

G=$(emptiest); echo "fusion rerun -> gpu $G"
CUDA_VISIBLE_DEVICES=$G $PY -u fuse_graph_token.py --mode repo > $R/fusion_v2.log 2>&1
echo "fusion_v2 exit=$?"
echo "JOINT_V2 done"

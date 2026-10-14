#!/bin/bash
cd /home/moses/TSE_Journal/scripts
CUDA_VISIBLE_DEVICES=0 /home/moses/venv/bin/python -u train_cwe.py --stage 1 \
  --backbone gtrans --mode repo --batch-size 512 --workers 6 \
  --epochs 100 --patience 15 --cache /home/moses/graph_cache_cwe \
  > /home/moses/TSE_Journal/runs/gtrans_repo.log 2>&1
echo "GTRANS exit=$?"

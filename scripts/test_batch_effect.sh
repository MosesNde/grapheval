#!/bin/bash
# Is group-gat-s2's 0.6139 caused by batch 32, or is seed 2 simply a low draw?
# Rerun seed 1 -- which scored 0.6564 at batch 256 -- at batch 32. If it lands
# near 0.6564 the batch size is irrelevant and s2 is seed noise; if it lands
# near 0.61 the batch size is the cause.
cd /home/moses/TSE_Journal/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
echo "group seed 1 at batch 32 -> gpu $G  (batch 256 gave 0.6564)"
CUDA_VISIBLE_DEVICES=$G /home/moses/venv/bin/python -u train_cwe.py --stage 1 \
  --mode group --seed 1 --workers 4 --batch-size 32 --tag b32test \
  > /home/moses/TSE_Journal/runs/b32test.log 2>&1
echo "exit=$?"
/home/moses/venv/bin/python -c "
import json; d=json.load(open('/home/moses/TSE_Journal/runs/group-gat-s1-b32test_stage1.json'))
print('  batch 32:', round(d['ensemble_mean_only']['roc_auc'],4), ' vs batch 256: 0.6564')"

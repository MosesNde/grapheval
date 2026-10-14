#!/bin/bash
# group-gat-s2 hit cudaErrorLaunchTimeout twice, on two different GPUs, both
# times entering fold 3. That points at an oversized batch producing a kernel
# longer than the driver watchdog rather than at a faulty card, so this retries
# alone with the batch halved.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -u "$(id -u)" -f "train_cwe.py --stage 1" > /dev/null; do sleep 60; done
G=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
echo "retrying group-gat-s2 alone on gpu $G with batch 128"
CUDA_VISIBLE_DEVICES=$G $PY -u train_cwe.py --stage 1 --mode group --seed 2 \
  --workers 6 --batch-size 128 > $R/rp2_group-gat-s2.log 2>&1
echo "exit=$?"

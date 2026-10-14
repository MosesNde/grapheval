#!/bin/bash
# Four more seeds for joint encoder+graph fine-tuning. The single run reached
# 0.6515 but peaked at epoch 1 and then overfit, so the effect needs replication
# before it can be reported.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
emptiest(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }
for s in 1 2 3 4; do
  G=$(emptiest); echo "seed $s -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u joint_finetune.py --mode repo --reverse-edges \
    --epochs 15 --batch-size 16 --accum 8 --seed $s > $R/joint_s$s.log 2>&1
  echo "seed $s exit=$?"
done
echo "JOINT_SEEDS done"

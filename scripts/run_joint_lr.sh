#!/bin/bash
# Best validation landed at epoch 1 in all five seeds, so 0.649 may be a
# stopping artifact rather than a ceiling. Two gentler configurations, one seed
# each first; whichever moves the validation peak later and higher gets seeds.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
emptiest(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }
run(){ G=$(emptiest); echo "$4 -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u joint_finetune.py --mode repo --reverse-edges \
    --epochs 20 --batch-size 16 --accum 8 --encoder-lr $1 --unfrozen-layers $2 \
    --seed $3 --tag "$4" > $R/jlr_$4.log 2>&1
  echo "$4 exit=$?"; }
run 5e-6 4 42 lr5e6_L4
run 2e-5 2 42 lr2e5_L2
run 5e-6 2 42 lr5e6_L2
echo "JOINT_LR done"

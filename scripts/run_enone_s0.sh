#!/bin/bash
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
pick(){ for g in 0 1 2 3; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$u" -lt 25000 ] && { echo $g; return; }; done; echo ""; }
for m in repo random; do
  G=$(pick); [ -z "$G" ] && { echo "$m NO_FREE_GPU"; exit 1; }
  echo "$m enone seed0 -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u train_cwe.py --stage 1 --mode $m --backbone gat \
    --edges none > $R/enone_s0_$m.log 2>&1
  echo "$m exit=$?"
done
echo "ENONE_S0 done"

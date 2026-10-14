#!/bin/bash
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
pick(){ for g in 0 1 2 3; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$u" -lt 25000 ] && { echo $g; return; }; done; echo ""; }
run(){ G=$(pick); [ -z "$G" ] && { echo "seed $1 NO_FREE_GPU"; return 1; }
  echo "start seed $1 -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u pair_slice_test.py --mode repo --seed $1 \
    > $R/pair_slice_s$1.log 2>&1
  echo "seed $1 exit=$?"; }
for pair in "1 2" "3 4"; do
  for s in $pair; do run $s & sleep 40; done
  wait
done
echo "PAIRSLICE_SEEDS done"

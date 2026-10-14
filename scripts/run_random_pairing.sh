#!/bin/bash
# `random` is now n=5 on the full-graph arm, so its paired comparators must
# match: edge-free and bag-of-words arms need seeds 3 and 4 too.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick(){ for g in 0 1 2 3; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$u" -lt 25000 ] && { echo $g; return; }; done; echo ""; }
BOW="--cache /home/moses/graph_cache_bow --tag bow"
run(){ G=$(pick); [ -z "$G" ] && { echo "$3 NO_FREE_GPU"; return 1; }
  echo "start $3 -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u train_cwe.py --stage 1 --mode random --backbone gat \
    --seed $1 $2 > $R/rp_$3.log 2>&1
  echo "$3 exit=$?"; }
JOBS=("3|--edges none|enone-s3" "4|--edges none|enone-s4"
      "3|$BOW|bow-s3"  "4|$BOW|bow-s4"
      "3|--edges none $BOW|enonebow-s3" "4|--edges none $BOW|enonebow-s4")
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for k in 0 1; do
    j=${JOBS[$((i+k))]:-}; [ -z "$j" ] && continue
    IFS='|' read -r s f l <<< "$j"
    run "$s" "$f" "$l" & sleep 40
  done
  wait; i=$((i+2))
done
echo "RANDOM_PAIRING done"

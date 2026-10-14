#!/bin/bash
# Bring every rung of the ladder to 5 seeds. The `group` rung was n=1, and the
# 46/23/31 decomposition is computed from it, so its seed noise propagates
# directly into a headline number.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick(){ for g in 0 1 2 3; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$u" -lt 25000 ] && { echo $g; return; }; done; echo ""; }
run(){ G=$(pick); [ -z "$G" ] && { echo "$3 NO_FREE_GPU"; return 1; }
  echo "start $3 -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u train_cwe.py --stage 1 --mode $1 --backbone gat --seed $2 \
    > $R/ladder_$3.log 2>&1
  echo "$3 exit=$?"; }
JOBS=("group|1|group-s1" "group|2|group-s2" "group|3|group-s3" "group|4|group-s4"
      "random|3|random-s3" "random|4|random-s4")
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for k in 0 1; do
    j=${JOBS[$((i+k))]:-}; [ -z "$j" ] && continue
    IFS='|' read -r m s l <<< "$j"
    run "$m" "$s" "$l" & sleep 40
  done
  wait; i=$((i+2))
done
echo "LADDER_SEEDS done"

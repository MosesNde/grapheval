#!/bin/bash
# Capacity control: same architecture and parameter count as the `hashed` arm,
# with the edge-to-annotation correspondence destroyed.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
pick(){ for g in 0 1 2 3; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$u" -lt 25000 ] && { echo $g; return; }; done; echo ""; }
run(){ G=$(pick); [ -z "$G" ] && { echo "$3 NO_FREE_GPU"; return 1; }
  echo "start $3 -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u train_cwe.py --stage 1 --mode $1 --backbone gat \
    --edge-ann shuffled $2 > $R/shuf_$3.log 2>&1
  echo "$3 exit=$?"; }
JOBS=("repo| |repo-s0" "repo|--seed 1|repo-s1" "repo|--seed 2|repo-s2" "repo|--seed 3|repo-s3" "repo|--seed 4|repo-s4"
      "random| |random-s0" "random|--seed 1|random-s1" "random|--seed 2|random-s2")
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for k in 0 1; do
    j=${JOBS[$((i+k))]:-}; [ -z "$j" ] && continue
    IFS='|' read -r m s l <<< "$j"
    run "$m" "$s" "$l" & sleep 40
  done
  wait; i=$((i+2))
done
echo "SHUFFLE_CONTROL done"

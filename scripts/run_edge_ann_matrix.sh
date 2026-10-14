#!/bin/bash
# 2 encodings x 2 protocols x seeds, against the existing --edges all baselines.
# 2-way concurrency: 4 concurrent jobs previously exhausted host memory.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
pick(){ for g in 0 1 2 3; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g); [ "$u" -lt 25000 ] && { echo $g; return; }; done; echo ""; }
run(){ # mode ann seedflag label
  G=$(pick); [ -z "$G" ] && { echo "$4 NO_FREE_GPU"; return 1; }
  echo "start $4 -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u train_cwe.py --stage 1 --mode $1 --backbone gat \
    --edge-ann $2 $3 > $R/ann_$4.log 2>&1
  echo "$4 exit=$?"
}
JOBS=()
for ann in hashed structural; do
  for s in "" "--seed 1" "--seed 2" "--seed 3" "--seed 4"; do
    JOBS+=("repo|$ann|$s|repo-$ann-${s// /}")
  done
  for s in "" "--seed 1" "--seed 2"; do
    JOBS+=("random|$ann|$s|random-$ann-${s// /}")
  done
done
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for k in 0 1; do
    j=${JOBS[$((i+k))]:-}; [ -z "$j" ] && continue
    IFS='|' read -r m a s l <<< "$j"
    run "$m" "$a" "$s" "$l" & sleep 40
  done
  wait; i=$((i+2))
done
echo "ANN_MATRIX done"

#!/bin/bash
# 4 seed replicates of edges=untyped on repo, to match the 5-seed all-edges baseline.
# 2-way concurrency: 4 concurrent jobs previously exceeded host memory.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python
R=/home/moses/TSE_Journal/runs
pick_gpu() {
  for g in 0 1 2 3; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    [ "$u" -lt 25000 ] && { echo $g; return; }
  done
  echo ""
}
run_one() {
  s=$1; G=$(pick_gpu)
  if [ -z "$G" ]; then echo "seed $s: NO_FREE_GPU"; return 1; fi
  echo "seed $s -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u train_cwe.py --stage 1 --mode repo \
    --backbone gat --edges untyped --seed $s > $R/untyped_s$s.log 2>&1
  echo "seed $s exit=$?"
}
for pair in "1 2" "3 4"; do
  for s in $pair; do run_one $s & sleep 45; done
  wait
done
echo "UNTYPED_SEEDS done"

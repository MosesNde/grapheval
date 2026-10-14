#!/bin/bash
# Bring GCN, GIN and the graph transformer to 5 seeds in both encodings, so the
# directed-vs-bidirectional comparison is uniform with GAT.
# Bidirectional graphs are ~2x the edges and OOM'd at 2-way earlier, so a failed
# job is retried once, alone, on the emptiest card.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
emptiest(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }

run(){ # backbone seed revflag label
  G=$(emptiest)
  echo "start $4 -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u train_cwe.py --stage 1 --mode repo \
    --backbone $1 --seed $2 $3 > $R/bb_$4.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$4 exit=$rc — retrying alone"
    while pgrep -u "$(id -u)" -f "train_cwe.py --stage 1 --mode repo --backbone" | grep -qv $$; do sleep 45; done
    G=$(emptiest)
    CUDA_VISIBLE_DEVICES=$G $PY -u train_cwe.py --stage 1 --mode repo \
      --backbone $1 --seed $2 $3 > $R/bb_$4.log 2>&1
    rc=$?
  fi
  echo "$4 exit=$rc"
}

JOBS=()
for b in gcn gin gtrans; do
  for s in 1 2 3 4; do
    JOBS+=("$b|$s||${b}-dir-s$s")
    JOBS+=("$b|$s|--reverse-edges|${b}-rev-s$s")
  done
done
echo "queued ${#JOBS[@]} runs"
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for k in 0 1; do
    j=${JOBS[$((i+k))]:-}; [ -z "$j" ] && continue
    IFS='|' read -r b s f l <<< "$j"
    run "$b" "$s" "$f" "$l" & sleep 45
  done
  wait; i=$((i+2))
done
echo "BACKBONE_SEEDS done"

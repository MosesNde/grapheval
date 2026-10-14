#!/bin/bash
# Three follow-ups, in priority order. Bidirectional graphs are ~2x the edges
# and OOM'd twice at 2-way concurrency, so depth arms run one at a time.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
emptiest(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }
solo(){ G=$(emptiest); echo "start $2 -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u $1 > $R/$2.log 2>&1; echo "$2 exit=$?"; }

echo "### 1. depth: 6 layers bidirectional, and 6 layers directed as capacity control"
for s in "" "--seed 1" "--seed 2" "--seed 3" "--seed 4"; do
  t=${s// /}; t=${t:-s0}
  solo "train_cwe.py --stage 1 --mode repo --backbone gat --reverse-edges --layers 6 $s" "L6rev_$t"
  solo "train_cwe.py --stage 1 --mode repo --backbone gat --layers 6 $s"                 "L6dir_$t"
done

echo "### 2. paired slice under reverse edges"
for s in 1 2 3; do
  solo "pair_slice_test.py --mode repo --seed $s --reverse-edges" "pairslice_rev_s$s"
done

echo "### 3. other backbones, bidirectional (single seed, matching the arch table)"
for b in gcn gin gtrans; do
  solo "train_cwe.py --stage 1 --mode repo --backbone $b --reverse-edges" "rev_$b"
done
echo "FOLLOWUPS done"

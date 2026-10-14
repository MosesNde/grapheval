#!/bin/bash
# Bring the bidirectional pair-slice to five seeds. Seeds 1-3 already exist;
# this adds seed 0 (unsuffixed) and seed 4.
# Explicit GPU assignment: polling memory.used races because allocation lags.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
run(){ local s=$1 G=$2 seedarg="" sfx=""
  [ -n "$s" ] && { seedarg="--seed $s"; sfx="-s$s"; }
  local out="$R/pair_slice_repo${sfx}-rev.json"
  [ -f "$out" ] && { echo "SKIP seed ${s:-0} (exists)"; return 0; }
  echo "START pair-slice rev seed ${s:-0} -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u pair_slice_test.py --mode repo --reverse-edges $seedarg \
    > "$R/pairslice_rev_s${s:-0}.log" 2>&1
  echo "DONE seed ${s:-0} exit=$?"; }
run ""  0 &
sleep 30
run 4   2 &
wait
echo "PAIRSLICE_REV_FILL_DONE"

#!/bin/bash
# Serial rerun of the corrected pair-slice script. The 3-way version was killed
# by host OOM (exit 137): each job loads the full graph cache and trains two
# models (graph and no-graph arms), so three at once exceeds host memory even
# though GPU memory was fine. One at a time, ~1h each.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
for s in "" 2 3 4; do
  sfx=""; seedarg=""
  [ -n "$s" ] && { seedarg="--seed $s"; sfx="-s$s"; }
  f="$R/pair_slice_repo${sfx}.json"
  if [ -f "$f" ] && $PY -c "
import json,sys; sys.exit(0 if '_disposition' in json.load(open('$f')) else 1)" 2>/dev/null; then
    echo "SKIP  seed ${s:-0} (already corrected)"; continue; fi
  echo "START seed ${s:-0} $(date '+%H:%M')"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u pair_slice_test.py --mode repo $seedarg \
    > "$R/pairslice_disp_s${s:-0}.log" 2>&1
  echo "DONE  seed ${s:-0} exit=$? $(date '+%H:%M')"
done
echo "PAIRSLICE_SERIAL_DONE"

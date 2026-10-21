#!/bin/bash
# Five seeds of the corrected pair-slice script.
#
# The rerun of seed 1 did not reproduce its own earlier result (structure-only
# 0.497 -> 0.600 on 165 pairs, identical training code), so the per-seed spread
# on these slices is far larger than the stored five-seed artifacts suggested.
# This re-measures all five with the exhaustive three-bucket disposition and
# per-pair win indicators, so both the seed-level and pair-level uncertainty can
# be reported honestly.
#
# Seed 1 already exists from the corrected single run; this adds 0, 2, 3, 4.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
GPUS=(0 2 3)
run(){ local s=$1 G=$2
  local sfx=""; local seedarg=""
  [ -n "$s" ] && { seedarg="--seed $s"; sfx="-s$s"; }
  [ -f "$R/pair_slice_repo${sfx}.json" ] && \
    /home/moses/venv/bin/python -c "
import json,sys; d=json.load(open('$R/pair_slice_repo${sfx}.json'))
sys.exit(0 if '_disposition' in d else 1)" && { echo "SKIP  seed ${s:-0}"; return 0; }
  echo "START seed ${s:-0} -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u pair_slice_test.py --mode repo $seedarg \
    > "$R/pairslice_disp_s${s:-0}.log" 2>&1
  echo "DONE  seed ${s:-0} exit=$?"; }
i=0
for s in "" 2 3 4; do
  run "$s" "${GPUS[$((i % 3))]}" &
  i=$((i+1)); if [ $((i % 3)) -eq 0 ]; then wait; else sleep 20; fi
done
wait; echo "PAIRSLICE_DISPOSITION_DONE"

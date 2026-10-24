#!/bin/bash
# Paired vulnerable/patched evaluation with full win/loss/tie disposition.
#
# The previous runs stored only per-pair win booleans, which had already folded
# ties into losses, so the tie mass could not be recovered without retraining.
# These runs record the raw per-pair scores, so wins, losses, ties, any tie
# tolerance and any pair-level interval can be recomputed afterwards.
#
# Serial: three concurrent pair-slice jobs OOM-killed this host (exit 137).
# Output carries the -tieaware tag so the earlier artifacts are preserved.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
for s in 0 1 2 3 4; do
  out="$R/pair_slice_repo-s${s}-tieaware.json"
  [ -f "$out" ] && { echo "SKIP  seed $s"; continue; }
  echo "START seed $s $(date '+%H:%M')"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u pair_slice_test.py --mode repo --seed "$s" --tag tieaware \
      > "$R/pairslice_tieaware_s${s}.log" 2>&1
  echo "DONE  seed $s exit=$? $(date '+%H:%M')"
done
echo "PAIRSLICE_TIEAWARE_DONE"

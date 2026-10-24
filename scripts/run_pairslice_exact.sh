#!/bin/bash
# Paired vulnerable/patched evaluation, full-precision re-run.
#
# The -tieaware runs stored per-pair scores rounded to six decimals. On the
# near-identical slice the scores are ~1e-3, so six decimals leaves about three
# significant figures: recomputing outcomes from them inflates apparent ties by
# 228 of 561 pairs and shifts the recomputed tie-aware rate by 0.021. Their
# stored win/loss/tie counts were computed before rounding and remain exact, so
# point estimates and seed-level intervals were unaffected, but the pair-level
# bootstrap had to be rebuilt from the scores and could not be made exact:
# 259 of 561 pairs are unresolvable from rounded data.
#
# These runs store full-precision scores, an explicit per-pair outcome code, and
# the whole test partition's scores with sample ids, so any slice definition or
# tie rule can be re-analysed without retraining again.
#
# Serial: three concurrent pair-slice jobs OOM-killed this host (exit 137).
# Output carries the -exact tag so the -tieaware artifacts are preserved.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
for s in 0 1 2 3 4; do
  out="$R/pair_slice_repo-s${s}-exact.json"
  [ -f "$out" ] && { echo "SKIP  seed $s"; continue; }
  echo "START seed $s $(date '+%H:%M')"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u pair_slice_test.py --mode repo --seed "$s" --tag exact \
      > "$R/pairslice_exact_s${s}.log" 2>&1
  echo "DONE  seed $s exit=$? $(date '+%H:%M')"
done
echo "PAIRSLICE_EXACT_DONE"

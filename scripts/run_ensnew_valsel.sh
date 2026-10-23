#!/bin/bash
# Five-fold ensemble under the VALIDATION-SELECTED recipe, five seeds.
#
# The earlier run_ensnew.sh passed no hyperparameters, so it used this file's
# defaults (15 epochs, lr 5e-4) while the single-model arm used the selected
# recipe (30 epochs, lr 2e-4). Table 7's selected-recipe column was therefore
# not a matched comparison. These runs fix that; the earlier artifacts are kept
# under their own names and are now labelled as the default-recipe ensemble.
#
# Serial: each run trains five members and loads the full graph cache, and
# three concurrent pair-slice jobs already OOMed this host.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
for s in 0 1 2 3 4; do
  out="$R/repo-gat-rev-s${s}_ensnew-valsel.json"
  [ -f "$out" ] && { echo "SKIP  seed $s"; continue; }
  echo "START seed $s $(date '+%H:%M')"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u train_ensemble_newrecipe.py --mode repo --backbone gat --reverse-edges \
      --seed "$s" --epochs 30 --batch-size 16 --accum 8 --lr 2e-4 --tag valsel \
      > "$R/ensnew_valsel_s${s}.log" 2>&1
  echo "DONE  seed $s exit=$? $(date '+%H:%M')"
done
echo "ENSNEW_VALSEL_DONE"

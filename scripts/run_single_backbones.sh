#!/bin/bash
# Bring the non-GAT backbones to five seeds under the SINGLE-MODEL procedure at
# repo, so Table 3 can report both procedures side by side. GAT already has 5.
# Seeds 1-4 only; seed 0 (unsuffixed) already exists for each backbone.
#
# Explicit round-robin GPU assignment: polling memory.used races, because a job
# takes minutes to allocate and consecutive picks all land on the same device.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
GPUS=(0 2 3)

run(){ local bb=$1 s=$2 G=$3
  local out="$R/repo-${bb}-s${s}_single.json"
  [ -f "$out" ] && { echo "SKIP  repo-${bb}-s${s}"; return 0; }
  echo "START repo-${bb}-s${s} -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u train_single.py --mode repo --backbone "$bb" --seed "$s" \
    > "$R/single_bb_${bb}_s${s}.log" 2>&1
  echo "DONE  repo-${bb}-s${s} exit=$?"; }

JOBS=(); for bb in gcn gin gtrans; do for s in 1 2 3 4; do JOBS+=("$bb:$s"); done; done
i=0
for j in "${JOBS[@]}"; do
  run "${j%%:*}" "${j#*:}" "${GPUS[$((i % 3))]}" &
  i=$((i+1))
  if [ $((i % 3)) -eq 0 ]; then wait; else sleep 20; fi
done
wait
echo "SINGLE_BACKBONES_DONE"

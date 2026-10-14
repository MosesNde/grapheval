#!/bin/bash
# Full single-model sweep: one model on the whole training partition, epoch
# chosen on the held-out validation repositories. Replaces the 5-fold tables.
# No node cap: capping truncates 8.4% of training graphs and made no
# significant difference in validation.
# 3-way concurrency; 4-way OOM-killed the earlier 5-fold sweep.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MARK=$R/.single_started; [ -f "$MARK" ] || touch "$MARK"
Q=/tmp/singq.$$; L=$Q.lock; echo 0 > $Q
mapfile -t JOBS < <(grep -v '^#' single_jobs.txt | grep -v '^$')
echo "queued ${#JOBS[@]} runs"

tag_for(){
  local m=$1; local e=$2; local t=$3; local out="$m-gat"
  [[ "$e" == *"--backbone gcn"* ]] && out="$m-gcn"
  [[ "$e" == *"--backbone gin"* ]] && out="$m-gin"
  [[ "$e" == *"--backbone gtrans"* ]] && out="$m-gtrans"
  [[ "$e" == *"type-only"* ]] && out="$out-type-only"
  [[ "$e" == *"topology"* ]] && out="$out-topology"
  [[ "$e" == *"--edges none"* ]] && out="$out-enone"
  [[ "$e" == *"--edges untyped"* ]] && out="$out-euntyped"
  [[ "$e" == *"--reverse-edges"* ]] && out="$out-rev"
  local s; s=$(grep -o '\-\-seed [0-9]*' <<< "$e" | awk '{print $2}')
  [ -n "$s" ] && out="$out-s$s"
  [ -n "$t" ] && out="$out-$t"
  echo "$out"
}
worker(){
  local gpu=$1
  while true; do
    local i; i=$( { flock 9; v=$(cat $Q); echo $((v+1)) > $Q; echo "$v"; } 9>$L )
    [ "$i" -ge "${#JOBS[@]}" ] && break
    IFS='|' read -r mode extra tag <<< "${JOBS[$i]}"
    local name; name=$(tag_for "$mode" "$extra" "$tag")
    local art="$R/${name}_single.json"
    if [ -f "$art" ] && [ "$art" -nt "$MARK" ]; then
      echo "[gpu $gpu] ($((i+1))/${#JOBS[@]}) skip $name"; continue; fi
    local tagarg=""; [ -n "$tag" ] && tagarg="--tag $tag"
    echo "[gpu $gpu] ($((i+1))/${#JOBS[@]}) $name"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u train_single.py --mode "$mode" \
      $extra $tagarg > "$R/sg_$name.log" 2>&1
    rc=$?
    [ $rc -ne 0 ] && echo "[gpu $gpu] FAILED rc=$rc : $name" || echo "[gpu $gpu] ok $name"
  done
}
for g in 0 1 2; do worker $g & done
wait; rm -f $Q $L; echo "SINGLE_SWEEP done"

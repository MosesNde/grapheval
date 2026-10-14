#!/bin/bash
# Repopulate every table under repository-disjoint model selection.
#
# Concurrency is 2, not 4. Each run holds the whole corpus as PyG objects
# (~69,698 graphs x ~165 nodes x 768 floats) plus DataLoader workers; four
# concurrent `random` runs were SIGKILLed by the OOM killer (rc=137) with
# nothing written. DataLoader workers are also reduced from 12 to 6.
#
# Resumable: a run whose artifact is newer than the marker file is skipped, so
# an interruption costs only the runs that were in flight.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MARK=$R/.repopulate_started
[ -f "$MARK" ] || touch "$MARK"
Q=/tmp/repopq.$$; L=$Q.lock; echo 0 > $Q
mapfile -t JOBS < <(grep -v '^#' repopulate_jobs.txt | grep -v '^$')
echo "queued ${#JOBS[@]} runs, marker $(date -r $MARK '+%F %T')"

# reproduce train_cwe's tag so we can tell whether a run already completed
tag_for(){ # mode extra tag
  local m=$1 e=$2 t=$3 out="$m-gat"
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
    local i
    i=$( { flock 9; v=$(cat $Q); echo $((v+1)) > $Q; echo "$v"; } 9>$L )
    [ "$i" -ge "${#JOBS[@]}" ] && break
    IFS='|' read -r mode extra tag <<< "${JOBS[$i]}"
    local name; name=$(tag_for "$mode" "$extra" "$tag")
    local art="$R/${name}_stage1.json"
    if [ -f "$art" ] && [ "$art" -nt "$MARK" ]; then
      echo "[gpu $gpu] ($((i+1))/${#JOBS[@]}) skip $name (already done)"; continue
    fi
    local tagarg=""; [ -n "$tag" ] && tagarg="--tag $tag"
    echo "[gpu $gpu] ($((i+1))/${#JOBS[@]}) $name"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u train_cwe.py --stage 1 --mode "$mode" \
      --workers 6 $extra $tagarg > "$R/rp2_$name.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then echo "[gpu $gpu] FAILED rc=$rc : $name"
    else echo "[gpu $gpu] ok $name"; fi
  done
}
for g in 0 1; do worker $g & done
wait; rm -f $Q $L
echo "REPOPULATE done"

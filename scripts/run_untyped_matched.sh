#!/bin/bash
# Matched typed-vs-untyped control, bidirectional throughout.
#
# WHY: the paper's typed-edge comparison used repo-gat-rev (bidirectional+typed)
# against repo-gat-euntyped (DIRECTED+untyped), so the reported +0.024 mixed the
# relation labels with the edge direction. The matched arm is bidirectional+
# untyped, which did not exist. This runs it for all four backbones so the
# label effect can also be read against each operator's ability to see labels:
#   GAT / GCN(RGCN) / gtrans condition on relation; GIN has no edge term.
#
# Baselines already on disk: repo-{gat,gcn,gin,gtrans}-rev  (5 seeds each)
# Produces:                  repo-{bb}-euntyped-rev[-sN]
#
# 3-way concurrency, 6 workers: 4-way x 12 workers previously OOMed the host.
# GPU 1 excluded: it produced every CUDA launch timeout in the earlier sweeps.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python
R=/home/moses/TSE_Journal/runs
GPUS=(0 2 3)

# GPU is assigned round-robin by job index. Polling memory.used does not work
# here: a job takes minutes to allocate, so consecutive picks all see the same
# idle device and pile onto it.
run_one() {
  local bb=$1 s=$2 G=$3
  local seedarg=""; local sfx=""
  if [ -n "$s" ]; then seedarg="--seed $s"; sfx="-s$s"; fi
  local tag="repo-${bb}-euntyped-rev${sfx}"
  if [ -f "$R/${tag}_stage1.json" ]; then echo "SKIP  $tag (exists)"; return 0; fi
  echo "START $tag -> gpu $G"
  # bidirectional+untyped doubles the edge count, so a single job can reach
  # ~65 GB; one job per GPU, and expandable_segments to limit fragmentation.
  CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u train_cwe.py --stage 1 --mode repo \
      --backbone "$bb" --edges untyped --reverse-edges --workers 6 \
      $seedarg > "$R/untyped_matched_${bb}${sfx}.log" 2>&1
  echo "DONE  $tag exit=$?"
}

JOBS=()
for bb in gat gcn gin gtrans; do
  for s in "" 1 2 3 4; do JOBS+=("$bb:$s"); done
done

i=0
for j in "${JOBS[@]}"; do
  bb=${j%%:*}; s=${j#*:}
  run_one "$bb" "$s" "${GPUS[$((i % 3))]}" &
  i=$((i+1))
  if [ $((i % 3)) -eq 0 ]; then wait; else sleep 20; fi
done
wait
echo "UNTYPED_MATCHED_ALL_DONE"

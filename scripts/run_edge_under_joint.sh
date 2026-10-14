#!/bin/bash
# Completes a 2x2: {frozen, joint} x {edges, no edges}, every cell trained on the
# full training partition with the same loop, schedule and architecture. The
# existing edge-contribution numbers were measured with frozen features only.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
emptiest(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }
while pgrep -u "$(id -u)" -f "joint_finetune.py.*frozenmatched" > /dev/null; do sleep 90; done
echo "frozen-matched arm clear"
# joint, no edges
for s in 42 1 2 3 4; do
  G=$(emptiest); echo "joint noedge s$s -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u joint_finetune.py --mode repo --edges none \
    --epochs 15 --batch-size 16 --accum 8 --seed $s --tag jointnoedge \
    > $R/jne_s$s.log 2>&1
  echo "joint noedge s$s exit=$?"
done
# frozen, no edges
for s in 42 1 2 3 4; do
  G=$(emptiest); echo "frozen noedge s$s -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u joint_finetune.py --mode repo --edges none \
    --epochs 15 --batch-size 16 --accum 8 --freeze-encoder --seed $s \
    --tag frozennoedge > $R/fne_s$s.log 2>&1
  echo "frozen noedge s$s exit=$?"
done
echo "EDGE_UNDER_JOINT done"

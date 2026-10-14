#!/bin/bash
# Which of three differences explains 0.578 -> 0.652?
#   A  100% data, repo-disjoint validation   (already measured: 0.6519)
#   B  100% data, internal validation        -> isolates model selection
#   C   80% data, repo-disjoint validation   -> isolates training-set size
# Schedule is held at the matched setting throughout, so anything left over
# after B and C is attributable to it.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
emptiest(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }
for s in 42 1 2 3 4; do
  G=$(emptiest); echo "B internal-val s$s -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u joint_finetune.py --mode repo --reverse-edges \
    --freeze-encoder --val-source internal --epochs 15 --batch-size 16 --accum 8 \
    --seed $s --tag B_internalval > $R/procB_s$s.log 2>&1
  echo "B s$s exit=$?"
done
for s in 42 1 2 3 4; do
  G=$(emptiest); echo "C 80pct-data s$s -> gpu $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u joint_finetune.py --mode repo --reverse-edges \
    --freeze-encoder --train-frac 0.8 --epochs 15 --batch-size 16 --accum 8 \
    --seed $s --tag C_80pct > $R/procC_s$s.log 2>&1
  echo "C s$s exit=$?"
done
echo "PROCEDURE_DIAGNOSTIC done"

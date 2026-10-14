#!/bin/bash
set -u
cd /home/moses/TSE_Journal/scripts
R=/home/moses/TSE_Journal/runs
PY=/home/moses/venv/bin/python

# 1. GBM probe on the already-cached repo representations (CPU only)
echo "[1/4] gbm+matched probes, repo"
$PY -u analyze_probes_matched.py repo > $R/probes_matched_repo.log 2>&1
echo "  exit=$?"

# 2-4. replicate the whole decomposition on the random split
for g in 0 1 2 3; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$u" -lt 25000 ] && { G=$g; break; }
done
if [ -z "${G:-}" ]; then echo "NO_FREE_GPU"; exit 1; fi
echo "[2/4] mechanism, random -> gpu $G"
CUDA_VISIBLE_DEVICES=$G $PY -u analyze_mechanism.py --mode random --epochs 60 \
  > $R/mechanism_random.log 2>&1
echo "  exit=$?"
echo "[3/4] residual + reps dump, random -> gpu $G"
CUDA_VISIBLE_DEVICES=$G $PY -u analyze_residual.py --mode random --epochs 60 \
  > $R/residual_random.log 2>&1
echo "  exit=$?"
CUDA_VISIBLE_DEVICES=$G $PY -u analyze_residual_robust.py random \
  > $R/residual_robust_random.log 2>&1
echo "  exit=$?"
echo "[4/4] matched probes, random"
$PY -u analyze_probes_matched.py random > $R/probes_matched_random.log 2>&1
echo "  exit=$?"
echo "ALLMECH done"

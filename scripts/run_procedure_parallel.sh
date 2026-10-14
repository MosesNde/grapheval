#!/bin/bash
# Remaining 9 diagnostic runs, one worker per free GPU. B seed 42 is already
# running on GPU 1 from the sequential launch, so it is excluded here.
cd /home/moses/TSE_Journal/scripts
PY=/home/moses/venv/bin/python; R=/home/moses/TSE_Journal/runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

JOBS=(
  "B|1" "B|2" "B|3" "B|4"
  "C|42" "C|1" "C|2" "C|3" "C|4"
)
worker(){                       # $1 = gpu id
  local gpu=$1
  while true; do
    local idx
    idx=$( { flock 9; i=$(cat /tmp/procq_$$ 2>/dev/null || echo 0); echo $((i+1)) > /tmp/procq_$$; echo "$i"; } 9>/tmp/procq_$$.lock )
    [ "$idx" -ge "${#JOBS[@]}" ] && break
    IFS='|' read -r arm s <<< "${JOBS[$idx]}"
    if [ "$arm" = "B" ]; then
      extra="--val-source internal"; tag=B_internalval; log=procB_s$s
    else
      extra="--train-frac 0.8";      tag=C_80pct;       log=procC_s$s
    fi
    echo "[gpu $gpu] start $arm seed $s"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u joint_finetune.py --mode repo --reverse-edges \
      --freeze-encoder $extra --epochs 15 --batch-size 16 --accum 8 \
      --seed $s --tag $tag > $R/$log.log 2>&1
    echo "[gpu $gpu] $arm seed $s exit=$?"
  done
}
echo 0 > /tmp/procq_$$
for g in 0 2 3; do worker $g & done
wait
rm -f /tmp/procq_$$ /tmp/procq_$$.lock
echo "PROCEDURE_PARALLEL done"

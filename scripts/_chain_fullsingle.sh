#!/bin/bash
# Wait for the smoke run, verify it produced a sane artifact, then run the sweep.
cd /home/moses/TSE_Journal
until [ "$(pgrep -c -f 'train_fullsingle[.]py')" -eq 0 ]; do sleep 20; done
echo "smoke finished $(date '+%H:%M')"
tail -3 runs/smoke_fullsingle.log

ART=$(ls runs/repo-gat*_fullsingle.json 2>/dev/null | head -1)
if [ -z "$ART" ]; then echo "ABORT: smoke produced no artifact"; exit 1; fi
OK=$(/home/moses/venv/bin/python -c "
import json,sys
d=json.load(open('$ART'))
r=d.get('single_full_partition',{}).get('roc_auc')
n=d.get('n_train')
print('ok' if (r and 0.4<r<0.95 and n and n>60000) else 'bad', r, n)")
echo "smoke check: $OK"
case "$OK" in ok*) ;; *) echo "ABORT: implausible smoke result"; exit 1;; esac

# the smoke used --epochs 2; remove its artifact so the real run is not skipped
rm -f runs/repo-gat_fullsingle.json runs/repo-gat-s0_fullsingle.json
echo "launching full sweep $(date '+%H:%M')"
bash scripts/run_fullsingle.sh

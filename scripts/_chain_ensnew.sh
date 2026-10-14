#!/bin/bash
cd /home/moses/TSE_Journal
until [ "$(pgrep -c -f 'train_fullsingle[.]py')" -eq 0 ]; do sleep 20; done
echo "fullsingle seed-0 retry finished $(date '+%H:%M')"
bash scripts/run_ensnew.sh

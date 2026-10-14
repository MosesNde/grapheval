#!/bin/bash
cd /home/moses/TSE_Journal
until [ "$(pgrep -c -f 'train_fullsingle[.]py')" -eq 0 ]; do sleep 20; done
echo "=== smoke finished at $(date '+%H:%M') ==="
tail -6 runs/smoke_fullsingle.log

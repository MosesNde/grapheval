#!/bin/bash
# Wait for the in-flight training jobs to write their artifacts, then run the
# (GPU round-robin) matched-untyped sweep, which skips anything already done.
# The bracket in the pattern stops pgrep from matching this script's own cmdline.
cd /home/moses/TSE_Journal
until [ "$(pgrep -c -f 'train_cwe[.]py')" -eq 0 ]; do sleep 30; done
echo "in-flight jobs finished at $(date)"
bash scripts/run_untyped_matched.sh

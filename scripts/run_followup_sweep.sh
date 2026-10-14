#!/bin/bash
# Waits for the current driver to drain, then relaunches on GPUs 0, 2 and 3.
# GPU 1 is excluded: it produced all four CUDA launch-timeout failures while
# running 9C hotter than its neighbours. The resume check skips every run whose
# artifact is newer than the marker, so this picks up only the queue remainder
# plus the four that failed.
cd /home/moses/TSE_Journal/scripts
R=/home/moses/TSE_Journal/runs
while pgrep -u "$(id -u)" -f "run_repopulate_fixed.sh" > /dev/null; do sleep 120; done
echo "first driver drained at $(date '+%F %T')"
echo "  completed before handover: $(grep -c '] ok ' $R/repopulate_driver.log)/80"
echo "  failures to retry: $(grep -c FAILED $R/repopulate_driver.log)"
# keep the same marker so completed runs are still recognised as done
./run_repopulate_healthy.sh
echo "FOLLOWUP_SWEEP done"

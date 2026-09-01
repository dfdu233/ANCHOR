#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
log=corrected_runs/evidence_addressability_gate_v2/logs/llava_restore_watcher_v1.log
mkdir -p "$(dirname "$log")"
echo "$(date -u +%FT%TZ) watcher started" >>"$log"
while tmux has-session -t addressability_stage2_supervisor_v2 2>/dev/null; do
  sleep 20
done
if ! tmux has-session -t baseline_llava_methods_v2 2>/dev/null; then
  tmux new-session -d -s baseline_llava_methods_v2 \
    "cd /home/dbw/ANCHOR && bash scripts/run_baseline_llava_methods_long_queue_v1.sh"
fi
echo "$(date -u +%FT%TZ) baseline_llava_methods_v2 restored" >>"$log"

#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
upstream=${SECOND_RUN_UPSTREAM:-corrected_runs/detached_jobs/second-vqa-rad-oe-v9.json}
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done
if [[ "$status" != "done" ]]; then
  echo "SECOND OE run unavailable; paired evaluation not started" >&2
  exit 2
fi

exec bash scripts/reevaluate_second_vs_canonical_v1.sh

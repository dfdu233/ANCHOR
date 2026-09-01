#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
state=corrected_runs/detached_jobs/llava-mimic-report-full-greedy-v1.json
while true; do
  status=$(/opt/miniconda3/envs/huatuo/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$state")
  [[ "$status" == "done" ]] && break
  if [[ "$status" == "failed" ]]; then
    echo "LLaVA full generation failed" >&2
    exit 1
  fi
  sleep 30
done

export PYTHONPATH=anchor
exec /opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.run_oe_sanity_audit \
  --analyze-existing corrected_runs/unified_eval/full/llava_mimic_report_greedy_v1/predictions.jsonl \
  --output-dir corrected_runs/unified_eval/full/llava_mimic_report_greedy_v1/sanity_audit

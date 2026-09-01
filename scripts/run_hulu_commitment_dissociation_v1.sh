#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
upstream=corrected_runs/detached_jobs/hulu-report-dependency-audit-v4.json
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done
if [[ "$status" != "done" ]]; then
  echo "Hulu v4 audit did not complete; dissociation analysis cannot run" >&2
  exit 2
fi

export PYTHONPATH=anchor
/opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.medeval.analyze_commitment_dissociation \
  --raw corrected_runs/unified_eval/sanity/hulu_mimic_report_dependency_v4/generation.raw.jsonl \
  --config configs/unified_eval/hulu_commitment_dissociation_v1.json \
  --output corrected_runs/unified_eval/sanity/hulu_mimic_report_dependency_v4/commitment_dissociation.json

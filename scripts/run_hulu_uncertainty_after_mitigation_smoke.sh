#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
state=corrected_runs/detached_jobs/mitigation-smoke-after-llava-full-v1.json
while true; do
  status=$(/opt/miniconda3/envs/huatuo/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$state")
  [[ "$status" == "done" ]] && break
  if [[ "$status" == "failed" ]]; then
    echo "mitigation smoke supervisor failed" >&2
    exit 1
  fi
  sleep 30
done

export PYTHONPATH=anchor
exec /home/dbw/.venvs/hulumed/bin/python \
  -m corrected_sgta.run_claim_universe_scoring \
  --model hulu \
  --model-path /home/dbw/models/Hulu-Med-4B \
  --questions corrected_runs/missing_third_state/mimic_report_triplets_v1/questions.json \
  --image-root data/medheval/images \
  --output-dir corrected_runs/missing_third_state/mimic_report_triplets_v1/hulu_scores_v1 \
  --skip-null \
  --generate-draft \
  --draft-answer-type ternary

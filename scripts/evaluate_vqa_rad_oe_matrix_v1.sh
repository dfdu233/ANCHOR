#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/vqa-rad-oe-evaluation.lock
if ! flock -n 9; then
  echo "Another VQA-RAD OE evaluator owns the run lock; exiting cleanly"
  exit 0
fi

upstream=corrected_runs/detached_jobs/vqa-rad-oe-baselines-v5.json
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done

export PYTHONPATH=anchor
/opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.medeval.evaluate_oe_matrix \
  --manifest corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json \
  --run-root corrected_runs/unified_eval/full/vqa_rad_oe_mitigation_v3/vqa_rad/official_test_oe/open_vqa \
  --selection corrected_runs/unified_eval/smoke/vqa_rad_oe_mitigation_v3/selection.json \
  --output-root corrected_runs/unified_eval/full/vqa_rad_oe_mitigation_v3/evaluation \
  --bootstrap-replicates 5000 \
  --seed 42

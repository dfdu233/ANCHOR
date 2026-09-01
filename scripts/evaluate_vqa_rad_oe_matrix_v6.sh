#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
upstream=${MATRIX_UPSTREAM:-corrected_runs/detached_jobs/vqa-rad-oe-baselines-v6.json}
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done
if [[ "$status" != "done" ]]; then
  echo "Corrected VQA-RAD mitigation matrix unavailable" >&2
  exit 2
fi

export PYTHONPATH=anchor
/opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.medeval.evaluate_oe_matrix \
  --manifest corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json \
  --run-root corrected_runs/unified_eval/full/vqa_rad_oe_mitigation_v4/vqa_rad/official_test_oe/open_vqa \
  --selection corrected_runs/unified_eval/smoke/vqa_rad_oe_mitigation_v4/selection.json \
  --output-root corrected_runs/unified_eval/full/vqa_rad_oe_mitigation_v4/evaluation \
  --bootstrap-replicates 5000 \
  --seed 42

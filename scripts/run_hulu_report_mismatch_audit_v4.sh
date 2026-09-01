#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/hulu-report-mismatch-audit-v4.lock
if ! flock -n 9; then
  echo "Another Hulu mismatched-image v4 audit owns the run lock; experiment not started"
  exit 75
fi

upstream=corrected_runs/detached_jobs/vqa-rad-oe-evaluation-v5.json
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done
if [[ "$status" != "done" ]]; then
  echo "VQA-RAD evaluation failed; GPU serialization predecessor did not complete" >&2
  exit 2
fi

export PYTHONPATH=anchor
/home/dbw/.venvs/hulumed/bin/python \
  -m corrected_sgta.run_oe_sanity_audit \
  --run-generation \
  --model hulu \
  --manifest corrected_runs/high_efficiency/full_generation_mmedrag_mimic_report_20260726/mmedrag/mimic/report_generation/greedy/chunk_0000.questions.json \
  --image-root data/medheval/images \
  --output-dir corrected_runs/unified_eval/sanity/hulu_mimic_report_dependency_v4 \
  --max-samples 16 \
  --max-new-tokens 160 \
  --conv-mode native \
  --prompt-mode mmedrag structured abnormality_focused \
  --view real null shuffled pixel_shuffled \
  --seed 20260801

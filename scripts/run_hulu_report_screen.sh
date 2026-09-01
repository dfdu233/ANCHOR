#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
export PYTHONPATH=anchor
exec /home/dbw/.venvs/hulumed/bin/python \
  -m corrected_sgta.run_claim_universe_scoring \
  --model hulu \
  --model-path /home/dbw/models/Hulu-Med-4B \
  --questions corrected_runs/claim_transport/mimic_report_grade_c_v1/score_questions.json \
  --image-root data/medheval/images \
  --output-dir corrected_runs/claim_transport/mimic_report_grade_c_v1/hulu_raw_scores_v1 \
  --skip-null \
  --resume

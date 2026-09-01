#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
PYTHONPATH=. python -c \
  'from scripts.research_status import print_detached_jobs; print_detached_jobs()'
echo
echo "GPU:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
echo
echo "Hulu report screen:"
raw=corrected_runs/claim_transport/mimic_report_grade_c_v1/hulu_raw_scores_v1/raw.jsonl
summary=corrected_runs/claim_transport/mimic_report_grade_c_v1/hulu_raw_scores_v1/summary.json
if [[ -f "$raw" ]]; then
  wc -l "$raw"
else
  echo "  not started"
fi
if [[ -f "$summary" ]]; then
  sed -n '1,80p' "$summary"
fi

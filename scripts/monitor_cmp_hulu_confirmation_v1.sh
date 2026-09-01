#!/usr/bin/env bash
set -euo pipefail
cd /home/dbw/ANCHOR
plain=corrected_runs/paper_baselines_v1/full_matrix_v1/shared_rag_generation/hulu/cxr_vishal/no_context
rag=corrected_runs/paper_baselines_v1/full_matrix_v1/shared_rag_generation/hulu/cxr_vishal/rag
log=corrected_runs/detached_jobs/logs/cmp-hulu-confirmation-v1.log
mkdir -p "$(dirname "$log")"
while true; do
  if [[ -f "$plain/answers.jsonl" && -f "$plain/evaluation_ce_v7.json" \
     && -f "$rag/answers.jsonl" && -f "$rag/evaluation_ce_v7.json" ]]; then
    PYTHONPATH=. /home/dbw/.venvs/hulumed/bin/python \
      -m anchor.corrected_sgta.analyze_intervention_tomography_v1 >>"$log" 2>&1
    date -u +'%Y-%m-%dT%H:%M:%SZ complete' >>"$log"
    exit 0
  fi
  date -u +'%Y-%m-%dT%H:%M:%SZ waiting_for_hulu_cxr_pair' >>"$log"
  sleep 60
done

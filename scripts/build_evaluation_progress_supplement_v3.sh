#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.build_evaluation_progress_supplement_v3 \
  --base-audit corrected_runs/paper/iclr_oral_completion_audit_v2/audit.json \
  --rag-audit corrected_runs/unified_eval/provenance/rag_dual_track_qualification_v2.json \
  --v1-failure corrected_runs/unified_eval/provenance/internal_controls_t3_v1_huatuo_cap_failure_prereg_v1.json \
  --v2-execution configs/unified_eval/internal_control_t3_execution_v2.json \
  --v2-provenance corrected_runs/unified_eval/inputs/vqa_rad_internal_control_t3_n120_v2.provenance.json \
  --v1-job-state corrected_runs/detached_jobs/internal-controls-t3-n120-v1.json \
  --v2-job-state corrected_runs/detached_jobs/internal-controls-t3-v2-repair-continuation-v1.json \
  --output corrected_runs/paper/iclr_oral_completion_audit_v2/evaluation_progress_supplement_v3.json

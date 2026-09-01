#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.build_evaluation_progress_supplement_v4 \
  --base-audit corrected_runs/paper/iclr_oral_completion_audit_v3/audit.json \
  --rag-audit corrected_runs/unified_eval/provenance/rag_dual_track_qualification_v2.json \
  --baseline-audit corrected_runs/unified_eval/provenance/baseline_coverage_audit_v5.json \
  --internal-generation corrected_runs/unified_eval/full/internal_controls_t3_v2/generation_audit.json \
  --internal-form corrected_runs/unified_eval/full/internal_controls_t3_v2/generation_form_audit_v1.json \
  --internal-postprocess corrected_runs/unified_eval/full/internal_controls_t3_v2/postprocess_summary.json \
  --internal-huatuo-archive corrected_runs/unified_eval/physician_review/internal_controls_t3_v2/huatuo/archives_v1/verification.json \
  --internal-hulu-archive corrected_runs/unified_eval/physician_review/internal_controls_t3_v2/hulu/archives_v1/verification.json \
  --internal-huatuo-monitor corrected_runs/unified_eval/physician_review/internal_controls_t3_v2/huatuo/clinical_returns_v1/monitor.heartbeat.json \
  --internal-hulu-monitor corrected_runs/unified_eval/physician_review/internal_controls_t3_v2/hulu/clinical_returns_v1/monitor.heartbeat.json \
  --llava-generation corrected_runs/unified_eval/full/vqa_rad_oe_mitigation_t3_n120_v1/generation_audit.json \
  --llava-postprocess corrected_runs/unified_eval/physician_review/vqa_rad_mitigation_t3_n120_v1/postprocess_summary.json \
  --llava-archive corrected_runs/unified_eval/physician_review/vqa_rad_mitigation_t3_n120_v1/archives_v1/verification.json \
  --llava-monitor corrected_runs/unified_eval/physician_review/vqa_rad_mitigation_t3_n120_v1/clinical_returns_v1/monitor.heartbeat.json \
  --system-handoff configs/cecd_system_pih_native_eager_canary_handoff_v3.json \
  --system-huatuo-canary corrected_runs/vindr_v2/system_pih_control_preflight_v1/canaries/huatuo_native_eager_canary_v3_full_query.json \
  --system-hulu-canary corrected_runs/vindr_v2/system_pih_control_preflight_v1/canaries/hulu_native_eager_canary_v3_query_chunked.json \
  --system-job-state corrected_runs/detached_jobs/cecd-system-pih-native-eager-canaries-v3-family-specific-query-shape.json \
  --output corrected_runs/paper/iclr_oral_completion_audit_v3/evaluation_progress_supplement_v4.json

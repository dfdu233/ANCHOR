#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
registry=corrected_runs/unified_eval/provenance/artifact_registry_v1.jsonl
artifact=corrected_runs/unified_eval/smoke/internal_controls_t2_v1/t2_n32_v1/calibrated_abstention_t2_qualification_v2.json

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.artifact_registry \
  --registry "$registry" \
  --artifact "$artifact" \
  --status failed_cutoff \
  --evaluator-version calibrated-abstention-fit-t2-v2 \
  --evidence-scope 'internal control qualification; calibrated_abstention; T2' \
  --reason 'execution completed, but Huatuo development proxy had zero positive rows and produced no selective action; outcome-driven threshold retuning is prohibited'

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_method_evidence_ladder \
  --t0-audit corrected_runs/unified_eval/provenance/method_ladder_t0_v3.json \
  --registry "$registry" \
  --identity-gate corrected_runs/unified_eval/sanity/post_restart_runtime_identity_v1/identity.json \
  --mitigation-identity-gate corrected_runs/unified_eval/smoke/vqa_rad_oe_mitigation_v5_t2_256/greedy256_backend_conformance.json \
  --rag-causal-summary corrected_runs/unified_eval/rag/common_protocol_v1/visual_ce_ladder_v3_causal_controls_v2.json \
  --output corrected_runs/unified_eval/provenance/method_evidence_ladder_v9.json

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_internal_baseline_controls_v1 \
  --contract configs/unified_eval/internal_baseline_control_contract_v1.json \
  --method-evidence corrected_runs/unified_eval/provenance/method_evidence_ladder_v9.json \
  --registry "$registry" \
  --output corrected_runs/unified_eval/provenance/internal_baseline_control_qualification_v4.json

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_baseline_coverage_v3 \
  --config configs/unified_eval/method_ladder_v1.json \
  --t0 corrected_runs/unified_eval/provenance/method_ladder_t0_v3.json \
  --evidence corrected_runs/unified_eval/provenance/method_evidence_ladder_v9.json \
  --registry "$registry" \
  --native-acceptance corrected_runs/unified_eval/full/native_oe_greedy256_acceptance_v1.json \
  --rag-causal corrected_runs/unified_eval/rag/common_protocol_v1/visual_ce_ladder_v3_causal_controls_v2.json \
  --internal-control-qualification corrected_runs/unified_eval/provenance/internal_baseline_control_qualification_v4.json \
  --report-audit hulu=corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v2/sanity_audit/summary.json \
  --report-audit llava=corrected_runs/unified_eval/full/llava_mimic_report_greedy_v1/sanity_audit/summary.json \
  --output corrected_runs/unified_eval/provenance/baseline_coverage_audit_v5.json

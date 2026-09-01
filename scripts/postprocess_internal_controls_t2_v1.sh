#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
root=corrected_runs/unified_eval/smoke/internal_controls_t2_v1/t2_n32_v1

# Revalidate the completed matrix from immutable inputs before any method
# artifact is allowed to consume it. Further claim/calibration stages are
# appended here only after their CPU smoke gates pass.
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_internal_control_generation_t2_v1 \
  --run-root "$root" \
  --pilot-manifest corrected_runs/unified_eval/inputs/vqa_rad_internal_control_t2_n32_v1.json \
  --freeze-provenance corrected_runs/unified_eval/inputs/vqa_rad_internal_control_t2_n32_v1.provenance.json \
  --execution-contract configs/unified_eval/internal_control_t2_execution_v1.json \
  --limit 32 \
  --output "$root/generation_audit.postprocess.json"

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.prepare_control_claim_extraction_v1 \
  --run-root "$root" \
  --generation-audit "$root/generation_audit.postprocess.json" \
  --aggregation-contract configs/unified_eval/claim_self_consistency_aggregation_v1.json \
  --output "$root/claim_extraction_input.jsonl" \
  --manifest "$root/claim_extraction_input.manifest.json"

exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
flock 8
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:third_party/baselines/radgraph \
  /home/dbw/.venvs/hulumed/bin/python \
  -m anchor.medeval.radgraph_surface_claims_v1 \
  --input "$root/claim_extraction_input.jsonl" \
  --output "$root/surface_claim_extraction.json" \
  --model-cache-dir /home/dbw/model_cache/report_metrics/radgraph \
  --tokenizer-cache-dir /home/dbw/model_cache/report_metrics/modernbert-base \
  --cuda 0
flock -u 8

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.aggregate_claim_self_consistency_v1 \
  --extraction "$root/surface_claim_extraction.json" \
  --extraction-manifest "$root/claim_extraction_input.manifest.json" \
  --aggregation-contract configs/unified_eval/claim_self_consistency_aggregation_v1.json \
  --freeze-provenance corrected_runs/unified_eval/inputs/vqa_rad_internal_control_t2_n32_v1.provenance.json \
  --output "$root/self_consistency_aggregation.json" \
  --selected-answers "$root/self_consistency_selected.answers.jsonl"

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.build_internal_control_t2_qualification_v1 \
  --generation-audit "$root/generation_audit.postprocess.json" \
  --aggregation "$root/self_consistency_aggregation.json" \
  --freeze-provenance corrected_runs/unified_eval/inputs/vqa_rad_internal_control_t2_n32_v1.provenance.json \
  --execution-contract configs/unified_eval/internal_control_t2_execution_v1.json \
  --aggregation-contract configs/unified_eval/claim_self_consistency_aggregation_v1.json \
  --qualification-contract configs/unified_eval/internal_baseline_control_contract_v1.json \
  --temperature-output "$root/temperature_length_t2_qualification.json" \
  --self-consistency-output "$root/self_consistency_t2_qualification.json"

registry=corrected_runs/unified_eval/provenance/artifact_registry_v1.jsonl
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.artifact_registry \
  --registry "$registry" \
  --artifact "$root/temperature_length_t2_qualification.json" \
  --status admissible \
  --evaluator-version internal-control-t2-qualification-builder-v1 \
  --evidence-scope 'internal control qualification; temperature_length_controls; T2' \
  --reason 'frozen image-disjoint development grid, exact traces, non-degenerate sampling, and native length arms passed'
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.artifact_registry \
  --registry "$registry" \
  --artifact "$root/self_consistency_t2_qualification.json" \
  --status admissible \
  --evaluator-version internal-control-t2-qualification-builder-v1 \
  --evidence-scope 'internal control qualification; self_consistency; T2' \
  --reason 'five-seed structured-claim aggregation, deterministic replay, and no exact-text vote passed'

if PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.fit_calibrated_abstention_t2_v1 \
  --extraction "$root/surface_claim_extraction.json" \
  --pilot-manifest corrected_runs/unified_eval/inputs/vqa_rad_internal_control_t2_n32_v1.json \
  --freeze-provenance corrected_runs/unified_eval/inputs/vqa_rad_internal_control_t2_n32_v1.provenance.json \
  --execution-contract configs/unified_eval/calibrated_abstention_t2_execution_v2.json \
  --qualification-contract configs/unified_eval/internal_baseline_control_contract_v1.json \
  --threshold-output "$root/calibrated_abstention_thresholds_v2.json" \
  --qualification-output "$root/calibrated_abstention_t2_qualification_v2.json"; then
  PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.artifact_registry \
    --registry "$registry" \
    --artifact "$root/calibrated_abstention_t2_qualification_v2.json" \
    --status admissible \
    --evaluator-version calibrated-abstention-fit-t2-v2 \
    --evidence-scope 'internal control qualification; calibrated_abstention; T2' \
    --reason 'disjoint 16/16 development calibration, non-degenerate coverage, and claim-selective accounting passed'
else
  printf '%s\n' 'calibrated abstention T2 failed its frozen non-degeneracy/action gate; registering an explicit failed_cutoff without retuning'
  PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.artifact_registry \
    --registry "$registry" \
    --artifact "$root/calibrated_abstention_t2_qualification_v2.json" \
    --status failed_cutoff \
    --evaluator-version calibrated-abstention-fit-t2-v2 \
    --evidence-scope 'internal control qualification; calibrated_abstention; T2' \
    --reason 'execution completed, but Huatuo development proxy had zero positive rows and produced no selective action; threshold retuning is prohibited'
fi

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_method_evidence_ladder \
  --t0-audit corrected_runs/unified_eval/provenance/method_ladder_t0_v3.json \
  --registry "$registry" \
  --identity-gate corrected_runs/unified_eval/sanity/post_restart_runtime_identity_v1/identity.json \
  --mitigation-identity-gate corrected_runs/unified_eval/smoke/vqa_rad_oe_mitigation_v5_t2_256/greedy256_backend_conformance.json \
  --rag-causal-summary corrected_runs/unified_eval/rag/common_protocol_v1/visual_ce_ladder_v3_causal_controls_v2.json \
  --output corrected_runs/unified_eval/provenance/method_evidence_ladder_v7.json

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_internal_baseline_controls_v1 \
  --contract configs/unified_eval/internal_baseline_control_contract_v1.json \
  --method-evidence corrected_runs/unified_eval/provenance/method_evidence_ladder_v7.json \
  --registry "$registry" \
  --output corrected_runs/unified_eval/provenance/internal_baseline_control_qualification_v2.json

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_baseline_coverage_v3 \
  --config configs/unified_eval/method_ladder_v1.json \
  --t0 corrected_runs/unified_eval/provenance/method_ladder_t0_v3.json \
  --evidence corrected_runs/unified_eval/provenance/method_evidence_ladder_v7.json \
  --registry "$registry" \
  --native-acceptance corrected_runs/unified_eval/full/native_oe_greedy256_acceptance_v1.json \
  --rag-causal corrected_runs/unified_eval/rag/common_protocol_v1/visual_ce_ladder_v3_causal_controls_v2.json \
  --internal-control-qualification corrected_runs/unified_eval/provenance/internal_baseline_control_qualification_v2.json \
  --report-audit hulu=corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v2/sanity_audit/summary.json \
  --report-audit llava=corrected_runs/unified_eval/full/llava_mimic_report_greedy_v1/sanity_audit/summary.json \
  --output corrected_runs/unified_eval/provenance/baseline_coverage_audit_v3.json

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.build_evaluation_progress_supplement_v1 \
  --base-audit corrected_runs/paper/iclr_oral_completion_audit_v1/audit.json \
  --internal-audit corrected_runs/unified_eval/provenance/internal_baseline_control_qualification_v2.json \
  --baseline-audit corrected_runs/unified_eval/provenance/baseline_coverage_audit_v3.json \
  --output corrected_runs/paper/iclr_oral_completion_audit_v1/evaluation_progress_supplement_v1.json

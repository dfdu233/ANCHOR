#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
root=corrected_runs/unified_eval/full/internal_controls_t3_v2
manifest=corrected_runs/unified_eval/inputs/vqa_rad_internal_control_t3_n120_v2.json
provenance=corrected_runs/unified_eval/inputs/vqa_rad_internal_control_t3_n120_v2.provenance.json
aggregation=configs/unified_eval/claim_self_consistency_aggregation_v1.json
clinical_contract=configs/unified_eval/internal_control_t3_clinical_analysis_v2.json
form_audit="$root/generation_form_audit_v1.json"

test -f "$root/generation_audit.json"
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_internal_control_generation_form_v1 \
  --run-root "$root" \
  --manifest "$manifest" \
  --execution-contract configs/unified_eval/internal_control_t3_execution_v2.json \
  --generation-audit "$root/generation_audit.json" \
  --output "$form_audit"
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.prepare_control_claim_extraction_v1 \
  --run-root "$root" \
  --generation-audit "$root/generation_audit.json" \
  --aggregation-contract "$aggregation" \
  --output "$root/claim_extraction_input.jsonl" \
  --manifest "$root/claim_extraction_input.manifest.json"

exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
flock 8
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:third_party/baselines/radgraph \
  /home/dbw/.venvs/hulumed/bin/python -m anchor.medeval.radgraph_surface_claims_v1 \
  --input "$root/claim_extraction_input.jsonl" \
  --output "$root/surface_claim_extraction.json" \
  --model-cache-dir /home/dbw/model_cache/report_metrics/radgraph \
  --tokenizer-cache-dir /home/dbw/model_cache/report_metrics/modernbert-base \
  --cuda 0
flock -u 8

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.aggregate_claim_self_consistency_v1 \
  --extraction "$root/surface_claim_extraction.json" \
  --extraction-manifest "$root/claim_extraction_input.manifest.json" \
  --aggregation-contract "$aggregation" \
  --freeze-provenance "$provenance" \
  --output "$root/self_consistency_aggregation.json" \
  --selected-answers "$root/self_consistency_selected.answers.jsonl"

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.split_answers_by_model_v1 \
  --source "$root/self_consistency_selected.answers.jsonl" \
  --output-dir "$root/self_consistency_by_model"

for model in huatuo hulu; do
  review=corrected_runs/unified_eval/physician_review/internal_controls_t3_v2/$model
  PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.prepare_physician_oe_review \
    --manifest "$manifest" \
    --image-root /home/dbw/datasets/public/vqa_rad_hf/test_images \
    --answer "greedy512=$root/$model/greedy512/answers.jsonl" \
    --answer "sample_t02_p09_seed42=$root/$model/sample_t02_p09_seed42/answers.jsonl" \
    --answer "sample_t07_p09_seed42=$root/$model/sample_t07_p09_seed42/answers.jsonl" \
    --answer "sample_t10_p09_seed42=$root/$model/sample_t10_p09_seed42/answers.jsonl" \
    --answer "structured_self_consistency=$root/self_consistency_by_model/$model.answers.jsonl" \
    --bundle "$review/review.template.jsonl" \
    --mapping "$review/review.private_mapping.jsonl" \
    --metadata "$review/review.metadata.json" \
    --n-qids 32 \
    --seed 20260803 \
    --deduplicate-exact-answers
  PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.build_physician_oe_deliveries \
    --template "$review/review.template.jsonl" \
    --metadata "$review/review.metadata.json" \
    --output-dir "$review/deliveries_v1" \
    --calibration-groups 8 \
    --double-review-groups 32
  PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.build_physician_oe_prereg_v1 \
    --template "$review/review.template.jsonl" \
    --mapping "$review/review.private_mapping.jsonl" \
    --delivery "$review/deliveries_v1/delivery_manifest.json" \
    --contract "$clinical_contract" \
    --baseline greedy512 \
    --candidate sample_t02_p09_seed42 \
    --candidate sample_t07_p09_seed42 \
    --candidate sample_t10_p09_seed42 \
    --candidate structured_self_consistency \
    --output "$review/clinical_analysis_prereg_v1.json"
  PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.package_physician_oe_deliveries \
    --delivery-dir "$review/deliveries_v1" \
    --metadata "$review/review.metadata.json" \
    --runbook docs/PHYSICIAN_OE_REVIEW_RUNBOOK.md \
    --output-dir "$review/archives_v1"
  PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.verify_physician_oe_delivery_archives \
    --delivery-dir "$review/archives_v1" \
    --output "$review/archives_v1/verification.json"
done

PYTHONPATH=. .venv-full/bin/python - <<'PY'
import json
from pathlib import Path
from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.store import atomic_write_json

root = Path('corrected_runs/unified_eval/full/internal_controls_t3_v2')
record = {
    'protocol_version': 'internal-controls-t3-postprocess-v2',
    'generation_audit_sha256': sha256_file(root / 'generation_audit.json'),
    'generation_form_audit_sha256': sha256_file(root / 'generation_form_audit_v1.json'),
    'claim_extraction_sha256': sha256_file(root / 'surface_claim_extraction.json'),
    'self_consistency_sha256': sha256_file(root / 'self_consistency_aggregation.json'),
    'execution_contract_sha256': sha256_file(Path('configs/unified_eval/internal_control_t3_execution_v2.json')),
    'clinical_analysis_contract_sha256': sha256_file(Path('configs/unified_eval/internal_control_t3_clinical_analysis_v2.json')),
    'v1_disposition': 'identity_and_length_stress_only',
    'physician_labels_present': False,
    'clinical_efficacy_authorized': False,
    'review_packs': {
        model: {
            'metadata_sha256': sha256_file(Path(f'corrected_runs/unified_eval/physician_review/internal_controls_t3_v2/{model}/review.metadata.json')),
            'delivery_index_sha256': sha256_file(Path(f'corrected_runs/unified_eval/physician_review/internal_controls_t3_v2/{model}/archives_v1/delivery_index.json')),
            'archive_verification_sha256': sha256_file(Path(f'corrected_runs/unified_eval/physician_review/internal_controls_t3_v2/{model}/archives_v1/verification.json')),
        }
        for model in ('huatuo', 'hulu')
    },
}
record['fingerprint'] = sha256_json(record)
atomic_write_json(root / 'postprocess_summary.json', record)
print(json.dumps(record, indent=2))
PY

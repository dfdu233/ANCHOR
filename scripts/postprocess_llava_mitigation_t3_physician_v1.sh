#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
run_root=corrected_runs/unified_eval/full/vqa_rad_oe_mitigation_t3_n120_v1
base=corrected_runs/unified_eval/physician_review/vqa_rad_mitigation_t3_n120_v1
selected_manifest=$base/selected_manifest.private.json
selection_prereg=$base/selection_prereg.json
execution_contract=configs/unified_eval/llava_mitigation_t3_execution_v1.json
clinical_contract=configs/unified_eval/llava_mitigation_t3_clinical_analysis_v1.json
selected_answers=$base/selected_answers

test -f "$run_root/generation_audit.json"
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.prepare_llava_mitigation_t3_physician_inputs_v1 \
  --run-root "$run_root" \
  --generation-audit "$run_root/generation_audit.json" \
  --execution-contract "$execution_contract" \
  --selection-prereg "$selection_prereg" \
  --selected-manifest "$selected_manifest" \
  --output-dir "$selected_answers" \
  --provenance "$base/selected_answers.provenance.json"

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.prepare_physician_oe_review \
  --manifest "$selected_manifest" \
  --image-root /home/dbw/datasets/public/vqa_rad_hf/test_images \
  --answer "greedy=$selected_answers/greedy.answers.jsonl" \
  --answer "beam=$selected_answers/beam.answers.jsonl" \
  --answer "VCD=$selected_answers/VCD.answers.jsonl" \
  --answer "opera=$selected_answers/opera.answers.jsonl" \
  --answer "PAI=$selected_answers/PAI.answers.jsonl" \
  --answer "avisc=$selected_answers/avisc.answers.jsonl" \
  --answer "VISTA_off=$selected_answers/VISTA_off.answers.jsonl" \
  --answer "VISTA_VSV=$selected_answers/VISTA_VSV.answers.jsonl" \
  --answer "VISTA_SLA=$selected_answers/VISTA_SLA.answers.jsonl" \
  --answer "VISTA=$selected_answers/VISTA.answers.jsonl" \
  --bundle "$base/review.template.jsonl" \
  --mapping "$base/review.private_mapping.jsonl" \
  --metadata "$base/review.metadata.json" \
  --n-qids 32 \
  --seed 20260803 \
  --deduplicate-exact-answers

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.build_physician_oe_deliveries \
  --template "$base/review.template.jsonl" \
  --metadata "$base/review.metadata.json" \
  --output-dir "$base/deliveries_v1" \
  --calibration-groups 8 \
  --double-review-groups 32

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.build_physician_oe_prereg_v1 \
  --template "$base/review.template.jsonl" \
  --mapping "$base/review.private_mapping.jsonl" \
  --delivery "$base/deliveries_v1/delivery_manifest.json" \
  --contract "$clinical_contract" \
  --baseline greedy \
  --candidate beam \
  --candidate VCD \
  --candidate opera \
  --candidate PAI \
  --candidate avisc \
  --candidate VISTA_off \
  --candidate VISTA_VSV \
  --candidate VISTA_SLA \
  --candidate VISTA \
  --output "$base/clinical_analysis_prereg_v1.json"

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.package_physician_oe_deliveries \
  --delivery-dir "$base/deliveries_v1" \
  --metadata "$base/review.metadata.json" \
  --runbook docs/PHYSICIAN_OE_REVIEW_RUNBOOK.md \
  --output-dir "$base/archives_v1"
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.verify_physician_oe_delivery_archives \
  --delivery-dir "$base/archives_v1" \
  --output "$base/archives_v1/verification.json"

PYTHONPATH=. .venv-full/bin/python - <<'PY'
import json
from pathlib import Path
from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.store import atomic_write_json

base = Path('corrected_runs/unified_eval/physician_review/vqa_rad_mitigation_t3_n120_v1')
metadata = json.loads((base / 'review.metadata.json').read_text())
record = {
    'protocol_version': 'llava-mitigation-t3-physician-postprocess-v1',
    'generation_operationally_qualified': True,
    'clinical_labels_present': False,
    'clinical_efficacy_authorized': False,
    'selected_groups': 32,
    'planned_model_assignments': 320,
    'unique_review_answer_units_after_exact_deduplication': metadata['n_answer_units'],
    'generation_audit_sha256': sha256_file(Path('corrected_runs/unified_eval/full/vqa_rad_oe_mitigation_t3_n120_v1/generation_audit.json')),
    'selected_answers_provenance_sha256': sha256_file(base / 'selected_answers.provenance.json'),
    'selection_prereg_sha256': sha256_file(base / 'selection_prereg.json'),
    'clinical_analysis_prereg_sha256': sha256_file(base / 'clinical_analysis_prereg_v1.json'),
    'delivery_index_sha256': sha256_file(base / 'archives_v1/delivery_index.json'),
    'archive_verification_sha256': sha256_file(base / 'archives_v1/verification.json'),
    'required_next_step': 'two independent physicians complete blinded A/B archives; monitor validates and advances without synthesizing labels',
}
record['fingerprint'] = sha256_json(record)
target = base / 'postprocess_summary.json'
if target.exists() and json.loads(target.read_text()) != record:
    raise FileExistsError(f'write-once postprocess collision: {target}')
if not target.exists():
    atomic_write_json(target, record)
print(json.dumps(record, indent=2))
PY

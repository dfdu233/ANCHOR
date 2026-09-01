#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
root=corrected_runs/unified_eval/full/internal_controls_t3_v1
qualification=$root/oe_generation_qualification.json

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_oe_generation_qualification_v1 \
  --run-root "$root" \
  --model huatuo \
  --model hulu \
  --arm greedy256 \
  --arm sample_t02_p09_seed42 \
  --arm sample_t07_p09_seed42 \
  --arm sample_t10_p09_seed42 \
  --expected-rows 120 \
  --output "$qualification"

PYTHONPATH=. .venv-full/bin/python - <<'PY'
import json
from pathlib import Path
from anchor.medeval.hashing import sha256_file, sha256_json
from anchor.medeval.store import atomic_write_json

root = Path('corrected_runs/unified_eval/full/internal_controls_t3_v1')
qualification = json.loads((root / 'oe_generation_qualification.json').read_text())
record = {
    'protocol_version': 'internal-controls-t3-v1-disposition-v1',
    'generation_audit_sha256': sha256_file(root / 'generation_audit.json'),
    'oe_generation_qualification_sha256': sha256_file(root / 'oe_generation_qualification.json'),
    'all_clinical_arms_eligible': qualification['all_eligible'],
    'physician_pack_authorized': False,
    'clinical_efficacy_authorized': False,
    'admissible_scope': 'identity_and_length_stress_only',
    'superseded_by': 'internal-control-t3-execution-v2',
    'supersession_basis_sha256': sha256_file(Path('corrected_runs/unified_eval/provenance/internal_controls_t3_v1_huatuo_cap_failure_prereg_v1.json')),
    'v1_clinical_labels_inspected': False,
}
record['fingerprint'] = sha256_json(record)
atomic_write_json(root / 'disposition.json', record)
print(json.dumps(record, indent=2))
PY

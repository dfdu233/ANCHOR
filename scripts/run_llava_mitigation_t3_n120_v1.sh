#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=anchor
export ANCHOR_MODEL_PATH=/home/dbw/models/LLaVA-Med-v1.5-mistral-7b
export ANCHOR_PYTHON=/opt/miniconda3/envs/huatuo/bin/python

contract=configs/unified_eval/llava_mitigation_t3_execution_v1.json
manifest=corrected_runs/unified_eval/inputs/vqa_rad_mitigation_t3_n120_v1.redacted.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
root=corrected_runs/unified_eval/full/vqa_rad_oe_mitigation_t3_n120_v1
mkdir -p "$root" corrected_runs/detached_jobs/locks

verify_frozen_sources() {
PYTHONPATH=. .venv-full/bin/python - "$contract" <<'PY'
import json,sys
from pathlib import Path
from anchor.medeval.hashing import sha256_file
root=Path('/home/dbw/ANCHOR')
contract=json.loads(Path(sys.argv[1]).read_text())
for name,binding in contract['source_bindings'].items():
    path=Path(binding['path'])
    path=path if path.is_absolute() else root/path
    if not path.is_file() or sha256_file(path) != binding['sha256']:
        raise SystemExit(f'frozen source binding failed: {name}: {path}')
PY
}

# The first check fails quickly when a queued job is already stale. The second
# is mandatory because this process may wait for the shared GPU lock for hours.
verify_frozen_sources

exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
flock 8
verify_frozen_sources
/opt/miniconda3/envs/huatuo/bin/python -u \
  -m corrected_sgta.run_llava_med_generation_matrix \
  --question-file "$manifest" \
  --image-folder "$images" \
  --out "$root" \
  --source vqa_rad \
  --dataset official_test_oe_image_disjoint_n120 \
  --task open_vqa \
  --methods greedy beam VCD opera PAI avisc VISTA_off VISTA_VSV VISTA_SLA VISTA \
  --chunk-size 120 \
  --limit 120 \
  --max-new-tokens 512 \
  --conv-mode mistral_instruct \
  --seed 42 \
  --disable-keyword-stopping \
  --continue-on-error
flock -u 8

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_llava_mitigation_t3_generation_v1 \
  --run-root "$root" \
  --execution-contract "$contract" \
  --output "$root/generation_audit.json"

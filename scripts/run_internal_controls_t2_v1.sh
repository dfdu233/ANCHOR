#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LIMIT=${LIMIT:-1}
RUN_TAG=${RUN_TAG:-smoke_n1}
manifest=corrected_runs/unified_eval/inputs/vqa_rad_internal_control_t2_n32_v1.json
images=/home/dbw/datasets/public/vqa_rad_hf/internal_control_dev_images
root=corrected_runs/unified_eval/smoke/internal_controls_t2_v1/${RUN_TAG}
mkdir -p "$root" corrected_runs/detached_jobs/locks

exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
flock 8

run_model() {
  local model=$1 python=$2
  PYTHONPATH=. "$python" -m anchor.medeval.run_native_oe_control_matrix_v1 \
    --model "$model" \
    --manifest "$manifest" \
    --image-root "$images" \
    --output-root "$root/$model" \
    --execution-contract configs/unified_eval/internal_control_t2_execution_v1.json \
    --limit "$LIMIT"
}

run_model huatuo /opt/miniconda3/envs/huatuo/bin/python
run_model hulu /home/dbw/.venvs/hulumed/bin/python

flock -u 8

PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_internal_control_generation_t2_v1 \
  --run-root "$root" \
  --pilot-manifest "$manifest" \
  --freeze-provenance corrected_runs/unified_eval/inputs/vqa_rad_internal_control_t2_n32_v1.provenance.json \
  --execution-contract configs/unified_eval/internal_control_t2_execution_v1.json \
  --limit "$LIMIT" \
  --output "$root/generation_audit.json"

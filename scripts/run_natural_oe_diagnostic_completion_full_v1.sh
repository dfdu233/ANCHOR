#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks

exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
flock 8

export PYTHONPATH=anchor:/home/dbw/HuatuoGPT-Vision
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

output_dir=corrected_runs/specificity_ratchet/natural_oe_diagnostic_completion_huatuo_v1
resume_args=()
if [[ -f "$output_dir/generation_config.json" ]]; then
  resume_args=(--resume)
fi

/opt/miniconda3/envs/huatuo/bin/python -u \
  -m corrected_sgta.run_natural_oe_diagnostic_completion_pilot_v1 \
  --pilot-contract corrected_runs/specificity_ratchet/natural_oe_diagnostic_completion_pilot_manifest_v2/pilot_contract.json \
  --authorization corrected_runs/specificity_ratchet/natural_oe_diagnostic_completion_pilot_manifest_v2/full_pilot_authorization.json \
  --output-dir "$output_dir" \
  --max-new-tokens 256 \
  "${resume_args[@]}"

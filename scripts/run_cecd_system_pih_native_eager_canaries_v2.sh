#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR

if [[ "${CECD_EXPLICIT_CANARY_LAUNCH:-}" != "run-native-eager-canaries-v2-query-chunked" ]]; then
  echo "INERT: no v2 canary was launched."
  echo "Use the exact frozen v2 launch token."
  exit 64
fi

lock=/home/dbw/ANCHOR/corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
output_root=/home/dbw/ANCHOR/corrected_runs/vindr_v2/system_pih_control_preflight_v1/canaries
handoff=/home/dbw/ANCHOR/configs/cecd_system_pih_native_eager_canary_handoff_v2.json
handoff_schema=cecd-system-pih-native-eager-canary-handoff-v2-query-chunked
verifier=/home/dbw/ANCHOR/anchor/corrected_sgta/verify_frozen_handoff_bindings_v1.py
huatuo_python=/opt/miniconda3/envs/huatuo/bin/python
hulu_python=/home/dbw/.venvs/hulumed/bin/python
mkdir -p "$(dirname "$lock")" "$output_root"

python "$verifier" --handoff "$handoff" --expected-schema "$handoff_schema" --phase pre_lock

exec 8>"$lock"
flock 8

# A queued job may wait for hours while another session edits the worktree.
# Revalidate after acquiring the GPU and before importing any model code.
python "$verifier" --handoff "$handoff" --expected-schema "$handoff_schema" --phase post_lock

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

huatuo_output="$output_root/huatuo_native_eager_canary_v2_query_chunked.json"
hulu_output="$output_root/hulu_native_eager_canary_v2_query_chunked.json"
if [[ -e "$huatuo_output" || -e "$hulu_output" ]]; then
  echo "ABORT: a write-once v2 canary path already exists; audit it before any rerun." >&2
  exit 65
fi

env PYTHONPATH=/home/dbw/ANCHOR:/home/dbw/HuatuoGPT-Vision \
  "$huatuo_python" -u -m anchor.corrected_sgta.cecd_system_pih_runtime_integration_v1 canary \
  --family huatuo \
  --model-factory anchor.corrected_sgta.cecd_system_pih_canary_factories_v1:huatuo_model_factory \
  --input-factory anchor.corrected_sgta.cecd_system_pih_canary_factories_v1:huatuo_input_factory \
  --output "$huatuo_output"

env PYTHONPATH=/home/dbw/ANCHOR PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$hulu_python" -u -m anchor.corrected_sgta.cecd_system_pih_runtime_integration_v1 canary \
  --family hulu \
  --model-factory anchor.corrected_sgta.cecd_system_pih_canary_factories_v1:hulu_model_factory \
  --input-factory anchor.corrected_sgta.cecd_system_pih_canary_factories_v1:hulu_input_factory \
  --output "$hulu_output"

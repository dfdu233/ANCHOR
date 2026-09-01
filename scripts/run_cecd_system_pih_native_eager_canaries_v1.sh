#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR

# This launcher is a handoff, not an authorization.  It remains inert unless
# the operator supplies the exact one-shot phrase below.
if [[ "${CECD_EXPLICIT_CANARY_LAUNCH:-}" != "run-native-eager-canaries-v1" ]]; then
  echo "INERT: no canary was launched."
  echo "After the unified evaluation releases GPU 0, execute explicitly with:"
  echo "CECD_EXPLICIT_CANARY_LAUNCH=run-native-eager-canaries-v1 bash scripts/run_cecd_system_pih_native_eager_canaries_v1.sh"
  exit 64
fi

lock=/home/dbw/ANCHOR/corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
output_root=/home/dbw/ANCHOR/corrected_runs/vindr_v2/system_pih_control_preflight_v1/canaries
huatuo_python=/opt/miniconda3/envs/huatuo/bin/python
hulu_python=/home/dbw/.venvs/hulumed/bin/python
mkdir -p "$(dirname "$lock")" "$output_root"

# Blocking flock is intentional: the explicitly launched process waits for the
# current unified-eval owner, then retains the same GPU lock across both models.
exec 8>"$lock"
flock 8

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

huatuo_output="$output_root/huatuo_native_eager_canary_v1.json"
hulu_output="$output_root/hulu_native_eager_canary_v1.json"
if [[ -e "$huatuo_output" || -e "$hulu_output" ]]; then
  echo "ABORT: a write-once canary path already exists; audit it before any rerun." >&2
  exit 65
fi

env PYTHONPATH=/home/dbw/ANCHOR:/home/dbw/HuatuoGPT-Vision \
  "$huatuo_python" -u -m anchor.corrected_sgta.cecd_system_pih_runtime_integration_v1 canary \
  --family huatuo \
  --model-factory anchor.corrected_sgta.cecd_system_pih_canary_factories_v1:huatuo_model_factory \
  --input-factory anchor.corrected_sgta.cecd_system_pih_canary_factories_v1:huatuo_input_factory \
  --output "$huatuo_output"

# Huatuo must pass before Hulu is loaded.  The shared lock remains held.
env PYTHONPATH=/home/dbw/ANCHOR \
  "$hulu_python" -u -m anchor.corrected_sgta.cecd_system_pih_runtime_integration_v1 canary \
  --family hulu \
  --model-factory anchor.corrected_sgta.cecd_system_pih_canary_factories_v1:hulu_model_factory \
  --input-factory anchor.corrected_sgta.cecd_system_pih_canary_factories_v1:hulu_input_factory \
  --output "$hulu_output"

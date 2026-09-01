#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/llava-mitigation-native-eos-t2-v1.lock
flock -n 9 || exit 75
exec 8>corrected_runs/detached_jobs/locks/gpu0-paper-baselines-v1.lock
flock 8

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ANCHOR_RULE_ROOT=/home/dbw/ANCHOR/third_party/paper_baseline_sources/RULE

PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.run_rule_mitigation \
  --out corrected_runs/paper_baselines_v1/llava_mitigation_leading_native_eos_t2_v1 \
  --datasets iuxray mimic \
  --methods DoLa VCD OPERA AVISC M3ID PAI PAIControl \
  --chunk-size 32 --limit 32 \
  --model-path /home/dbw/models/LLaVA-Med-v1.5-mistral-7b \
  --python /home/dbw/ANCHOR/.venv-full/bin/python \
  --gpu 0 --continue-on-error

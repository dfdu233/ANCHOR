#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks corrected_runs/paper_baselines_v1/cross_model_dola
exec 9>corrected_runs/detached_jobs/locks/cross-model-dola-t2-v1.lock
flock -n 9 || exit 75
exec 8>corrected_runs/detached_jobs/locks/gpu0-paper-baselines-v1.lock
flock 8

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images

PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python -u \
  -m corrected_sgta.run_cross_model_dola_gate_v1 \
  --model huatuo --manifest "$manifest" --image-root "$images" \
  --output corrected_runs/paper_baselines_v1/cross_model_dola/huatuo_t2_n32.jsonl \
  --limit 32 --max-new-tokens 64 --seed 42

PYTHONPATH=anchor /home/dbw/.venvs/hulumed/bin/python -u \
  -m corrected_sgta.run_cross_model_dola_gate_v1 \
  --model hulu --manifest "$manifest" --image-root "$images" \
  --output corrected_runs/paper_baselines_v1/cross_model_dola/hulu_t2_n32.jsonl \
  --limit 32 --max-new-tokens 64 --seed 42

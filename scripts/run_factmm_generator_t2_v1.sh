#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/factmm-generator-t2-v1.lock
flock -n 9 || exit 75
exec 8>corrected_runs/detached_jobs/locks/gpu0-paper-baselines-v1.lock
flock 8

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python=/opt/miniconda3/envs/huatuo/bin/python
manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
output=corrected_runs/paper_baselines_v1/trained_llava_t2_v1/factmm-rag-generator

PYTHONPATH=anchor "$python" -u -m corrected_sgta.run_trained_llava_baseline_v1 \
  --variant factmm-rag-generator --manifest "$manifest" --image-root "$images" \
  --output-dir "$output" --limit 32 --max-new-tokens 128 --seed 42
PYTHONPATH=. "$python" -m anchor.medeval.qualify_oe_generation \
  --manifest "$manifest" --answers "$output/answers.jsonl" --limit 32 \
  --max-new-tokens 128 --terminal-question-policy explicit_sentence_instruction \
  --output "$output/qualification.json"

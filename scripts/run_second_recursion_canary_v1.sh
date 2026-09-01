#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/second-recursion-canary.lock
if ! flock -n 9; then
  echo "Another SECOND recursion canary owns the lock" >&2
  exit 75
fi

second_root=/home/dbw/third_party/SECOND
second_eval=$second_root/lmms-eval-vicuna
second_python=/home/dbw/envs/second/bin/python
task_path=/home/dbw/ANCHOR/configs/unified_eval/lmms_tasks/vqa_rad_oe
out=/home/dbw/ANCHOR/corrected_runs/unified_eval/sanity/second_recursion_canary_v1
mkdir -p "$out"

export PYTHONPATH=$second_eval:$second_eval/LLaVA-NeXT:/home/dbw/ANCHOR
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

cd "$second_eval"
"$second_python" -m lmms_eval \
  --model llava \
  --model_args pretrained=/home/dbw/models/LLaVA-Med-v1.5-mistral-7b,conv_template=mistral_instruct,device_map=custom \
  --include_path "$task_path" \
  --tasks vqa_rad_official_test_oe \
  --batch_size 1 \
  --limit 1 \
  --log_samples \
  --log_samples_suffix second_recursion_canary \
  --output_path "$out/lmms" \
  --generation_type recursion \
  --fix_grid 2x2 \
  --attention_thresholding_type attn_topk \
  --attention_threshold 1.0 \
  --positional_embedding_type bilinear_interpolation \
  --stages -2 -1 0 1 \
  --contrastive_alphas 0.8 0.8 0.8 \
  --seed 42 \
  --verbosity INFO
cd /home/dbw/ANCHOR

mapfile -t samples < <(find "$out/lmms" -type f -name '*_samples_vqa_rad_official_test_oe.jsonl' | sort)
if [[ ${#samples[@]} -eq 0 ]]; then
  echo "SECOND recursion canary produced no per-sample log" >&2
  exit 2
fi
sample=${samples[$((${#samples[@]} - 1))]}
"$second_python" -m anchor.medeval.import_lmms_samples \
  --samples "$sample" --output "$out/answers.jsonl"
"$second_python" -m anchor.medeval.qualify_oe_generation \
  --manifest corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json \
  --answers "$out/answers.jsonl" --limit 1 \
  --output "$out/qualification.json"

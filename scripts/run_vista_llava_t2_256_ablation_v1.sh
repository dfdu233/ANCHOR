#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks

exec 8>corrected_runs/detached_jobs/locks/gpu0-llava.lock
flock 8

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

/opt/miniconda3/envs/huatuo/bin/python -u \
  -m corrected_sgta.run_llava_med_generation_matrix \
  --question-file corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json \
  --image-folder /home/dbw/datasets/public/vqa_rad_hf/test_images \
  --out corrected_runs/unified_eval/smoke/vista_llava_t2_256_ablation_v1 \
  --source vqa_rad \
  --dataset official_test_oe \
  --task open_vqa \
  --methods greedy VISTA_off VISTA_VSV VISTA_SLA VISTA \
  --chunk-size 32 \
  --limit 32 \
  --max-new-tokens 256 \
  --conv-mode mistral_instruct \
  --disable-keyword-stopping \
  --qualification-run

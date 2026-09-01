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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_dir=corrected_runs/vindr_v2/clinical_autoregressive_lockin_huatuo_dev_v1

/opt/miniconda3/envs/huatuo/bin/python -u \
  -m corrected_sgta.run_huatuo_lockin_pipeline_v1 \
  --manifest corrected_runs/vindr_v2/clinical_autoregressive_lockin_dev_v4/dev_manifest.jsonl \
  --metadata corrected_runs/vindr_v2/clinical_autoregressive_lockin_dev_v4/metadata.json \
  --image-root /workspace/vinbigdata \
  --output-dir "$run_dir" \
  --model-dir /home/dbw/models/HuatuoGPT-Vision-7B \
  --huatuo-root /home/dbw/HuatuoGPT-Vision \
  --device cuda:0

# Release the GPU before the deterministic block bootstrap.  Scientific
# failure is a successful completed experiment, not an operational retry.
flock -u 8

/opt/miniconda3/envs/huatuo/bin/python -u \
  -m corrected_sgta.analyze_clinical_autoregressive_lockin_v1 \
  --run-dir "$run_dir" \
  --output "$run_dir/analysis_v1.json" \
  --bootstrap-replicates 2000 \
  --seed 20260802

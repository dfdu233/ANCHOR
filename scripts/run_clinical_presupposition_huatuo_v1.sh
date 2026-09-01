#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks

# Serialize with all existing VinDr mechanism collectors.  The job may wait on
# this lock safely under the detached PPID1 supervisor.
exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
flock 8

export PYTHONPATH=anchor:/home/dbw/HuatuoGPT-Vision
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_generation() {
  local output_dir=$1
  local limit=$2
  local resume_args=()
  if [[ -f "$output_dir/generation_config.json" ]]; then
    resume_args=(--resume)
  fi
  /opt/miniconda3/envs/huatuo/bin/python -u \
    -m corrected_sgta.run_clinical_presupposition_generation_v1 \
    --labels-csv /home/dbw/datasets/physionet/vindr-cxr/1.0.0/annotations/image_labels_train.csv \
    --ontology /home/dbw/ANCHOR/configs/missing_third_state_vindr_ontology.json \
    --image-root /workspace/vinbigdata/train \
    --output-dir "$output_dir" \
    --split pilot \
    --limit "$limit" \
    --seed 42 \
    --max-new-tokens 256 \
    "${resume_args[@]}"
}

# A real one-image/three-condition pass exercises DICOM decoding, multimodal
# generation, token accounting, and aggregate creation.  Only a clean exit
# permits the 200-image generation-only screen to start.
run_generation corrected_runs/vindr_v2/clinical_presupposition_huatuo_canary_v1 1
run_generation corrected_runs/vindr_v2/clinical_presupposition_huatuo_generation_v1 200

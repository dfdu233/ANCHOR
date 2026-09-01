#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/llava-trace-conformance.lock
if ! flock -n 9; then
  echo "Another LLaVA trace run owns the lock" >&2
  exit 75
fi

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
model=/home/dbw/models/LLaVA-Med-v1.5-mistral-7b
manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
canonical_root=/home/dbw/ANCHOR/data/medheval/code/baselines/Med-LVLMs/llava-med-1.5
port_root=/home/dbw/ANCHOR/data/medheval/code/baselines/Mitigation/llava-med-1.5
transformers_root=/home/dbw/ANCHOR/data/medheval/code/baselines/Med-LVLMs/llava-med-1.5/transformers-4.37.2/src
output=corrected_runs/unified_eval/sanity/llava_trace_conformance_v2
mkdir -p "$output"

PYTHONPATH=.:"$canonical_root" \
  /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.trace_llava_backend trace \
  --backend-root "$canonical_root" --model-path "$model" \
  --manifest "$manifest" --image-root "$images" \
  --qids vqa-rad-test-0011 vqa-rad-test-0020 \
  --output "$output/canonical.trace.json"

PYTHONPATH=.:"$port_root":"$transformers_root" \
  /home/dbw/ANCHOR/.venv-full/bin/python -m anchor.medeval.trace_llava_backend trace \
  --backend-root "$port_root" --model-path "$model" \
  --manifest "$manifest" --image-root "$images" \
  --qids vqa-rad-test-0011 vqa-rad-test-0020 \
  --output "$output/port.trace.json"

PYTHONPATH=.:"$port_root":"$transformers_root" \
  /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.trace_llava_backend trace \
  --backend-root "$port_root" --model-path "$model" \
  --manifest "$manifest" --image-root "$images" \
  --qids vqa-rad-test-0011 vqa-rad-test-0020 \
  --output "$output/port_canonical_runtime.trace.json"

PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.medeval.trace_llava_backend compare \
  --left "$output/canonical.trace.json" --right "$output/port.trace.json" \
  --output "$output/comparison.json"

PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.medeval.trace_llava_backend compare \
  --left "$output/canonical.trace.json" \
  --right "$output/port_canonical_runtime.trace.json" \
  --output "$output/comparison_canonical_runtime.json"

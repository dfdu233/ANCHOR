#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
upstream=corrected_runs/detached_jobs/llava-canonical-runtime-gate-v2.json
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/huatuo-native-vqa-rad-oe-512.lock
if ! flock -n 9; then
  echo "Another Huatuo 512-token OE run owns the lock" >&2
  exit 75
fi

export PYTHONPATH=anchor:/home/dbw/HuatuoGPT-Vision
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
smoke=corrected_runs/unified_eval/smoke/huatuo_native_vqa_rad_oe_v3_512
full=corrected_runs/unified_eval/full/huatuo_native_vqa_rad_oe_v3_512

/opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.run_huatuo_native_oe_vqa \
  --manifest "$manifest" --image-root "$images" --output-dir "$smoke" \
  --limit 32 --max-new-tokens 512 --seed 42
/opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_oe_generation \
  --manifest "$manifest" --answers "$smoke/answers.jsonl" --limit 32 \
  --max-new-tokens 512 --require-terminal-completeness \
  --output "$smoke/qualification.json"

/opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.run_huatuo_native_oe_vqa \
  --manifest "$manifest" --image-root "$images" --output-dir "$full" \
  --max-new-tokens 512 --seed 42
/opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_oe_generation \
  --manifest "$manifest" --answers "$full/answers.jsonl" --limit 200 \
  --max-new-tokens 512 --require-terminal-completeness \
  --output "$full/qualification.json"
/opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.evaluate_oe_vqa \
  --manifest "$manifest" --answers "$full/answers.jsonl" \
  --output "$full/evaluation.json" --bootstrap-replicates 5000 --seed 42

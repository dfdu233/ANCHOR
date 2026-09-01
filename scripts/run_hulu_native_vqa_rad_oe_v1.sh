#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/hulu-native-vqa-rad-oe.lock
if ! flock -n 9; then
  echo "Another native Hulu OE run owns the lock; experiment not started"
  exit 75
fi
upstream=corrected_runs/detached_jobs/llava-native-vqa-rad-oe-v1.json
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done

export PYTHONPATH=anchor
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
smoke=corrected_runs/unified_eval/smoke/hulu_native_vqa_rad_oe_v1
full=corrected_runs/unified_eval/full/hulu_native_vqa_rad_oe_v1

/home/dbw/.venvs/hulumed/bin/python -m anchor.medeval.run_native_oe_vqa \
  --model hulu --manifest "$manifest" --image-root "$images" --output-dir "$smoke" \
  --limit 32 --max-new-tokens 64 --seed 42
/home/dbw/.venvs/hulumed/bin/python -m anchor.medeval.qualify_oe_generation \
  --manifest "$manifest" --answers "$smoke/answers.jsonl" --limit 32 \
  --output "$smoke/qualification.json"
/home/dbw/.venvs/hulumed/bin/python -m anchor.medeval.run_native_oe_vqa \
  --model hulu --manifest "$manifest" --image-root "$images" --output-dir "$full" \
  --max-new-tokens 64 --seed 42
/home/dbw/.venvs/hulumed/bin/python -m anchor.medeval.evaluate_oe_vqa \
  --manifest "$manifest" --answers "$full/answers.jsonl" \
  --output "$full/evaluation.json" --bootstrap-replicates 5000 --seed 42

#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR

# This predecessor is a resource-serialization barrier, not a scientific
# dependency. Native Huatuo remains valid when the mitigation backend fails
# its identity gate, so wait for terminal state but do not require success.
upstream=${RESOURCE_UPSTREAM:-corrected_runs/detached_jobs/huatuo-native-vqa-rad-oe-v4.json}
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done
if [[ "$status" == "done" ]]; then
  echo "Native Huatuo OE was completed by the serialized predecessor"
  exit 0
fi

mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/huatuo-native-vqa-rad-oe.lock
if ! flock -n 9; then
  echo "Another native Huatuo OE run owns the lock" >&2
  exit 75
fi

export PYTHONPATH=anchor:/home/dbw/HuatuoGPT-Vision
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
smoke=corrected_runs/unified_eval/smoke/huatuo_native_vqa_rad_oe_v1
full=corrected_runs/unified_eval/full/huatuo_native_vqa_rad_oe_v1

/opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.run_huatuo_native_oe_vqa \
  --manifest "$manifest" --image-root "$images" --output-dir "$smoke" \
  --limit 32 --max-new-tokens 64 --seed 42
/opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_oe_generation \
  --manifest "$manifest" --answers "$smoke/answers.jsonl" --limit 32 \
  --output "$smoke/qualification.json"
/opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.run_huatuo_native_oe_vqa \
  --manifest "$manifest" --image-root "$images" --output-dir "$full" \
  --max-new-tokens 64 --seed 42
/opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.evaluate_oe_vqa \
  --manifest "$manifest" --answers "$full/answers.jsonl" \
  --output "$full/evaluation.json" --bootstrap-replicates 5000 --seed 42

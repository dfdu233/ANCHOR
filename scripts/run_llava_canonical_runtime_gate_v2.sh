#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
upstream=corrected_runs/detached_jobs/huatuo-native-vqa-rad-oe-256-v1.json
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done

mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/llava-canonical-runtime-gate.lock
if ! flock -n 9; then
  echo "Another LLaVA runtime gate owns the lock" >&2
  exit 75
fi

export PYTHONPATH=anchor
export ANCHOR_PYTHON=/opt/miniconda3/envs/huatuo/bin/python
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
root=corrected_runs/unified_eval/sanity/llava_canonical_runtime_gate_v2

for limit in 32 128; do
  canonical="$root/n${limit}/canonical"
  port="$root/n${limit}/port"
  /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.run_native_oe_vqa \
    --model llava --manifest "$manifest" --image-root "$images" \
    --output-dir "$canonical" --limit "$limit" --max-new-tokens 64 --seed 42

  /opt/miniconda3/envs/huatuo/bin/python \
    -m corrected_sgta.run_llava_med_generation_matrix \
    --question-file "$manifest" --image-folder "$images" --out "$port" \
    --source vqa_rad --dataset official_test_oe --task open_vqa \
    --methods greedy --chunk-size "$limit" --limit "$limit" \
    --max-new-tokens 64 --conv-mode mistral_instruct --seed 42 \
    --disable-keyword-stopping --qualification-run

  candidate="$port/vqa_rad/official_test_oe/open_vqa/greedy/chunk_0000.answers.jsonl"
  /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.medeval.evaluate_backend_conformance \
    --canonical "$canonical/answers.jsonl" --candidate "$candidate" \
    --min-normalized-exact 1 --min-token-f1 1 --require-token-exact \
    --output "$root/n${limit}/conformance.json"
done

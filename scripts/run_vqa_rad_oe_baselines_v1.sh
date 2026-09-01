#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/vqa-rad-oe-baselines.lock
if ! flock -n 9; then
  echo "Another VQA-RAD OE baseline pipeline owns the run lock; exiting cleanly"
  exit 0
fi

upstream=corrected_runs/detached_jobs/hulu-report-dependency-audit-v1.json
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done

export PYTHONPATH=anchor
export ANCHOR_MODEL_PATH=/home/dbw/models/LLaVA-Med-v1.5-mistral-7b
input=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
smoke=corrected_runs/unified_eval/smoke/vqa_rad_oe_mitigation_v3

/opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.run_llava_med_generation_matrix \
  --question-file "$input" \
  --image-folder "$images" \
  --out "$smoke" \
  --source vqa_rad \
  --dataset official_test_oe \
  --task open_vqa \
  --methods greedy beam DoLa PAI opera avisc m3id VCD damro \
  --chunk-size 32 \
  --limit 32 \
  --max-new-tokens 64 \
  --conv-mode mistral_instruct \
  --seed 42 \
  --qualification-run \
  --continue-on-error

methods=$(/opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.medeval.select_mitigation_smoke \
  --smoke-root "$smoke" \
  --output "$smoke/selection.json")
if [[ " $methods " != *" greedy "* ]]; then
  echo "Base greedy failed VQA-RAD OE qualification; full mitigation run skipped"
  exit 0
fi

full=corrected_runs/unified_eval/full/vqa_rad_oe_mitigation_v3
# Intentional word splitting: selector emits validated registry names only.
/opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.run_llava_med_generation_matrix \
  --question-file "$input" \
  --image-folder "$images" \
  --out "$full" \
  --source vqa_rad \
  --dataset official_test_oe \
  --task open_vqa \
  --methods $methods \
  --chunk-size 64 \
  --max-new-tokens 64 \
  --conv-mode mistral_instruct \
  --seed 42 \
  --continue-on-error

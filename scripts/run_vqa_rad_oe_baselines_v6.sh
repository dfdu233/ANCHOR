#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/vqa-rad-oe-baselines-v6.lock
if ! flock -n 9; then
  echo "Another corrected VQA-RAD OE baseline pipeline owns the run lock" >&2
  exit 75
fi

upstream=${SECOND_RUN_UPSTREAM:-corrected_runs/detached_jobs/second-vqa-rad-oe-v9.json}
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done
if [[ "$status" != "done" && "${ALLOW_FAILED_UPSTREAM:-false}" != "true" ]]; then
  echo "SECOND predecessor failed; corrected mitigation matrix not started" >&2
  exit 2
fi

export PYTHONPATH=anchor
export ANCHOR_MODEL_PATH=/home/dbw/models/LLaVA-Med-v1.5-mistral-7b
input=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
smoke=corrected_runs/unified_eval/smoke/vqa_rad_oe_mitigation_v4
full=corrected_runs/unified_eval/full/vqa_rad_oe_mitigation_v4
canonical32=corrected_runs/unified_eval/sanity/second_identity_conformance_v1/canonical.answers.jsonl

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
  --disable-keyword-stopping \
  --continue-on-error

greedy_smoke="$smoke/vqa_rad/official_test_oe/open_vqa/greedy/chunk_0000.answers.jsonl"
/opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.medeval.evaluate_backend_conformance \
  --canonical "$canonical32" \
  --candidate "$greedy_smoke" \
  --output "$smoke/greedy_backend_conformance.json"

methods=$(/opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.medeval.select_mitigation_smoke \
  --smoke-root "$smoke" \
  --output "$smoke/selection.json")
if [[ " $methods " != *" greedy "* ]]; then
  echo "Corrected greedy failed qualification; full mitigation run blocked" >&2
  exit 2
fi

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
  --disable-keyword-stopping \
  --continue-on-error

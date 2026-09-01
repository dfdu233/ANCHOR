#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/llava-port-diagnostic.lock
if ! flock -n 9; then
  echo "Another LLaVA port diagnostic owns the run lock; experiment not started"
  exit 75
fi

upstream=corrected_runs/detached_jobs/second-vqa-rad-oe-v8.json
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done

out=corrected_runs/unified_eval/sanity/llava_mitigation_port_diagnostic_v1
manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
model=/home/dbw/models/LLaVA-Med-v1.5-mistral-7b
mkdir -p "$out"

/opt/miniconda3/bin/python -c \
  'import json,sys; rows=json.load(open(sys.argv[1]))[:4]; json.dump(rows,open(sys.argv[2],"w"),indent=2)' \
  "$manifest" "$out/questions.json"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/dbw/ANCHOR/data/medheval/code/baselines/Mitigation/llava-med-1.5:/home/dbw/ANCHOR/data/medheval/code/baselines/Mitigation/llava-med-1.5/llava/eval:/home/dbw/ANCHOR/data/medheval/code/baselines/Med-LVLMs/llava-med-1.5/transformers-4.37.2/src:/home/dbw/ANCHOR

/home/dbw/ANCHOR/.venv-full/bin/python \
  data/medheval/code/baselines/Mitigation/llava-med-1.5/llava/eval/model_vqa.py \
  --model-path "$model" --image-folder "$images" --question-file "$out/questions.json" \
  --answers-file "$out/port_stopping.answers.jsonl" --conv-mode mistral_instruct \
  --temperature 0 --top_p 1 --num_beams 1 --baseline greedy \
  --max-new-tokens 64 --seed 42

/home/dbw/ANCHOR/.venv-full/bin/python \
  data/medheval/code/baselines/Mitigation/llava-med-1.5/llava/eval/model_vqa.py \
  --model-path "$model" --image-folder "$images" --question-file "$out/questions.json" \
  --answers-file "$out/port_no_stopping.answers.jsonl" --conv-mode mistral_instruct \
  --temperature 0 --top_p 1 --num_beams 1 --baseline greedy \
  --max-new-tokens 64 --seed 42 --disable-keyword-stopping

export PYTHONPATH=/home/dbw/ANCHOR/anchor:/home/dbw/ANCHOR/data/medheval/code/baselines/Med-LVLMs/llava-med-1.5
/opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.diagnose_llava_port canonical \
  --manifest "$manifest" --image-root "$images" --output "$out/canonical.answers.jsonl" \
  --limit 4 --max-new-tokens 64 --seed 42

if ! /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.evaluate_backend_conformance \
  --canonical "$out/canonical.answers.jsonl" \
  --candidate "$out/port_stopping.answers.jsonl" \
  --output "$out/port_stopping.conformance.json"; then
  echo "Keyword-stopping port failed identity conformance as a scientific result"
fi
if ! /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.evaluate_backend_conformance \
  --canonical "$out/canonical.answers.jsonl" \
  --candidate "$out/port_no_stopping.answers.jsonl" \
  --output "$out/port_no_stopping.conformance.json"; then
  echo "No-keyword-stopping port failed identity conformance as a scientific result"
fi

/opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.diagnose_llava_port summarize \
  --answers canonical="$out/canonical.answers.jsonl" \
  port_stopping="$out/port_stopping.answers.jsonl" \
  port_no_stopping="$out/port_no_stopping.answers.jsonl" \
  --output "$out/summary.json"

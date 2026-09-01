#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
upstream=corrected_runs/detached_jobs/huatuo-native-vqa-rad-oe-512-v1.json
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done

mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/common-rag-ce-ladder-v1.lock
if ! flock -n 9; then
  echo "Another common RAG CE ladder owns the lock" >&2
  exit 75
fi

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python=/opt/miniconda3/envs/huatuo/bin/python
root=corrected_runs/unified_eval/rag/common_protocol_v1

run_generation() {
  local model=$1 manifest=$2 images=$3 out=$4 limit=$5
  if [[ "$model" == "huatuo" ]]; then
    PYTHONPATH=anchor:/home/dbw/HuatuoGPT-Vision "$python" \
      -m anchor.medeval.run_huatuo_native_oe_vqa \
      --manifest "$manifest" --image-root "$images" --output-dir "$out" \
      --limit "$limit" --max-new-tokens 64 --seed 42
  else
    PYTHONPATH=anchor "$python" -m anchor.medeval.run_native_oe_vqa \
      --model "$model" --manifest "$manifest" --image-root "$images" \
      --output-dir "$out" --limit "$limit" --max-new-tokens 64 --seed 42
  fi
}

qualify_and_score() {
  local manifest=$1 out=$2
  PYTHONPATH=. "$python" -m anchor.medeval.qualify_ce_generation \
    --manifest "$manifest" --answers "$out/answers.jsonl" \
    --max-new-tokens 64 --output "$out/qualification.json" || return 1
  PYTHONPATH=anchor "$python" -m corrected_sgta.evaluate_medheval_answers \
    --answers "$out/answers.jsonl" --questions "$manifest" \
    --output "$out/evaluation.json" || return 1
}

failed=0
for dataset in iuxray mimic; do
  if [[ "$dataset" == "iuxray" ]]; then
    images=/home/dbw/ANCHOR/data/medheval/images/IU-Xray
  else
    images=/home/dbw/ANCHOR/data/medheval/images
  fi
  prompts="$root/$dataset/t3_n200_top3/prompts"
  for model in huatuo hulu llava; do
    smoke_ok=1
    for arm in no_context rag; do
      manifest="$prompts/$arm.json"
      out="$root/$dataset/ladder_v1/T2_n32/$model/$arm"
      if ! run_generation "$model" "$manifest" "$images" "$out" 32 || \
         ! qualify_and_score "$manifest" "$out"; then
        echo "T2 failed: dataset=$dataset model=$model arm=$arm" >&2
        smoke_ok=0
        failed=1
      fi
    done
    [[ "$smoke_ok" == 1 ]] || continue
    pilot_ok=1
    for arm in no_context rag; do
      manifest="$prompts/$arm.json"
      out="$root/$dataset/ladder_v1/T3_n200/$model/$arm"
      if ! run_generation "$model" "$manifest" "$images" "$out" 200 || \
         ! qualify_and_score "$manifest" "$out"; then
        echo "T3 failed: dataset=$dataset model=$model arm=$arm" >&2
        pilot_ok=0
        failed=1
      fi
    done
    [[ "$pilot_ok" == 1 ]] || continue
    PYTHONPATH=. "$python" -m anchor.medeval.compare_ce_arms \
      --manifest "$prompts/no_context.json" \
      --baseline "$root/$dataset/ladder_v1/T3_n200/$model/no_context/answers.jsonl" \
      --candidate "$root/$dataset/ladder_v1/T3_n200/$model/rag/answers.jsonl" \
      --output "$root/$dataset/ladder_v1/T3_n200/$model/comparison.json" \
      --bootstrap-draws 5000 --seed 42 || failed=1
  done
done

exit "$failed"

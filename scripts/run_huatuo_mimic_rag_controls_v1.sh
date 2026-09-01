#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/huatuo-mimic-rag-controls-v1.lock
flock -n 9 || exit 75
exec 8>corrected_runs/detached_jobs/locks/gpu0-paper-baselines-v1.lock
flock 8

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python=/opt/miniconda3/envs/huatuo/bin/python
root=corrected_runs/unified_eval/rag/common_protocol_v1/mimic/visual_ce_v2
prompts=$root/t3_n200_top3/prompts
controls=$root/t3_n200_top3/controls_v1
outroot=$root/ladder_v3/causal_controls_v1
images=/home/dbw/ANCHOR/data/medheval/images
max_tokens=128

run_arm() {
  local manifest=$1 output=$2 limit=$3
  mkdir -p "$output"
  PYTHONPATH=anchor:/home/dbw/HuatuoGPT-Vision "$python" \
    -m anchor.medeval.run_huatuo_native_oe_vqa \
    --manifest "$manifest" --image-root "$images" --output-dir "$output" \
    --limit "$limit" --max-new-tokens "$max_tokens" --seed 42
  PYTHONPATH=. "$python" -m anchor.medeval.qualify_ce_generation \
    --manifest "$manifest" --answers "$output/answers.jsonl" --limit "$limit" \
    --max-new-tokens "$max_tokens" --output "$output/qualification.json"
  PYTHONPATH=anchor "$python" -m corrected_sgta.evaluate_medheval_answers \
    --answers "$output/answers.jsonl" --questions "$manifest" \
    --output "$output/evaluation.json"
}

for tier in T2_n32 T3_n200; do
  if [[ "$tier" == T2_n32 ]]; then limit=32; else limit=200; fi
  run_arm "$controls/shuffled_context.json" \
    "$outroot/$tier/huatuo/shuffled_context" "$limit"
done

PYTHONPATH=. "$python" -m anchor.medeval.compare_ce_arms \
  --manifest "$prompts/rag.json" \
  --baseline "$outroot/T3_n200/huatuo/shuffled_context/answers.jsonl" \
  --candidate "$root/ladder_v3/T3_n200/huatuo/rag/answers.jsonl" \
  --output "$outroot/rag_vs_shuffled_context_huatuo.json" \
  --bootstrap-draws 5000 --seed 42

set +e
"$python" - "$outroot/rag_vs_shuffled_context_huatuo.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
raise SystemExit(0 if result.get("full_run_authorized") is True else 10)
PY
relevance_status=$?
set -e
if [[ "$relevance_status" -eq 10 ]]; then
  exit 0
fi

for tier in T2_n32 T3_n200; do
  if [[ "$tier" == T2_n32 ]]; then limit=32; else limit=200; fi
  run_arm "$controls/image_swap.json" "$outroot/$tier/huatuo/image_swap" "$limit"
done

PYTHONPATH=. "$python" -m anchor.medeval.compare_ce_arms \
  --manifest "$prompts/rag.json" \
  --baseline "$outroot/T3_n200/huatuo/image_swap/answers.jsonl" \
  --candidate "$root/ladder_v3/T3_n200/huatuo/rag/answers.jsonl" \
  --output "$outroot/rag_vs_image_swap_huatuo.json" \
  --bootstrap-draws 5000 --seed 42

#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
DATA=/root/autodl-tmp/MedHEval/benchmark_data
GPU_IDS="${GPU_IDS:-0}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-384}"
CANDIDATE_BATCH="${CANDIDATE_BATCH:-4}"
GENERATION_SEED="${GENERATION_SEED:-42}"
read -r -a GAMMAS <<< "${GAMMAS:-0.8 1.2}"
read -r -a ANALYSIS_SEEDS <<< "${ANALYSIS_SEEDS:-42}"

case "$MODE" in
  smoke)
    OUT="${OUT_DIR:-$ROOT/corrected_runs/optimized_oe_smoke_v53}"
    LIMIT="${MAX_SAMPLES:-16}"
    CANDIDATES="${CANDIDATES:-4}"
    ;;
  full)
    OUT="${OUT_DIR:-$ROOT/corrected_runs/optimized_oe_full_v53}"
    LIMIT="${MAX_SAMPLES:-0}"
    CANDIDATES="${CANDIDATES:-8}"
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT"
cd "$ROOT"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
MODELS=(hulu llava hulu llava)
NAMES=(knowledge_oe knowledge_oe report_oe report_oe)
TOKENS=(128 128 192 192)
DATASETS=(
  "$DATA/Knowledge_Deficiency_Hallucination/open-ended/MIMIC-CXR_pairs.json"
  "$DATA/Knowledge_Deficiency_Hallucination/open-ended/MIMIC-CXR_pairs.json"
  "$DATA/Visual_Misinterpretation_Hallucination/open-ended/MIMIC-CXR_pairs.json"
  "$DATA/Visual_Misinterpretation_Hallucination/open-ended/MIMIC-CXR_pairs.json"
)

run_job() {
  local index="$1"
  local gpu="$2"
  local model="${MODELS[$index]}"
  local name="${NAMES[$index]}"
  local tokens="${TOKENS[$index]}"
  local dataset="${DATASETS[$index]}"
  local cache="$OUT/${model}_${name}.jsonl"
  local analysis_seed
  echo "[$(date -Is)] gpu=$gpu model=$model dataset=$name"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m corrected_sgta.infer_oe \
    --model "$model" \
    --dataset "$dataset" \
    --output "$cache" \
    --max-samples "$LIMIT" \
    --max-image-side "$MAX_IMAGE_SIDE" \
    --candidates "$CANDIDATES" \
    --candidate-batch "$CANDIDATE_BATCH" \
    --max-new-tokens "$tokens" \
    --seed "$GENERATION_SEED" \
    --style-augmentation \
    --center-policy all \
    --feddg-l-values 0.003 \
    --feddg-source-ratios 0.0 \
    --gammas "${GAMMAS[@]}"
  for analysis_seed in "${ANALYSIS_SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m corrected_sgta.analyze_confgen \
      --cache "$cache" \
      --output "$OUT/${model}_${name}.confgen_optimized.seed${analysis_seed}.json" \
      --seed "$analysis_seed"
  done
}

worker() {
  local worker_index="$1"
  local gpu="$2"
  local index
  for ((index=worker_index; index<${#MODELS[@]}; index+=${#GPU_LIST[@]})); do
    run_job "$index" "$gpu"
  done
}

pids=()
for worker_index in "${!GPU_LIST[@]}"; do
  worker "$worker_index" "${GPU_LIST[$worker_index]}" &
  pids+=("$!")
done
trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "Completed optimized OE matrix: $OUT"

#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
DATA=/root/autodl-tmp/MedHEval/benchmark_data
GPU_IDS="${GPU_IDS:-0}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-384}"
read -r -a FEDDG_L_VALUES <<< "${FEDDG_L_VALUES:-0.003 0.008}"
read -r -a FEDDG_SOURCE_RATIOS <<< "${FEDDG_SOURCE_RATIOS:-0.0}"
read -r -a GAMMAS <<< "${GAMMAS:-0.8 1.2}"
read -r -a ANALYSIS_SEEDS <<< "${ANALYSIS_SEEDS:-42}"

case "$MODE" in
  smoke)
    OUT="${OUT_DIR:-$ROOT/corrected_runs/optimized_ce_smoke_v53}"
    LIMIT="${MAX_SAMPLES:-128}"
    ;;
  full)
    OUT="${OUT_DIR:-$ROOT/corrected_runs/optimized_ce_full_v53}"
    LIMIT="${MAX_SAMPLES:-0}"
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT"
cd "$ROOT"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
# With two GPUs this ordering keeps Hulu on GPU 0 and LLaVA on GPU 1.
MODELS=(hulu llava hulu llava hulu llava hulu llava)
NAMES=(cxr_vishal cxr_vishal mm_vishal mm_vishal context context knowledge_ce knowledge_ce)
DATASETS=(
  "$DATA/Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json"
  "$DATA/Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json"
  "$DATA/Visual_Misinterpretation_Hallucination/close-ended/MM-VisHal.json"
  "$DATA/Visual_Misinterpretation_Hallucination/close-ended/MM-VisHal.json"
  "$DATA/Context_Misalignment_Hallucination/MIMIC-CXR_pairs.json"
  "$DATA/Context_Misalignment_Hallucination/MIMIC-CXR_pairs.json"
  "$DATA/Knowledge_Deficiency_Hallucination/close-ended/MIMIC-CXR_sampled.json"
  "$DATA/Knowledge_Deficiency_Hallucination/close-ended/MIMIC-CXR_sampled.json"
)

run_job() {
  local index="$1"
  local gpu="$2"
  local model="${MODELS[$index]}"
  local name="${NAMES[$index]}"
  local dataset="${DATASETS[$index]}"
  local cache="$OUT/${model}_${name}.jsonl"
  local analysis_seed
  echo "[$(date -Is)] gpu=$gpu model=$model dataset=$name"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m corrected_sgta.infer_ce \
    --model "$model" \
    --dataset "$dataset" \
    --output "$cache" \
    --max-samples "$LIMIT" \
    --max-image-side "$MAX_IMAGE_SIDE" \
    --center-policy all \
    --feddg-l-values "${FEDDG_L_VALUES[@]}" \
    --feddg-source-ratios "${FEDDG_SOURCE_RATIOS[@]}" \
    --gammas "${GAMMAS[@]}"
  for analysis_seed in "${ANALYSIS_SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m corrected_sgta.analyze_ce \
      --cache "$cache" \
      --output "$OUT/${model}_${name}.summary.seed${analysis_seed}.json" \
      --seed "$analysis_seed" \
      --transductive-window 256
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m corrected_sgta.tune_sgta \
      --cache "$cache" \
      --output "$OUT/${model}_${name}.sgta_optimized.seed${analysis_seed}.json" \
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

echo "Completed optimized CE matrix: $OUT"

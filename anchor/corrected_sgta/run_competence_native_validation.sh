#!/usr/bin/env bash
set -euo pipefail
GPU="${CUDA_VISIBLE_DEVICES:-0}"
ROOT="/root/autodl-tmp/Hulu-Med/MedUniEval"
PHASE="${1:-pilot}"
DATASET_NAME="${2:-cxr_vishal}"
case "${PHASE}" in
  pilot) MAX_SAMPLES=128 ;;
  val256) MAX_SAMPLES=256 ;;
  val512) MAX_SAMPLES=512 ;;
  smoke) MAX_SAMPLES=32 ;;
  *) echo "unknown phase: ${PHASE}" >&2; exit 2 ;;
esac
case "${DATASET_NAME}" in
  cxr_vishal|knowledge_ce)
    SUBSET="${ROOT}/corrected_runs/medheval_mitigation_${PHASE}_v1/${DATASET_NAME}/subset_seed42_n${MAX_SAMPLES}.json"
    GREEDY="${ROOT}/corrected_runs/medheval_mitigation_${PHASE}_v1/${DATASET_NAME}/greedy_seed42_tok16.eval.json"
    PAI="${ROOT}/corrected_runs/medheval_mitigation_${PHASE}_v1/${DATASET_NAME}/PAI_seed42_tok16.eval.json"
    OUT="${ROOT}/corrected_runs/competence_native_${PHASE}_v1/${DATASET_NAME}"
    ;;
  *) echo "unknown dataset: ${DATASET_NAME}" >&2; exit 2 ;;
esac
mkdir -p "${OUT}"
CUDA_VISIBLE_DEVICES="${GPU}" python -u -m corrected_sgta.competence_native_risk \
  --dataset "${SUBSET}" \
  --greedy-eval "${GREEDY}" \
  --mitigation-eval "${PAI}" \
  --output "${OUT}/competence_native_analysis.json" \
  --max-samples "${MAX_SAMPLES}" \
  --max-image-side 384 \
  --seed 42 \
  --question-type binary \
  --train-frac 0.4 \
  --max-token-bank 12000

echo "results: ${OUT}/competence_native_analysis.json"

#!/usr/bin/env bash
set -euo pipefail

GPU="${CUDA_VISIBLE_DEVICES:-0}"
ROOT="/root/autodl-tmp/Hulu-Med/MedUniEval"
SOURCE_BANK="${ROOT}/corrected_runs/source_bank_v2/source_bank.json"
PROTOTYPES="${ROOT}/corrected_runs/local_source_consensus_smoke_v1/llava_xray_local_prototypes.npz"
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
  cxr_vishal)
    SUBSET="${ROOT}/corrected_runs/medheval_mitigation_${PHASE}_v1/cxr_vishal/subset_seed42_n${MAX_SAMPLES}.json"
    GREEDY="${ROOT}/corrected_runs/medheval_mitigation_${PHASE}_v1/cxr_vishal/greedy_seed42_tok16.eval.json"
    PAI="${ROOT}/corrected_runs/medheval_mitigation_${PHASE}_v1/cxr_vishal/PAI_seed42_tok16.eval.json"
    OUT="${ROOT}/corrected_runs/source_domain_risk_${PHASE}_v1/cxr_vishal"
    ;;
  knowledge_ce)
    SUBSET="${ROOT}/corrected_runs/medheval_mitigation_${PHASE}_v1/knowledge_ce/subset_seed42_n${MAX_SAMPLES}.json"
    GREEDY="${ROOT}/corrected_runs/medheval_mitigation_${PHASE}_v1/knowledge_ce/greedy_seed42_tok16.eval.json"
    PAI="${ROOT}/corrected_runs/medheval_mitigation_${PHASE}_v1/knowledge_ce/PAI_seed42_tok16.eval.json"
    OUT="${ROOT}/corrected_runs/source_domain_risk_${PHASE}_v1/knowledge_ce"
    ;;
  *) echo "unknown dataset: ${DATASET_NAME}" >&2; exit 2 ;;
esac

mkdir -p "${OUT}"
CUDA_VISIBLE_DEVICES="${GPU}" python -u -m corrected_sgta.score_source_domain_risk \
  --dataset "${SUBSET}" \
  --output "${OUT}/risk_scores.jsonl" \
  --source-bank "${SOURCE_BANK}" \
  --local-prototypes "${PROTOTYPES}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-image-side 384 \
  --seed 42 \
  --question-type binary

python -u -m corrected_sgta.analyze_source_risk_routing \
  --risk-jsonl "${OUT}/risk_scores.jsonl" \
  --greedy-eval "${GREEDY}" \
  --mitigation-eval "${PAI}" \
  --output "${OUT}/risk_routing_analysis.json" \
  --mitigation-name PAI

echo "results: ${OUT}/risk_routing_analysis.json"

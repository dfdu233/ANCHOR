#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-smoke}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
ROOT="/root/autodl-tmp/Hulu-Med/MedUniEval"
SOURCE_BANK="${ROOT}/corrected_runs/source_bank_v2/source_bank.json"
DATASET="/root/autodl-tmp/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json"
OUT_ROOT="${ROOT}/corrected_runs/local_source_consensus_${PHASE}_v1"
PROTO="${OUT_ROOT}/llava_xray_local_prototypes.npz"
case "${PHASE}" in
  tiny) MAX_SAMPLES=4; PROTO_IMAGES="${PROTO_IMAGES:-8}" ;;
  smoke) MAX_SAMPLES=32; PROTO_IMAGES="${PROTO_IMAGES:-32}" ;;
  pilot) MAX_SAMPLES=128; PROTO_IMAGES="${PROTO_IMAGES:-64}" ;;
  *) echo "unknown phase: ${PHASE}" >&2; exit 2 ;;
esac
mkdir -p "${OUT_ROOT}"
if [[ ! -s "${PROTO}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" python -u -m corrected_sgta.build_local_source_prototypes \
    --model llava \
    --source-bank "${SOURCE_BANK}" \
    --output "${PROTO}" \
    --modality xray \
    --max-images-per-source "${PROTO_IMAGES}" \
    --max-tokens-per-source "${MAX_TOKENS_PER_SOURCE:-12000}" \
    --prototypes-per-source "${PROTOTYPES_PER_SOURCE:-48}" \
    --kmeans-iters "${KMEANS_ITERS:-6}" \
    --seed 42
fi
CACHE="${OUT_ROOT}/llava_cxr_vishal.jsonl"
ANALYSIS="${OUT_ROOT}/llava_cxr_vishal.analysis.json"
CUDA_VISIBLE_DEVICES="${GPU}" python -u -m corrected_sgta.infer_local_source_consensus \
  --model llava \
  --dataset "${DATASET}" \
  --output "${CACHE}" \
  --source-bank "${SOURCE_BANK}" \
  --local-prototypes "${PROTO}" \
  --max-samples "${MAX_SAMPLES}" \
  --max-image-side 384 \
  --seed 42 \
  --beta "${BETA:-0.25}" \
  --confidence-power "${CONFIDENCE_POWER:-2.0}" \
  --decode-max-new-tokens 8 \
  --question-type binary
python -u -m corrected_sgta.analyze_local_source_consensus \
  --cache "${CACHE}" \
  --output "${ANALYSIS}" \
  --laplacian-lambda-grid 0.1 0.3 1.0 3.0
echo "analysis: ${ANALYSIS}"

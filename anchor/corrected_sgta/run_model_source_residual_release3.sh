#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-llava}"; DOMAIN="${2:-iu}"; MAX_SAMPLES=32
GPU="${CUDA_VISIBLE_DEVICES:-0}"; ROOT="/root/autodl-tmp/Hulu-Med/MedUniEval"
SOURCE_BANK="${ROOT}/corrected_runs/source_bank_v2/source_bank.json"; CENTERS="${ROOT}/corrected_runs/source_bank_v2/visual_centers_${MODEL}.npz"; OUT="${ROOT}/corrected_runs/model_source_residual_smoke_release3"
case "${DOMAIN}" in iu) DATASET="/root/autodl-tmp/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/xray_closed_pairs.json" ;; mimic) DATASET="/root/autodl-tmp/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/mimic_cxr_closed_pairs.json" ;; *) exit 2 ;; esac
case "${MODEL}" in hulu|llava) ;; *) exit 2 ;; esac
mkdir -p "${OUT}"; PREFIX="${OUT}/${MODEL}_${DOMAIN}_n32"; CACHE="${PREFIX}.jsonl"; ANALYSIS="${PREFIX}_analysis.json"; AUDIT="${PREFIX}_audit.json"; FINAL="${PREFIX}_final.json"
CUDA_VISIBLE_DEVICES="${GPU}" python -u -m corrected_sgta.infer_model_source_residual_release3 --model "${MODEL}" --domain "${DOMAIN}" --dataset "${DATASET}" --output "${CACHE}" --source-bank "${SOURCE_BANK}" --visual-centers "${CENTERS}" --max-samples "${MAX_SAMPLES}" --max-image-side 384 --seed 42 --beta 0.5 --decode-max-new-tokens 8
python -u -m corrected_sgta.analyze_model_source_residual_release2 --cache "${CACHE}" --output "${ANALYSIS}" --laplacian-lambda-grid 0.1 0.3 1.0 3.0
python -u -m corrected_sgta.audit_model_source_residual_release3 --cache "${CACHE}" --source-bank "${SOURCE_BANK}" --output "${AUDIT}"
python -u -m corrected_sgta.adjudicate_model_source_residual_release2 --analysis "${ANALYSIS}" --audit "${AUDIT}" --output "${FINAL}"
echo "model-source residual final: ${FINAL}"

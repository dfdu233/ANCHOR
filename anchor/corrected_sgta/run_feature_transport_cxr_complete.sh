#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-smoke}"
MODEL="${2:-llava}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
ROOT="/root/autodl-tmp/Hulu-Med/MedUniEval"
SOURCE_BANK="${SOURCE_BANK:-${ROOT}/corrected_runs/source_bank_v1/source_bank.json}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/corrected_runs/feature_transport_${PHASE}_v1}"
case "${PHASE}" in smoke) MAX_SAMPLES=32 ;; pilot) MAX_SAMPLES=256 ;; *) exit 2 ;; esac
case "${MODEL}" in hulu) CENTER_BATCH=1 ;; llava) CENTER_BATCH=8 ;; *) exit 2 ;; esac
mkdir -p "${OUT_ROOT}"
VISUAL_CENTERS="${ROOT}/corrected_runs/source_bank_v1/visual_centers_release2_${MODEL}.npz"
CACHE="${OUT_ROOT}/${MODEL}_cxr_vishal.jsonl"
DIAGNOSTIC="${OUT_ROOT}/${MODEL}_cxr_vishal_diagnostic.json"
FROZEN="${OUT_ROOT}/${MODEL}_cxr_vishal_frozen_pre_structure.json"
STRUCTURE="${OUT_ROOT}/${MODEL}_cxr_vishal_structure.json"
FINAL="${OUT_ROOT}/${MODEL}_cxr_vishal_final.json"
if [[ ! -s "${VISUAL_CENTERS}" || ! -s "${VISUAL_CENTERS}.meta.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" python -u -m corrected_sgta.build_visual_centers_release2 \
    --model "${MODEL}" --source-bank "${SOURCE_BANK}" --output "${VISUAL_CENTERS}" \
    --max-per-source 64 --batch-size "${CENTER_BATCH}" --max-image-side 384 --seed 42
fi
CUDA_VISIBLE_DEVICES="${GPU}" python -u -m corrected_sgta.infer_feature_transport \
  --model "${MODEL}" \
  --dataset /root/autodl-tmp/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json \
  --output "${CACHE}" --source-bank "${SOURCE_BANK}" --visual-centers "${VISUAL_CENTERS}" \
  --max-samples "${MAX_SAMPLES}" --beta-grid 0.1 0.2 0.3 0.4 \
  --target-relative-closure 0.20 --decode-max-new-tokens 8 --seed 42
python -u -m corrected_sgta.analyze_feature_transport \
  --cache "${CACHE}" --output "${DIAGNOSTIC}" --laplacian-lambda-grid 0.1 0.3 1.0 3.0
python -u -m corrected_sgta.freeze_alignment_report_release4 \
  --cache "${CACHE}" --diagnostic-analysis "${DIAGNOSTIC}" --output "${FROZEN}"
python -u -m corrected_sgta.feature_transport_structure_audit \
  --cache "${CACHE}" --source-bank "${SOURCE_BANK}" --output "${STRUCTURE}"
python -u -m corrected_sgta.merge_feature_transport_gate_complete \
  --frozen-report "${FROZEN}" --structure-audit "${STRUCTURE}" --output "${FINAL}"
echo "feature transport final report: ${FINAL}"

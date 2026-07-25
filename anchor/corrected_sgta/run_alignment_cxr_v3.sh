#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-smoke}"
MODEL="${2:-llava}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
ROOT="/root/autodl-tmp/Hulu-Med/MedUniEval"
SOURCE_BANK="${SOURCE_BANK:-${ROOT}/corrected_runs/source_bank_v1/source_bank.json}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/corrected_runs/alignment_wave_a_${PHASE}_v3}"

case "${PHASE}" in
  smoke) MAX_SAMPLES=32 ;;
  pilot) MAX_SAMPLES=256 ;;
  *) echo "phase must be smoke or pilot" >&2; exit 2 ;;
esac
case "${MODEL}" in
  hulu) CENTER_BATCH=1 ;;
  llava) CENTER_BATCH=8 ;;
  *) echo "model must be hulu or llava" >&2; exit 2 ;;
esac

mkdir -p "${OUT_ROOT}"
VISUAL_CENTERS="${ROOT}/corrected_runs/source_bank_v1/visual_centers_v3_${MODEL}.npz"
CACHE="${OUT_ROOT}/${MODEL}_cxr_vishal.jsonl"
DIAGNOSTIC="${OUT_ROOT}/${MODEL}_cxr_vishal_diagnostic.json"
FROZEN="${OUT_ROOT}/${MODEL}_cxr_vishal_frozen.json"

if [[ ! -s "${VISUAL_CENTERS}" || ! -s "${VISUAL_CENTERS}.meta.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" python -u -m corrected_sgta.build_visual_centers_v3 \
    --model "${MODEL}" \
    --source-bank "${SOURCE_BANK}" \
    --output "${VISUAL_CENTERS}" \
    --max-per-source 64 \
    --batch-size "${CENTER_BATCH}" \
    --max-image-side 384 \
    --seed 42
fi

CUDA_VISIBLE_DEVICES="${GPU}" python -u -m corrected_sgta.infer_alignment_v3 \
  --model "${MODEL}" \
  --dataset /root/autodl-tmp/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json \
  --output "${CACHE}" \
  --source-bank "${SOURCE_BANK}" \
  --visual-centers "${VISUAL_CENTERS}" \
  --max-samples "${MAX_SAMPLES}" \
  --l-grid 0.01 0.03 0.05 0.1 0.2 \
  --source-ratio 0.0 \
  --max-views 2 \
  --min-relative-closure 0.0 \
  --min-style-psnr 20 \
  --min-edge-correlation 0.90 \
  --decode-labels \
  --decode-max-new-tokens 8 \
  --seed 42

python -u -m corrected_sgta.analyze_alignment_v2 \
  --cache "${CACHE}" \
  --output "${DIAGNOSTIC}" \
  --laplacian-lambda-grid 0.1 0.3 1.0 3.0

python -u -m corrected_sgta.freeze_alignment_report_v3 \
  --cache "${CACHE}" \
  --diagnostic-analysis "${DIAGNOSTIC}" \
  --output "${FROZEN}"

echo "alignment frozen report: ${FROZEN}"

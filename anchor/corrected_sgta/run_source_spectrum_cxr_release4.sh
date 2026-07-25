#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-llava}"; DOMAIN="${2:-iu}"; MAX_SAMPLES="${3:-32}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"; ROOT="/root/autodl-tmp/Hulu-Med/MedUniEval"
SOURCE_BANK="${ROOT}/corrected_runs/source_bank_v2/source_bank.json"
OUT_ROOT="${ROOT}/corrected_runs/source_spectrum_smoke_release4"
case "${MODEL}" in hulu) CENTER_BATCH=1 ;; llava) CENTER_BATCH=8 ;; *) exit 2 ;; esac
case "${DOMAIN}" in
  iu) DATASET="/root/autodl-tmp/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/xray_closed_pairs.json" ;;
  mimic) DATASET="/root/autodl-tmp/MedHEval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/mimic_cxr_closed_pairs.json" ;;
  *) exit 2 ;;
esac
mkdir -p "${OUT_ROOT}"
VISUAL_CENTERS="${ROOT}/corrected_runs/source_bank_v2/visual_centers_${MODEL}.npz"
PREFIX="${OUT_ROOT}/${MODEL}_${DOMAIN}_n${MAX_SAMPLES}"
CACHE="${PREFIX}.jsonl"; ANALYSIS="${PREFIX}_analysis.json"
STRUCTURE="${PREFIX}_structure.json"; FINAL="${PREFIX}_final.json"
if [[ ! -s "${VISUAL_CENTERS}" || ! -s "${VISUAL_CENTERS}.meta.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" python -u -m corrected_sgta.build_visual_centers_release2 \
    --model "${MODEL}" --source-bank "${SOURCE_BANK}" --output "${VISUAL_CENTERS}" \
    --max-per-source 64 --batch-size "${CENTER_BATCH}" --max-image-side 384 --seed 42
fi
CUDA_VISIBLE_DEVICES="${GPU}" python -u -m corrected_sgta.infer_alignment_source_spectrum_release4 \
  --model "${MODEL}" --dataset "${DATASET}" --output "${CACHE}" \
  --source-bank "${SOURCE_BANK}" --visual-centers "${VISUAL_CENTERS}" \
  --max-samples "${MAX_SAMPLES}" --l-grid 0.03 0.05 0.10 --source-ratio 0.0 \
  --max-views 1 --min-relative-closure -1000000000 --min-style-psnr 0 \
  --min-edge-correlation 0.90 --decode-labels --decode-max-new-tokens 8 --seed 42
python -u -m corrected_sgta.analyze_alignment_source_spectrum_release3 \
  --cache "${CACHE}" --output "${ANALYSIS}" --laplacian-lambda-grid 0.1 0.3 1.0 3.0
python -u -m corrected_sgta.structure_audit_source_spectrum_release2 \
  --cache "${CACHE}" --source-bank "${SOURCE_BANK}" --output "${STRUCTURE}"
python -u -m corrected_sgta.adjudicate_source_spectrum_release2 \
  --analysis "${ANALYSIS}" --structure "${STRUCTURE}" --output "${FINAL}"
echo "source-spectrum final: ${FINAL}"

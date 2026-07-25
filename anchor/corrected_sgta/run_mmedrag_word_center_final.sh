#!/usr/bin/env bash
set -euo pipefail

mode="${1:-validation}"
case "${mode}" in
  pilot)
    iu_n=16
    mimic_n=16
    harvard_n=16
    ;;
  validation)
    iu_n=128
    mimic_n=128
    harvard_n=128
    ;;
  full)
    iu_n=590
    mimic_n=694
    harvard_n=713
    ;;
  *)
    echo "usage: $0 {pilot|validation|full}" >&2
    exit 2
    ;;
esac

run_root="${RUN_ROOT:-corrected_runs/mmedrag_word_center_final_${mode}}"
source_centers="${SOURCE_CENTERS:-corrected_runs/mmedrag_sequence_anchor_v1/word_centers.json}"
manifest="${run_root}/manifest.json"
raw="${run_root}/predictions.raw.jsonl"
predictions="${run_root}/predictions.json"
metrics="${run_root}/metrics.json"
mkdir -p "${run_root}"

if [[ ! -f "${source_centers}" ]]; then
  python -m corrected_sgta.build_mmedrag_word_centers \
    --radiology-json /root/autodl-tmp/MMed-RAG/data/training/retriever/radiology/radiology_train.json \
    --iu-root /root/autodl-tmp/MedHEval/images/IU-Xray \
    --harvard-json /root/autodl-tmp/MMed-RAG/data/training/retriever/ophthalmology/harvard_train_7000.json \
    --harvard-root /root/autodl-tmp/source_data/FairVLMed/extracted/Training \
    --output "${source_centers}"
fi

if [[ ! -f "${manifest}" ]]; then
  python -m corrected_sgta.build_mmedrag_generation_manifest \
    --iuxray-json /root/autodl-tmp/MMed-RAG/data/test/report/iuxray_test.json \
    --iuxray-root /root/autodl-tmp/MedHEval/images/IU-Xray \
    --mimic-json /root/autodl-tmp/MMed-RAG/data/test/report/mimic_test.json \
    --mimic-root /root/autodl-tmp/MedHEval/images \
    --harvard-json /root/autodl-tmp/MMed-RAG/data/test/report/harvard_test.json \
    --harvard-root /root/autodl-tmp/source_data/FairVLMed/extracted/Test \
    --output "${manifest}" \
    --iuxray-samples "${iu_n}" \
    --mimic-samples "${mimic_n}" \
    --harvard-samples "${harvard_n}" \
    --seed 20260728
fi

if [[ ! -f "${predictions}" ]]; then
  python -m corrected_sgta.run_mmedrag_word_center_final \
    --manifest "${manifest}" \
    --source-centers "${source_centers}" \
    --raw "${raw}" \
    --output "${predictions}" \
    --max-new-tokens 160
fi

if [[ ! -f "${metrics}" ]]; then
  NLTK_DATA=/root/autodl-tmp/nltk_data \
    python -m corrected_sgta.evaluate_mmedrag_sequence_anchor \
      --input "${predictions}" \
      --output "${metrics}" \
      --bootstrap 10000
fi

clinical_python="/root/autodl-tmp/envs/medheval-report-eval/bin/python"
if [[ -x "${clinical_python}" ]]; then
  for variant in baseline source_word_center; do
    clinical_pairs="${run_root}/clinical_${variant}.jsonl"
    clinical_output="${run_root}/clinical_${variant}"
    if [[ ! -f "${clinical_output}/aggregate.json" ]]; then
      python -m corrected_sgta.export_mmedrag_clinical_pairs \
        --input "${predictions}" \
        --output "${clinical_pairs}" \
        --variant "${variant}"
      "${clinical_python}" -m corrected_sgta.evaluate_medheval_report_clinical \
        --input "${clinical_pairs}" \
        --output-dir "${clinical_output}" \
        --batch-size 8
    fi
  done
fi

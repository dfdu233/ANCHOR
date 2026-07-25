#!/usr/bin/env bash
set -euo pipefail

PHASE=${1:-pilot}
case "$PHASE" in
  pilot) MAX_SAMPLES=${MAX_SAMPLES:-16} ;;
  validation) MAX_SAMPLES=${MAX_SAMPLES:-256} ;;
  *) echo "usage: $0 pilot|validation" >&2; exit 2 ;;
esac

ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
DATA=/root/autodl-tmp/MedHEval/benchmark_data
OUT=${OUT:-$ROOT/corrected_runs/sgta_v4_wave1_stress_${PHASE}_v54}
mkdir -p "$OUT"
cd "$ROOT"

CXR="$DATA/Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json"
MM="$DATA/Visual_Misinterpretation_Hallucination/close-ended/MM-VisHal.json"

for model in hulu llava; do
  for task in cxr_vishal mm_vishal; do
    if [[ "$task" == cxr_vishal ]]; then dataset=$CXR; else dataset=$MM; fi
    cache="$OUT/${model}_${task}.jsonl"
    python -u -m corrected_sgta.infer_ce       --model "$model" --dataset "$dataset" --output "$cache"       --max-samples "$MAX_SAMPLES" --center-policy matched       --feddg-l-values 0.05 0.10 0.20       --feddg-source-ratios 0.0 0.5 0.8       --gammas 0.6 0.8 1.2 1.4       --min-style-psnr 15 --min-edge-correlation 0.85
    prototype="$ROOT/corrected_runs/full_v52/${model}_yes_no_scat_prototypes.npz"
    python -u -m corrected_sgta.analyze_sgta_v4       --cache "$cache" --output "$OUT/${model}_${task}.sgta_v4.json"       --prototypes "$prototype"       --min-psnr 15 --min-edge-correlation 0.85
  done
done

python -u -m corrected_sgta.summarize_wave_gates   --kind ce --input "$OUT" --output "$OUT/gates.json"

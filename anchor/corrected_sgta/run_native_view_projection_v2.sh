#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Hulu-Med/MedUniEval
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${OUT:-corrected_runs/native_view_projection_v2}"
N="${N:-128}"

run_safe() {
  local task="$1"
  python -u -m corrected_sgta.native_view_projection \
    --dataset "corrected_runs/medheval_mitigation_val512_v1/${task}/subset_seed42_n512.json" \
    --greedy-eval "corrected_runs/medheval_mitigation_val512_v1/${task}/greedy_seed42_tok16.eval.json" \
    --output "${OUT}/${task}/native_projection_n${N}.json" \
    --max-samples "${N}" --max-image-side 384 --question-type binary --train-frac 0.4 \
    --top-native 2 --max-views-forward 4 --min-psnr 18 --min-edge-correlation 0.85 \
    --transport-betas 0.05 0.1 0.2 0.4 --prototype-tokens-per-image 8
}

run_aggressive_cxr() {
  python -u -m corrected_sgta.native_view_projection \
    --dataset corrected_runs/medheval_mitigation_val512_v1/cxr_vishal/subset_seed42_n512.json \
    --greedy-eval corrected_runs/medheval_mitigation_val512_v1/cxr_vishal/greedy_seed42_tok16.eval.json \
    --output "${OUT}/cxr_vishal/native_projection_aggressive_n${N}.json" \
    --max-samples "${N}" --max-image-side 384 --question-type binary --train-frac 0.4 \
    --top-native 3 --max-views-forward 5 --min-psnr 15 --min-edge-correlation 0.75 \
    --l-values 0.01 0.03 0.06 0.1 --source-ratios 0.2 0.5 0.8 \
    --transport-betas 0.4 0.8 1.2 1.6 --transport-confidence-power 0.0 \
    --prototype-tokens-per-image 24 --fusion-temperature 0.1 --laplacian-lambda 1.0
}

run_safe cxr_vishal
run_safe knowledge_ce
# Uncomment only for diagnostics; not recommended as the paper method unless it shows rescues without harmful flips.
# run_aggressive_cxr

python -m corrected_sgta.analyze_native_view_projection \
  "${OUT}/cxr_vishal/native_projection_n${N}.json" \
  "${OUT}/knowledge_ce/native_projection_n${N}.json"

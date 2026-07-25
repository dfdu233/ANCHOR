#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Hulu-Med/MedUniEval
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${OUT:-corrected_runs/native_gradient_projection_v1}"
N="${N:-16}"

run_grad() {
  local task="$1"
  local target="$2"
  python -u -m corrected_sgta.native_gradient_projection \
    --dataset "corrected_runs/medheval_mitigation_val512_v1/${task}/subset_seed42_n512.json" \
    --greedy-eval "corrected_runs/medheval_mitigation_val512_v1/${task}/greedy_seed42_tok16.eval.json" \
    --output "${OUT}/${task}_selfconf_${target}_n${N}.json" \
    --max-samples "${N}" --max-image-side 384 --question-type binary --train-frac 0.4 \
    --support-mode self_conf --support-top-frac 0.8 --min-support 8 --max-support 24 \
    --target "${target}" --steps 10 --lr 0.01 --grid-size 8 --epsilon 0.03 --l2-weight 2.0 --tv-weight 0.2
}

run_grad cxr_vishal center
run_grad cxr_vishal nearest_support

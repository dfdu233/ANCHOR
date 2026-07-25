#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/Hulu-Med/MedUniEval
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${OUT:-corrected_runs/native_single_projection_v1}"
N="${N:-32}"
CAND="${CAND:-16}"

run_single() {
  local task="$1"
  local support="$2"
  python -u -m corrected_sgta.native_single_projection \
    --dataset "corrected_runs/medheval_mitigation_val512_v1/${task}/subset_seed42_n512.json" \
    --greedy-eval "corrected_runs/medheval_mitigation_val512_v1/${task}/greedy_seed42_tok16.eval.json" \
    --output "${OUT}/${task}_${support}_forced_n${N}_c${CAND}.json" \
    --max-samples "${N}" --max-image-side 384 --question-type binary --train-frac 0.4 \
    --support-mode "${support}" --support-top-frac 0.8 --min-support 8 --max-support 24 \
    --projection-mode forced --max-candidates "${CAND}"
}

run_single cxr_vishal self_conf
run_single knowledge_ce self_conf
run_single cxr_vishal competence

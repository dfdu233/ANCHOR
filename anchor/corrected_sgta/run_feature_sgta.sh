#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-validation}"
SEED="${2:-42}"
SELECTOR="${SELECTOR:-tim-kl-only}"
ITERATIONS="${ITERATIONS:-100}"
CACHE_ROOT="${CACHE_ROOT:-corrected_runs/sgta_v4_wave1_${PHASE}_v54}"
OUT_ROOT="${OUT_ROOT:-corrected_runs/feature_sgta_${PHASE}_v2_seed${SEED}}"

mkdir -p "${OUT_ROOT}"

echo "[feature-sgta] phase=${PHASE} seed=${SEED} selector=${SELECTOR}"
echo "[feature-sgta] cache=${CACHE_ROOT} out=${OUT_ROOT}"

for model in hulu llava; do
  for task in cxr_vishal mm_vishal; do
    cache="${CACHE_ROOT}/${model}_${task}.jsonl"
    proto="corrected_runs/full_v52/${model}_yes_no_scat_prototypes.npz"
    out="${OUT_ROOT}/${model}_${task}.json"
    if [[ ! -s "${cache}" ]]; then
      echo "[feature-sgta] missing cache: ${cache}" >&2
      exit 2
    fi
    if [[ ! -s "${proto}" ]]; then
      echo "[feature-sgta] missing prototypes: ${proto}" >&2
      exit 2
    fi
    python -u -m corrected_sgta.analyze_feature_sgta \
      --cache "${cache}" \
      --prototypes "${proto}" \
      --output "${out}" \
      --seed "${SEED}" \
      --selector "${SELECTOR}" \
      --iterations "${ITERATIONS}" \
      --feature-alpha-grid 0 0.1 0.25 0.5 0.75 1.0 \
      --temperature-grid 0.03 0.05 0.1 0.2 0.5
  done
done

echo "[feature-sgta] done: ${OUT_ROOT}"

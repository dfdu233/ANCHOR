#!/usr/bin/env bash
set -euo pipefail

PHASE=${1:-validation}
ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
IN=${IN:-$ROOT/corrected_runs/sgta_v4_wave1_${PHASE}_v54}
OUT=${OUT:-$ROOT/corrected_runs/sgta_v4_wave2_${PHASE}_v54}
mkdir -p "$OUT"
cd "$ROOT"

for model in hulu llava; do
  prototype="$ROOT/corrected_runs/full_v52/${model}_yes_no_scat_prototypes.npz"
  for task in cxr_vishal mm_vishal; do
    cache="$IN/${model}_${task}.jsonl"
    [[ -f "$cache" ]] || { echo "missing Wave-1 cache: $cache" >&2; exit 1; }
    python -u -m corrected_sgta.analyze_ce \
      --cache "$cache" --output "$OUT/${model}_${task}.baselines.json"
    python -u -m corrected_sgta.analyze_scat \
      --cache "$cache" --prototypes "$prototype" \
      --output "$OUT/${model}_${task}.scat.json"
    python -u -m corrected_sgta.analyze_sgta_v4 \
      --cache "$cache" --prototypes "$prototype" \
      --output "$OUT/${model}_${task}.sgta_v4.json"
  done
done
python -u -m corrected_sgta.summarize_wave_gates \
  --input "$OUT" --output "$OUT/gates.json"

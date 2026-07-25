#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
DATA=/root/autodl-tmp/MedHEval/benchmark_data
OUT="$ROOT/corrected_runs/full_v52"
mkdir -p "$OUT"
cd "$ROOT"

run_one() {
  local model="$1"
  local name="$2"
  local dataset="$3"
  local cache="$OUT/${model}_${name}.jsonl"
  python -u -m corrected_sgta.infer_ce \
    --model "$model" \
    --dataset "$dataset" \
    --output "$cache"
  python -u -m corrected_sgta.analyze_ce \
    --cache "$cache" \
    --output "$OUT/${model}_${name}.summary.json" \
    --transductive-window 256
}

for model in hulu llava; do
  run_one "$model" cxr_vishal \
    "$DATA/Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json"
  run_one "$model" mm_vishal \
    "$DATA/Visual_Misinterpretation_Hallucination/close-ended/MM-VisHal.json"
  run_one "$model" context \
    "$DATA/Context_Misalignment_Hallucination/MIMIC-CXR_pairs.json"
  run_one "$model" knowledge_ce \
    "$DATA/Knowledge_Deficiency_Hallucination/close-ended/MIMIC-CXR_sampled.json"
done

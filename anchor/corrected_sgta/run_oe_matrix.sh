#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
DATA=/root/autodl-tmp/MedHEval/benchmark_data
if [[ "$MODE" == "smoke" ]]; then
  OUT="$ROOT/corrected_runs/oe_smoke_v52"
  LIMIT=4
  CANDIDATES=4
elif [[ "$MODE" == "full" ]]; then
  OUT="$ROOT/corrected_runs/oe_full_v52"
  LIMIT=0
  CANDIDATES=8
else
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi
mkdir -p "$OUT"
cd "$ROOT"

run_one() {
  local model="$1"
  local name="$2"
  local dataset="$3"
  local tokens="$4"
  local cache="$OUT/${model}_${name}.jsonl"
  python -u -m corrected_sgta.infer_oe \
    --model "$model" \
    --dataset "$dataset" \
    --output "$cache" \
    --max-samples "$LIMIT" \
    --candidates "$CANDIDATES" \
    --candidate-batch 4 \
    --max-new-tokens "$tokens" \
    --style-augmentation
  python -u -m corrected_sgta.analyze_confgen \
    --cache "$cache" \
    --output "$OUT/${model}_${name}.confgen.json"
}

for model in hulu llava; do
  run_one "$model" knowledge_oe \
    "$DATA/Knowledge_Deficiency_Hallucination/open-ended/MIMIC-CXR_pairs.json" 128
  run_one "$model" report_oe \
    "$DATA/Visual_Misinterpretation_Hallucination/open-ended/MIMIC-CXR_pairs.json" 192
done

#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-pilot-generate}
case "$MODE" in
  pilot-generate|pilot-analyze) PHASE=pilot; KNOWLEDGE_MAX=${MAX_SAMPLES:-128}; REPORT_MAX=${REPORT_MAX_SAMPLES:-128}; CAL_FRAC=0.20; MIN_CAL=24 ;;
  validation-generate|validation-analyze) PHASE=validation; KNOWLEDGE_MAX=${MAX_SAMPLES:-640}; REPORT_MAX=${REPORT_MAX_SAMPLES:-490}; CAL_FRAC=0.30; MIN_CAL=128 ;;
  *) echo "usage: $0 pilot-generate|pilot-analyze|validation-generate|validation-analyze" >&2; exit 2 ;;
esac
ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
DATA=/root/autodl-tmp/MedHEval/benchmark_data
WAVE2=${WAVE2:-$ROOT/corrected_runs/sgta_v4_wave2_validation_v54/gates.json}
OUT=${OUT:-$ROOT/corrected_runs/sgta_confgen_v2_${PHASE}_v54}
mkdir -p "$OUT"
cd "$ROOT"

if [[ "${FORCE:-0}" != "1" ]]; then
  python - "$WAVE2" <<'PY_GATE'
import json,sys
p=sys.argv[1]
r=json.load(open(p))
if not r.get('passed'):
    raise SystemExit(f'Wave 2 gate failed; refusing OE generation: {p}')
PY_GATE
fi

KNOWLEDGE="$DATA/Knowledge_Deficiency_Hallucination/open-ended/MIMIC-CXR_pairs.json"
REPORT="$DATA/Visual_Misinterpretation_Hallucination/open-ended/MIMIC-CXR_pairs.json"

if [[ "$MODE" == *-generate ]]; then
  for model in hulu llava; do
    for task in knowledge report; do
      if [[ "$task" == knowledge ]]; then dataset=$KNOWLEDGE; task_max=$KNOWLEDGE_MAX; else dataset=$REPORT; task_max=$REPORT_MAX; fi
      cache="$OUT/${model}_${task}.jsonl"
      python -u -m corrected_sgta.infer_oe \
        --model "$model" --dataset "$dataset" --output "$cache" \
        --max-samples "$task_max" --candidates 8 --style-augmentation \
        --center-policy matched --max-style-views 2 --selector consistency \
        --feddg-l-values 0.01 0.03 --feddg-source-ratios 0.0 0.5
      python -u -m corrected_sgta.prepare_oe_judging \
        --cache "$cache" --task "$task" --seed 42 \
        --output "$OUT/${model}_${task}.blind.jsonl" \
        --manifest "$OUT/${model}_${task}.manifest.jsonl"
    done
  done
else
  : "${JUDGMENT_DIR:?Set JUDGMENT_DIR to completed clinical JSONL files}"
  : "${KNOWLEDGE_AGREEMENT_DIR:?Set KNOWLEDGE_AGREEMENT_DIR to per-model passing Qwen/human agreement reports}"
  : "${REPORT_METRIC_JSON:?Set REPORT_METRIC_JSON to the passing local report-metric validation report}"
  for model in hulu llava; do
    for task in knowledge report; do
      if [[ "$task" == knowledge ]]; then
        evidence_gate="$KNOWLEDGE_AGREEMENT_DIR/${model}_knowledge.agreement.json"
      else
        evidence_gate=$REPORT_METRIC_JSON
      fi
      python -u -m corrected_sgta.analyze_confgen_v2 \
        --cache "$OUT/${model}_${task}.jsonl" --task "$task" \
        --admissibility clinical \
        --judgments "$JUDGMENT_DIR/${model}_${task}.jsonl" \
        --judge-agreement "$evidence_gate" \
        --router-fraction 0.20 --calibration-fraction "$CAL_FRAC" \
        --min-proper-calibration "$MIN_CAL" --candidate-budget 8 --gamma 0.90 \
        --output "$OUT/${model}_${task}.confgen_v2.json"
    done
  done
  python -u -m corrected_sgta.summarize_wave_gates \
    --kind confgen --input "$OUT" --output "$OUT/gates.json"
fi

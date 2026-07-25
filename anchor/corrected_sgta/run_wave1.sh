#!/usr/bin/env bash
set -euo pipefail

PHASE=${1:-pilot}
case "$PHASE" in
  pilot) MAX_SAMPLES=${MAX_SAMPLES:-16}; DECODE="--decode-labels --decode-max-new-tokens 24" ;;
  validation) MAX_SAMPLES=${MAX_SAMPLES:-256}; DECODE= ;;
  full) MAX_SAMPLES=${MAX_SAMPLES:-0}; DECODE= ;;
  *) echo "usage: $0 pilot|validation|full" >&2; exit 2 ;;
esac
ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
DATA=/root/autodl-tmp/MedHEval/benchmark_data
OUT=${OUT:-$ROOT/corrected_runs/sgta_v4_wave1_${PHASE}_v54}
mkdir -p "$OUT"
cd "$ROOT"

if [[ "$PHASE" == validation ]]; then
  PILOT_READINESS=${PILOT_READINESS:-$ROOT/corrected_runs/sgta_v4_wave1_pilot_v54/readiness.json}
  python - "$PILOT_READINESS" <<'PY_GATE'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get("gates", {}).get("ce_interface_locked"):
    raise SystemExit(f"CE pilot interface gate failed: {sys.argv[1]}")
PY_GATE
fi

CXR="$DATA/Visual_Misinterpretation_Hallucination/close-ended/CXR-VisHal.json"
MM="$DATA/Visual_Misinterpretation_Hallucination/close-ended/MM-VisHal.json"

for model in hulu llava; do
  for task in cxr_vishal mm_vishal; do
    if [[ "$task" == cxr_vishal ]]; then dataset=$CXR; else dataset=$MM; fi
    cache="$OUT/${model}_${task}.jsonl"
    python -u -m corrected_sgta.infer_ce \
      --model "$model" --dataset "$dataset" --output "$cache" \
      --max-samples "$MAX_SAMPLES" --center-policy matched \
      --feddg-l-values 0.003 0.01 0.03 \
      --feddg-source-ratios 0.0 0.5 0.8 \
      --gammas 0.8 1.2 --min-style-psnr 20 --min-edge-correlation 0.90 \
      $DECODE
    prototype="$ROOT/corrected_runs/full_v52/${model}_yes_no_scat_prototypes.npz"
    python -u -m corrected_sgta.analyze_sgta_v4 \
      --cache "$cache" --output "$OUT/${model}_${task}.sgta_v4.json" \
      --prototypes "$prototype"
  done
done

if [[ "$PHASE" == pilot ]]; then
  python -u -m corrected_sgta.audit_wave0 \
    --output "$OUT/readiness.json" \
    --ce-cache \
      "$OUT/hulu_cxr_vishal.jsonl" "$OUT/hulu_mm_vishal.jsonl" \
      "$OUT/llava_cxr_vishal.jsonl" "$OUT/llava_mm_vishal.jsonl"
  python - "$OUT/readiness.json" <<'PY_GATE'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get("gates", {}).get("ce_interface_locked"):
    raise SystemExit("CE pilot interface gate failed; validation is blocked")
PY_GATE
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
OUT=${OUT:-$ROOT/corrected_runs/sgta_v4_wave0_v54}
QWEN=${QWEN:-/root/autodl-tmp/hf_hub_cache/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac}
mkdir -p "$OUT"
cd "$ROOT"

CE=(
  corrected_runs/optimized_ce_probe256_v53/hulu_cxr_vishal.jsonl
  corrected_runs/optimized_ce_probe256_v53/hulu_mm_vishal.jsonl
  corrected_runs/optimized_ce_probe256_v53/llava_cxr_vishal.jsonl
  corrected_runs/optimized_ce_probe256_v53/llava_mm_vishal.jsonl
)
python -u -m corrected_sgta.audit_wave0 \
  --output "$OUT/readiness.json" --qwen-snapshot "$QWEN" --ce-cache "${CE[@]}"

for model in hulu llava; do
  for task in knowledge report; do
    cache="corrected_runs/optimized_oe_probe32_v53/${model}_${task}_oe.jsonl"
    python -u -m corrected_sgta.prepare_oe_judging \
      --cache "$cache" --task "$task" --max-items 100 --seed 42 \
      --output "$OUT/${model}_${task}.blind.jsonl" \
      --manifest "$OUT/${model}_${task}.manifest.jsonl"
  done
done

if [[ "${RUN_QWEN:-0}" == "1" ]]; then
  for model in hulu llava; do
    python -u -m corrected_sgta.judge_knowledge_local \
      --input "$OUT/${model}_knowledge.blind.jsonl" \
      --output "$OUT/${model}_knowledge.qwen.jsonl" --model "$QWEN"
  done
fi

if [[ -n "${ANNOTATOR_LEFT:-}" && -n "${ANNOTATOR_RIGHT:-}" ]]; then
  python -u -m corrected_sgta.analyze_judge_agreement \
    --left "$ANNOTATOR_LEFT" --right "$ANNOTATOR_RIGHT" \
    --output "$OUT/knowledge_judge_agreement.json"
fi

python - "$OUT/readiness.json" <<'PY_GATE'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get("gates", {}).get("passed"):
    raise SystemExit("Wave 0 readiness failed; inspect readiness.json (blind exports were still created)")
PY_GATE

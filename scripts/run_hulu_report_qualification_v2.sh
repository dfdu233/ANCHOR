#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
export PYTHONPATH=anchor
root=corrected_runs/unified_eval/smoke/hulu_mimic_report_v2
/home/dbw/.venvs/hulumed/bin/python \
  -m corrected_sgta.infer_oe \
  --model hulu \
  --dataset corrected_runs/high_efficiency/full_generation_mmedrag_mimic_report_20260726/mmedrag/mimic/report_generation/greedy/chunk_0000.questions.json \
  --output "$root/predictions.jsonl" \
  --max-samples 32 \
  --candidates 1 \
  --candidate-batch 1 \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-new-tokens 160 \
  --seed 42 \
  --report-prompt-mode official_zero_shot

/home/dbw/.venvs/hulumed/bin/python \
  -m corrected_sgta.run_oe_sanity_audit \
  --analyze-existing "$root/predictions.jsonl" \
  --output-dir "$root/sanity_audit"

admissible=$(/home/dbw/.venvs/hulumed/bin/python -c \
  'import json,sys; print("yes" if json.load(open(sys.argv[1]))["admissible_for_report_generation_claim"] else "no")' \
  "$root/sanity_audit/summary.json")
if [[ "$admissible" != "yes" ]]; then
  echo "Hulu report smoke failed the frozen admissibility gate; full run skipped"
  exit 0
fi

/home/dbw/.venvs/hulumed/bin/python \
  -m corrected_sgta.infer_oe \
  --model hulu \
  --dataset corrected_runs/high_efficiency/full_generation_mmedrag_mimic_report_20260726/mmedrag/mimic/report_generation/greedy/chunk_0000.questions.json \
  --output corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v2/predictions.jsonl \
  --max-samples 0 \
  --candidates 1 \
  --candidate-batch 1 \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-new-tokens 160 \
  --seed 42 \
  --report-prompt-mode official_zero_shot

/home/dbw/.venvs/hulumed/bin/python \
  -m corrected_sgta.run_oe_sanity_audit \
  --analyze-existing corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v2/predictions.jsonl \
  --output-dir corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v2/sanity_audit

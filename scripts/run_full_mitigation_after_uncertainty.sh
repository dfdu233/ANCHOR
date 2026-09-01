#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
state=corrected_runs/detached_jobs/hulu-mimic-uncertainty-v1.json
while true; do
  status=$(/opt/miniconda3/envs/huatuo/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$state")
  [[ "$status" == "done" ]] && break
  if [[ "$status" == "failed" ]]; then
    echo "Hulu uncertainty screen failed" >&2
    exit 1
  fi
  sleep 30
done

export PYTHONPATH=anchor
analysis=corrected_runs/missing_third_state/mimic_report_triplets_v1/hulu_scores_v1/dev_analysis.json
if [[ ! -s "$analysis" ]]; then
  /opt/miniconda3/envs/huatuo/bin/python \
    -m corrected_sgta.analyze_mimic_uncertainty_triplets \
    --manifest corrected_runs/missing_third_state/mimic_report_triplets_v1/manifest.json \
    --scores corrected_runs/missing_third_state/mimic_report_triplets_v1/hulu_scores_v1/raw.jsonl \
    --split dev \
    --output "$analysis"
fi

smoke_root=corrected_runs/unified_eval/smoke/hulu_mimic_report_v1
if ! /usr/bin/env PYTHONPATH=anchor /home/dbw/.venvs/hulumed/bin/python \
  -m corrected_sgta.infer_oe \
  --model hulu \
  --dataset corrected_runs/high_efficiency/full_generation_mmedrag_mimic_report_20260726/mmedrag/mimic/report_generation/greedy/chunk_0000.questions.json \
  --output "$smoke_root/predictions.jsonl" \
  --max-samples 32 \
  --candidates 1 \
  --candidate-batch 1 \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-new-tokens 160 \
  --seed 42 \
  --report-prompt-mode official_zero_shot; then
  echo "Hulu report qualification smoke failed" >&2
else
  /usr/bin/env PYTHONPATH=anchor /home/dbw/.venvs/hulumed/bin/python \
    -m corrected_sgta.run_oe_sanity_audit \
    --analyze-existing "$smoke_root/predictions.jsonl" \
    --output-dir "$smoke_root/sanity_audit"
  admissible=$(/home/dbw/.venvs/hulumed/bin/python -c \
    'import json,sys; print("yes" if json.load(open(sys.argv[1]))["admissible_for_report_generation_claim"] else "no")' \
    "$smoke_root/sanity_audit/summary.json")
  if [[ "$admissible" == "yes" ]]; then
    /usr/bin/env PYTHONPATH=anchor /home/dbw/.venvs/hulumed/bin/python \
      -m corrected_sgta.infer_oe \
      --model hulu \
      --dataset corrected_runs/high_efficiency/full_generation_mmedrag_mimic_report_20260726/mmedrag/mimic/report_generation/greedy/chunk_0000.questions.json \
      --output corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v1/predictions.jsonl \
      --max-samples 0 \
      --candidates 1 \
      --candidate-batch 1 \
      --temperature 0.7 \
      --top-p 0.9 \
      --max-new-tokens 160 \
      --seed 42 \
      --report-prompt-mode official_zero_shot
    /usr/bin/env PYTHONPATH=anchor /home/dbw/.venvs/hulumed/bin/python \
      -m corrected_sgta.run_oe_sanity_audit \
      --analyze-existing corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v1/predictions.jsonl \
      --output-dir corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v1/sanity_audit
  else
    echo "Hulu report smoke failed the frozen admissibility gate; full run skipped" >&2
  fi
fi

export PYTHONPATH=.
methods=$(/opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.medeval.select_mitigation_smoke \
  --smoke-root corrected_runs/unified_eval/smoke/mitigation_matrix_v1 \
  --output corrected_runs/unified_eval/smoke/mitigation_matrix_v1/selection.json)
echo "report smoke implementation-qualified methods: ${methods:-none}"
echo "Full LLaVA report mitigation matrix intentionally skipped: base model-task qualification failed."

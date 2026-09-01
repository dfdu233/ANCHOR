#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
state=corrected_runs/detached_jobs/llava-mimic-report-full-greedy-v1.json
while true; do
  status=$(/opt/miniconda3/envs/huatuo/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$state")
  if [[ "$status" == "done" ]]; then
    break
  fi
  if [[ "$status" == "failed" ]]; then
    echo "upstream LLaVA full run failed; mitigation smoke not started" >&2
    exit 1
  fi
  sleep 30
done

export PYTHONPATH=anchor
/opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.run_oe_sanity_audit \
  --analyze-existing corrected_runs/unified_eval/full/llava_mimic_report_greedy_v1/predictions.jsonl \
  --output-dir corrected_runs/unified_eval/full/llava_mimic_report_greedy_v1/sanity_audit

export ANCHOR_MODEL_PATH=/home/dbw/models/LLaVA-Med-v1.5-mistral-7b
exec /opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.run_llava_med_generation_matrix \
  --question-file corrected_runs/high_efficiency/full_generation_mmedrag_mimic_report_20260726/mmedrag/mimic/report_generation/greedy/chunk_0000.questions.json \
  --image-folder data/medheval/images \
  --out corrected_runs/unified_eval/smoke/mitigation_matrix_v1 \
  --source mmedrag \
  --dataset mimic \
  --task report_generation \
  --methods greedy beam DoLa PAI opera avisc m3id VCD damro \
  --chunk-size 4 \
  --limit 4 \
  --max-new-tokens 160 \
  --conv-mode mistral_instruct \
  --seed 42 \
  --continue-on-error

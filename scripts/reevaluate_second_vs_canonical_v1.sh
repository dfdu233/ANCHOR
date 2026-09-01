#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
upstream=corrected_runs/detached_jobs/llava-native-vqa-rad-oe-v1.json
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done
if [[ "$status" != "done" ]]; then
  echo "Canonical LLaVA OE baseline unavailable" >&2
  exit 2
fi

second=corrected_runs/unified_eval/full/second_vqa_rad_oe_v1/answers.jsonl
canonical=corrected_runs/unified_eval/full/llava_native_vqa_rad_oe_v1/answers.jsonl
if [[ ! -f "$second" ]] || [[ $(wc -l < "$second") -ne 200 ]]; then
  echo "SECOND full answers unavailable or incomplete" >&2
  exit 2
fi
export PYTHONPATH=anchor
/opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.evaluate_oe_vqa \
  --manifest corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json \
  --answers "$second" --baseline-answers "$canonical" \
  --output corrected_runs/unified_eval/full/second_vqa_rad_oe_v1/evaluation_vs_canonical.json \
  --bootstrap-replicates 5000 --seed 42

#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
OUT="$ROOT/corrected_runs/full_v52"
cd "$ROOT"

for model in hulu llava; do
  for name in cxr_vishal mm_vishal context knowledge_ce; do
    cache="$OUT/${model}_${name}.jsonl"
    if [[ ! -f "$cache" ]]; then
      echo "Missing CE cache: $cache" >&2
      exit 1
    fi
    python -u -m corrected_sgta.tune_sgta \
      --cache "$cache" \
      --output "$OUT/${model}_${name}.sgta_tuned.json"
  done
done

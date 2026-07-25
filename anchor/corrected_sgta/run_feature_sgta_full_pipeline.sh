#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
cd "${ROOT}"

echo "[pipeline] starting full Wave-1 cache generation: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
corrected_sgta/run_wave1.sh full

echo "[pipeline] starting feature-SGTA full analysis: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for seed in 42 43 44; do
  corrected_sgta/run_feature_sgta.sh full "${seed}"
done

python -u -m corrected_sgta.summarize_feature_sgta \
  --input-dir corrected_runs/feature_sgta_full_v2_seed42 \
  --input-dir corrected_runs/feature_sgta_full_v2_seed43 \
  --input-dir corrected_runs/feature_sgta_full_v2_seed44 \
  --output corrected_runs/feature_sgta_full_v2_multiseed_summary.json

echo "[pipeline] done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

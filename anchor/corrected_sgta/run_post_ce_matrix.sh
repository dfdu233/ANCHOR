#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
cd "$ROOT"

corrected_sgta/run_sgta_tuned.sh
corrected_sgta/run_scat.sh
corrected_sgta/run_ce_confgen.sh
corrected_sgta/run_oe_matrix.sh smoke
corrected_sgta/run_oe_matrix.sh full

python -u -m corrected_sgta.summarize_results \
  --ce-dir "$ROOT/corrected_runs/full_v52" \
  --oe-dir "$ROOT/corrected_runs/oe_full_v52" \
  --output "$ROOT/corrected_runs/corrected_v52_aggregate.json"

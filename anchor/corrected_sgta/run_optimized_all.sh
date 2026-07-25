#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
cd "$ROOT"

bash corrected_sgta/run_optimized_ce.sh "$MODE"
bash corrected_sgta/run_optimized_oe.sh "$MODE"

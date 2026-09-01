#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR

if [[ $# -ne 2 ]]; then
  echo "usage: $0 /absolute/scheduler_handoff.json EXPECTED_SHA256" >&2
  exit 2
fi

# This entrypoint is never called by the return monitor or handoff builder.
# Invocation is the explicit detached-execution decision after both upstream
# hashes and the genuine human admission have already produced the handoff.
PYTHONPATH=.:anchor exec .venv-full/bin/python -u -m \
  corrected_sgta.run_vindr_cecd_listing_pipeline_v1 execute \
  --handoff "$1" --expected-handoff-sha256 "$2"

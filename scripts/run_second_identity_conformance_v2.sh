#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR

# Serialize GPU-heavy OE work.  The native Hulu baseline starts after the
# canonical LLaVA baseline and is the final predecessor in that queue.
upstream=corrected_runs/detached_jobs/hulu-native-vqa-rad-oe-v1.json
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done
if [[ "$status" != "done" ]]; then
  echo "Native Hulu OE predecessor failed; SECOND identity audit not started" >&2
  exit 2
fi

# The underlying v1 protocol and artifact paths stay frozen.  v2 denotes the
# rerun after fixing SECOND's CHW/NCHW interpolation bug; the failed v1 job
# ledger remains untouched for provenance.
exec bash scripts/run_second_identity_conformance_v1.sh

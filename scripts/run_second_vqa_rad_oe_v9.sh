#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
upstream=${SECOND_IDENTITY_UPSTREAM:-corrected_runs/detached_jobs/second-identity-conformance-v2.json}
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done
if [[ "$status" != "done" ]]; then
  echo "SECOND backend identity conformance did not complete successfully" >&2
  exit 2
fi

/opt/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("corrected_runs/unified_eval/sanity/second_identity_conformance_v1/conformance.json")
if not path.exists() or not json.loads(path.read_text()).get("passed", False):
    raise SystemExit("SECOND backend identity conformance did not pass")
PY

# v9 is the first recursive rerun containing the NCHW pyramid fix.  The v8
# failed ledger and log remain immutable; successful answers retain the frozen
# v1 evaluation-protocol directory name.
exec bash scripts/run_second_vqa_rad_oe_v1.sh

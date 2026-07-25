#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Hulu-Med/MedUniEval
GATE=${GATE:-$ROOT/corrected_runs/sgta_confgen_v2_validation_v54/gates.json}
[[ -f "$GATE" ]] || { echo "missing frozen Wave-3 gate report: $GATE" >&2; exit 1; }
python - "$GATE" <<'PY_GATE'
import json,sys
r=json.load(open(sys.argv[1]))
if not r.get('passed'):
    raise SystemExit('Wave 3 gate failed; full experiment and paper editing are blocked')
PY_GATE

echo "Wave 3 passed. Freeze config/commit, then run:"
echo "  corrected_sgta/run_wave1.sh full"
echo "  MAX_SAMPLES=0 corrected_sgta/run_wave3.sh validation-generate"
echo "Run three seeds via the experiment queue only after fingerprints are frozen."

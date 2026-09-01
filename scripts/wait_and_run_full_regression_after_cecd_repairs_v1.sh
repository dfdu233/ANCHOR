#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
heartbeat=corrected_runs/detached_jobs/full-regression-after-cecd-repairs-v1.heartbeat.json
probe_log=corrected_runs/detached_jobs/full-regression-after-cecd-repairs-v1.probe.log
mkdir -p corrected_runs/detached_jobs

while true; do
  set +e
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
    .venv-full/bin/pytest -q \
      tests/test_cecd_reader_threshold_alias_sensitivity_v1.py \
      tests/test_cecd_stage1_power_audit_v1.py \
      tests/test_verify_cecd_three_stage_v3.py \
      >"$probe_log" 2>&1
  code=$?
  set -e
  .venv-full/bin/python - "$heartbeat" "$probe_log" "$code" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "version": "full-regression-cecd-repair-gate-v1",
    "time": datetime.now(timezone.utc).isoformat(),
    "probe_log": str(Path(sys.argv[2]).resolve()),
    "focused_pytest_exit_code": int(sys.argv[3]),
    "stage": "launching_full_regression" if int(sys.argv[3]) == 0 else "waiting_for_companion_cecd_repairs",
    "companion_sources_modified": False,
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(temporary, path)
PY
  if [[ "$code" -eq 0 ]]; then
    break
  fi
  sleep 60
done

export REGRESSION_ARTIFACT=corrected_runs/unified_eval/provenance/full_regression_after_llava_t3_chain_v2.json
export REGRESSION_LOG_BASE=corrected_runs/unified_eval/provenance/full_regression_after_llava_t3_chain_v2
exec bash scripts/run_full_regression_quiescent_after_llava_t3_v1.sh

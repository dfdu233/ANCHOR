#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

cd /home/dbw/ANCHOR
artifact=${REGRESSION_ARTIFACT:-corrected_runs/unified_eval/provenance/full_regression_after_llava_t3_chain_v1.json}
log_base=${REGRESSION_LOG_BASE:-corrected_runs/unified_eval/provenance/full_regression_after_llava_t3_chain_v1}
mkdir -p "$(dirname "$artifact")"

fingerprint_source() {
  rg --files anchor tests scripts configs \
    | LC_ALL=C sort \
    | xargs sha256sum \
    | sha256sum \
    | awk '{print $1}'
}

for attempt in 1 2 3 4 5; do
  candidate=$(fingerprint_source)
  sleep 12
  started=$(fingerprint_source)
  if [[ "$candidate" != "$started" ]]; then
    continue
  fi
  started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  log=${log_base}.attempt-${attempt}.log
  set +e
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
    .venv-full/bin/pytest -q 2>&1 | tee "$log"
  pytest_code=${PIPESTATUS[0]}
  set -e
  finished=$(fingerprint_source)
  finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  source_quiescent=false
  if [[ "$started" = "$finished" ]]; then
    source_quiescent=true
  fi
  .venv-full/bin/python - "$artifact" "$started" "$finished" "$pytest_code" \
    "$started_at" "$finished_at" "$log" "$attempt" "$source_quiescent" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "version": "source-quiescent-full-regression-v3",
    "scope": "post LLaVA T3 label-firewall, trace audit, clinical contract, and physician continuation chain",
    "started_source_fingerprint": sys.argv[2],
    "finished_source_fingerprint": sys.argv[3],
    "pytest_exit_code": int(sys.argv[4]),
    "started_at": sys.argv[5],
    "finished_at": sys.argv[6],
    "log": str(Path(sys.argv[7]).resolve()),
    "attempt": int(sys.argv[8]),
    "source_quiescent": sys.argv[9] == "true",
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(record, indent=2) + "\n")
temporary.replace(path)
print(json.dumps(record, indent=2))
PY
  if [[ "$source_quiescent" = true ]]; then
    exit "$pytest_code"
  fi
done

echo "no source-quiescent regression window after five attempts" >&2
exit 3

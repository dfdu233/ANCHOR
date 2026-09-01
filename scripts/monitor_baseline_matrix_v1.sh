#!/usr/bin/env bash
set -uo pipefail
root=/home/dbw/ANCHOR
run_root="$root/corrected_runs/paper_baselines_v1/full_matrix_v1"
audit="$run_root/coverage_audit.json"
history="$run_root/monitor_history.jsonl"
cd "$root"
mkdir -p "$run_root/paper_tables"

while true; do
  if ! PYTHONPATH=. /opt/miniconda3/bin/python \
    -m anchor.medeval.audit_baseline_matrix_execution_v1 \
    --output "$audit" >"$run_root/coverage_audit.log" 2>&1; then
    printf '%s coverage audit failed; see %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_root/coverage_audit.log" >&2
  fi
  PYTHONPATH=. /opt/miniconda3/bin/python -m anchor.medeval.export_baseline_paper_tables_v1 \
    --coverage "$audit" --output-dir "$run_root/paper_tables" --allow-incomplete \
    >"$run_root/paper_tables/export.log" 2>&1 || true
  /opt/miniconda3/bin/python - "$audit" "$history" <<'PY'
import datetime,json,subprocess,sys
from pathlib import Path
audit_path,history_path=map(Path,sys.argv[1:])
if not audit_path.is_file():
    raise SystemExit(0)
audit=json.loads(audit_path.read_text())
try:
    gpu=subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],
        text=True,stderr=subprocess.DEVNULL).strip().splitlines()
except Exception:
    gpu=[]
try:
    sessions=subprocess.check_output(['tmux','list-sessions','-F','#{session_name}'],text=True).splitlines()
except Exception:
    sessions=[]
row={
    'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'summary':audit['summary'],
    'complete':audit['complete'],
    'gpu_processes':gpu,
    'tmux_sessions':sorted(s for s in sessions if s.startswith('baseline_')),
}
with history_path.open('a') as handle:
    handle.write(json.dumps(row,sort_keys=True)+'\n')
PY
  if /opt/miniconda3/bin/python - "$audit" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['complete'] else 1)
PY
  then
    exit 0
  fi
  sleep 60
done

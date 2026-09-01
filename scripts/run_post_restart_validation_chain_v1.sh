#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=.

python_bin=.venv-full/bin/python
output_dir=corrected_runs/paper/iclr_oral_completion_audit_v1

[[ -x "$python_bin" ]] || {
  echo "persistent validation runtime is missing: $python_bin" >&2
  exit 4
}
[[ -s /home/dbw/.anchor_persistent_volume_v1.json ]] || {
  echo "persistent-volume sentinel is missing" >&2
  exit 4
}
findmnt -T /home/dbw >/dev/null

source_snapshot() {
  find anchor scripts tests docs \
    -type f \
    \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.json' \) \
    -printf '%T@ %s %p\n' \
    | LC_ALL=C sort \
    | sha256sum \
    | cut -d ' ' -f 1
}

wait_for_source_quiescence() {
  local required_polls=${SOURCE_QUIESCENCE_POLLS:-6}
  local poll_seconds=${SOURCE_QUIESCENCE_POLL_SECONDS:-2}
  local maximum_polls=${SOURCE_QUIESCENCE_MAX_POLLS:-450}
  local prior current stable_polls
  prior=$(source_snapshot)
  stable_polls=0
  for poll in $(seq 1 "$maximum_polls"); do
    sleep "$poll_seconds"
    current=$(source_snapshot)
    if [[ "$current" == "$prior" ]]; then
      stable_polls=$((stable_polls + 1))
      if [[ "$stable_polls" -ge "$required_polls" ]]; then
        echo "source quiescent snapshot=$current stable_polls=$stable_polls"
        return 0
      fi
    else
      echo "source activity observed poll=$poll before=$prior after=$current; waiting"
      prior=$current
      stable_polls=0
    fi
  done
  echo "source did not become quiescent within $maximum_polls polls" >&2
  return 1
}

# The companion session may still be finishing a CPU audit.  A test run whose
# source changes underneath it is repeated, never promoted as a clean result.
stable=0
for attempt in 1 2 3 4 5 6 7 8; do
  wait_for_source_quiescence || exit 6
  before=$(source_snapshot)
  echo "validation attempt=$attempt source_snapshot=$before"
  "$python_bin" -m pytest -q
  after=$(source_snapshot)
  if [[ "$before" == "$after" ]]; then
    stable=1
    break
  fi
  echo "source drifted during validation: before=$before after=$after; repeating"
done
[[ "$stable" -eq 1 ]] || {
  echo "source did not stabilize across four complete validation attempts" >&2
  exit 5
}

mkdir -p "$output_dir"
temporary_root=$(mktemp -d /home/dbw/ANCHOR/corrected_runs/paper/.completion-audit-verify.XXXXXX)
trap 'rm -rf -- "$temporary_root"' EXIT

"$python_bin" -m anchor.corrected_sgta.build_iclr_oral_completion_audit_v1 \
  --root /home/dbw/ANCHOR \
  --output-dir "$output_dir"
"$python_bin" -m anchor.corrected_sgta.build_iclr_oral_completion_audit_v1 \
  --root /home/dbw/ANCHOR \
  --output-dir "$temporary_root"

cmp "$output_dir/audit.json" "$temporary_root/audit.json"
cmp "$output_dir/EVIDENCE.md" "$temporary_root/EVIDENCE.md"
cmp "$output_dir/manifest.json" "$temporary_root/manifest.json"

"$python_bin" scripts/research_watchdog.py \
  --once \
  --heartbeat corrected_runs/detached_jobs/post-restart-validation-watchdog-once.json
"$python_bin" scripts/research_status.py

echo "POST_RESTART_VALIDATION_AND_CONTINUATION_HANDOFF_COMPLETE"

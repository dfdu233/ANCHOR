#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
eval_python=/opt/miniconda3/envs/huatuo/bin/python
collection_state=corrected_runs/detached_jobs/vindr-v2-confirmation-collections-v3.json
analysis_root=corrected_runs/vindr_v2/reader_residual_confirmation_v1

while true; do
  status="$($eval_python - "$collection_state" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
print(json.load(p.open()).get("status", "missing") if p.is_file() else "missing")
PY
)"
  case "$status" in
    done) break ;;
    failed) echo "confirmation collection failed; analysis is fail-closed" >&2; exit 2 ;;
    starting|running|missing) sleep 30 ;;
    *) echo "invalid confirmation collection state: $status" >&2; exit 2 ;;
  esac
done

mkdir -p "$analysis_root"

PYTHONPATH=anchor "$eval_python" -m corrected_sgta.confirm_reader_residual_v1 \
  --dev corrected_runs/vindr_v2/hidden_dev_huatuo_all_findings_v3 \
  --confirmation corrected_runs/vindr_v2/hidden_confirmation_huatuo_all_findings_v1 \
  --lock corrected_runs/vindr_v2/reader_residual_locks/huatuo_v1.json \
  --output "$analysis_root/huatuo_v1.json" --draws 5000 --seed 42

PYTHONPATH=anchor "$eval_python" -m corrected_sgta.confirm_reader_residual_v1 \
  --dev corrected_runs/vindr_v2/hidden_dev_hulu_all_findings_v1 \
  --confirmation corrected_runs/vindr_v2/hidden_confirmation_hulu_all_findings_v1 \
  --lock corrected_runs/vindr_v2/reader_residual_locks/hulu_v1.json \
  --output "$analysis_root/hulu_v1.json" --draws 5000 --seed 42

PYTHONPATH=anchor "$eval_python" -m corrected_sgta.summarize_reader_residual_confirmation_v1 \
  --input "$analysis_root/huatuo_v1.json" \
  --input "$analysis_root/hulu_v1.json" \
  --output "$analysis_root/two_model_summary_v1.json" --min-finding-n 100

PYTHONPATH=anchor "$eval_python" -m corrected_sgta.build_reader_boundary_paper_packet_v1 \
  --summary "$analysis_root/two_model_summary_v1.json" \
  --confirmation "$analysis_root/huatuo_v1.json" \
  --confirmation "$analysis_root/hulu_v1.json" \
  --output-dir "$analysis_root/paper_packet_v1"

# These tests are the executable acceptance contract for the frozen analysis.
"$eval_python" -m pytest -q \
  tests/test_confirm_reader_residual.py \
  tests/test_summarize_reader_residual_confirmation.py \
  tests/test_build_reader_boundary_paper_packet.py \
  tests/test_freeze_reader_residual_specs.py \
  tests/test_vindr_hidden_collector_resume.py

"$eval_python" - "$analysis_root/two_model_summary_v1.json" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
assert row["status"] == "complete"
assert len(row["models"]) >= 2
assert row["method_authorized"] is False
print(json.dumps({
    "paper_gate": row["paper_gate"],
    "observational_early_erasure_all_models": row["observational_early_erasure_all_models"],
    "method_authorized": row["method_authorized"],
    "next_action": row["next_action"],
}, indent=2))
PY

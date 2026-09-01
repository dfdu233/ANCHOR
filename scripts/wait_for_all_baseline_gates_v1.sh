#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR

scope=${1:-all}
poll_seconds=${BASELINE_GATE_POLL_SECONDS:-30}
timeout_seconds=${BASELINE_GATE_WAIT_TIMEOUT_SECONDS:-86400}
files=()

case "$scope" in
  native|rag)
    # Native greedy/beam and shared-RAG have no dependency on mitigation or
    # trained-checkpoint conformance gates. Their own per-run qualification is
    # enforced after generation.
    exit 0
    ;;
  llava)
    files=(corrected_runs/paper_baselines_v1/full_matrix_v1/gates/llava_methods/t1_t2_audit.json)
    ;;
  cross)
    files=(
      corrected_runs/paper_baselines_v1/cross_model_gates_v2/huatuo/vcd/t1_t2_audit.json
      corrected_runs/paper_baselines_v1/cross_model_gates_v2/huatuo/dola/t1_t2_audit.json
      corrected_runs/paper_baselines_v1/cross_model_gates_v2/hulu/vcd/t1_t2_audit.json
      corrected_runs/paper_baselines_v1/cross_model_gates_v2/hulu/dola/t1_t2_audit.json
    )
    ;;
  qwen)
    files=(
      corrected_runs/paper_baselines_v1/full_matrix_v1/cross_model_methods/qwen/vcd/t1_t2_audit.json
      corrected_runs/paper_baselines_v1/full_matrix_v1/gates/qwen_dola/summary.json
    )
    ;;
  qwen_vcd)
    files=(corrected_runs/paper_baselines_v1/full_matrix_v1/cross_model_methods/qwen/vcd/t1_t2_audit.json)
    ;;
  qwen_dola)
    files=(corrected_runs/paper_baselines_v1/full_matrix_v1/gates/qwen_dola/summary.json)
    ;;
  vhr)
    files=(corrected_runs/paper_baselines_v1/full_matrix_v1/trained_llava15/vhr_gates/t1_t2_audit.json)
    ;;
  trained)
    files=(corrected_runs/paper_baselines_v1/trained_llava_t2_v2/t2_audit.json)
    ;;
  all)
    files=(
      corrected_runs/paper_baselines_v1/full_matrix_v1/gates/llava_methods/t1_t2_audit.json
      corrected_runs/paper_baselines_v1/cross_model_gates_v2/huatuo/vcd/t1_t2_audit.json
      corrected_runs/paper_baselines_v1/cross_model_gates_v2/huatuo/dola/t1_t2_audit.json
      corrected_runs/paper_baselines_v1/cross_model_gates_v2/hulu/vcd/t1_t2_audit.json
      corrected_runs/paper_baselines_v1/cross_model_gates_v2/hulu/dola/t1_t2_audit.json
      corrected_runs/paper_baselines_v1/full_matrix_v1/cross_model_methods/qwen/vcd/t1_t2_audit.json
      corrected_runs/paper_baselines_v1/full_matrix_v1/gates/qwen_dola/summary.json
      corrected_runs/paper_baselines_v1/trained_llava_t2_v2/t2_audit.json
    )
    ;;
  *)
    echo "unknown baseline gate scope: $scope" >&2
    exit 64
    ;;
esac

started=$SECONDS
while true; do
  pending=()
  for path in "${files[@]}"; do
    if [[ ! -f "$path" ]] || ! /opt/miniconda3/bin/python - "$path" <<'PY'
import json,sys
try:
    payload=json.load(open(sys.argv[1]))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
terminal=(
    isinstance(payload.get('passed'), bool)
    or isinstance(payload.get('all_evaluated'), bool)
    or isinstance(payload.get('passed_variants'), list)
)
raise SystemExit(0 if terminal else 1)
PY
    then
      pending+=("$path")
    fi
  done
  [[ "${#pending[@]}" -eq 0 ]] && exit 0
  if (( SECONDS - started >= timeout_seconds )); then
    echo "timed out waiting for baseline gate scope=$scope after ${timeout_seconds}s" >&2
    printf 'pending or invalid gate: %s\n' "${pending[@]}" >&2
    exit 75
  fi
  sleep "$poll_seconds"
done

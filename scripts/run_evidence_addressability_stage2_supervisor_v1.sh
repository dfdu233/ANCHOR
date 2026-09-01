#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR

root=corrected_runs/evidence_addressability_gate_v1
huatuo_raw="$root/raw_visual_huatuo_v4"
hulu_raw="$root/raw_visual_hulu_v4"
huatuo_result="$root/huatuo_raw_formal_v1.json"
hulu_result="$root/hulu_raw_formal_v1.json"
combined="$root/raw_stage2_combined_v1.json"
log="$root/logs/stage2_supervisor_v1.log"
mkdir -p "$root/logs"

timestamp() { date -u +%FT%TZ; }

restart_baselines() {
  if ! tmux has-session -t baseline_cross_methods_v3 2>/dev/null; then
    tmux new-session -d -s baseline_cross_methods_v3 \
      "cd /home/dbw/ANCHOR && bash scripts/run_baseline_cross_model_methods_long_queue_v1.sh"
  fi
  if ! tmux has-session -t baseline_matrix_v1 2>/dev/null; then
    tmux new-session -d -s baseline_matrix_v1 \
      "cd /home/dbw/ANCHOR && bash scripts/run_baseline_native_long_queue_v1.sh"
  fi
  if ! tmux has-session -t baseline_shared_rag_v1 2>/dev/null; then
    tmux new-session -d -s baseline_shared_rag_v1 \
      "cd /home/dbw/ANCHOR && bash scripts/run_baseline_shared_rag_long_queue_v1.sh"
  fi
  echo "$(timestamp) restored paused baseline queues" >>"$log"
}

stop_remaining_captures() {
  for session in addressability_raw_huatuo_v4 addressability_raw_hulu_v4; do
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
    fi
  done
  echo "$(timestamp) stopped remaining Addressability capture waiters after failure" >>"$log"
}

wait_for_capture() {
  local session=$1 directory=$2
  while [[ ! -f "$directory/summary.json" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "$(timestamp) capture failed before summary: $session" >>"$log"
      return 1
    fi
    sleep 20
  done
  /opt/miniconda3/bin/python - "$directory/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
raise SystemExit(0 if d.get('status') == 'complete' else 1)
PY
}

echo "$(timestamp) supervisor started" >>"$log"
if ! wait_for_capture addressability_raw_huatuo_v4 "$huatuo_raw"; then
  stop_remaining_captures
  restart_baselines
  exit 1
fi
echo "$(timestamp) Huatuo capture complete" >>"$log"
if ! wait_for_capture addressability_raw_hulu_v4 "$hulu_raw"; then
  stop_remaining_captures
  restart_baselines
  exit 1
fi
echo "$(timestamp) Hulu capture complete" >>"$log"

PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.corrected_sgta.analyze_evidence_addressability_raw_gate_v1 \
  --dev corrected_runs/vindr_v2/hidden_dev_huatuo_all_findings_v3 \
  --confirmation corrected_runs/vindr_v2/hidden_confirmation_huatuo_recoverability_v1 \
  --raw "$huatuo_raw" --output "$huatuo_result" \
  >>"$log" 2>&1
huatuo_rc=$?

PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.corrected_sgta.analyze_evidence_addressability_raw_gate_v1 \
  --dev corrected_runs/vindr_v2/hidden_dev_hulu_all_findings_v1 \
  --confirmation corrected_runs/vindr_v2/hidden_confirmation_hulu_recoverability_v1 \
  --raw "$hulu_raw" --output "$hulu_result" \
  >>"$log" 2>&1
hulu_rc=$?

/opt/miniconda3/bin/python - "$combined" "$huatuo_result" "$hulu_result" "$huatuo_rc" "$hulu_rc" <<'PY'
import datetime,hashlib,json,sys
from pathlib import Path
target,huatuo_path,hulu_path=map(Path,sys.argv[1:4])
returncodes={'huatuo':int(sys.argv[4]),'hulu':int(sys.argv[5])}
models={}
for name,path in [('huatuo',huatuo_path),('hulu',hulu_path)]:
    if returncodes[name] == 0 and path.is_file():
        payload=json.loads(path.read_text())
        models[name]={
            'result':str(path),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'model_pass':bool(payload.get('raw_stage_model_gate',{}).get('model_pass')),
        }
    else:
        models[name]={'result':str(path),'model_pass':False,'returncode':returncodes[name]}
both=all(row['model_pass'] for row in models.values())
result={
    'protocol':'evidence-addressability-raw-stage2-combined-v1',
    'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'models':models,
    'both_models_pass':both,
    'decision':'GO_LOCALIZATION_AND_CAUSALITY' if both else 'CLOSE_INTERNAL_DECODING_ROUTE',
    'baseline_action':'restore_all_paused_queues',
}
temporary=target.with_suffix(target.suffix+'.tmp')
temporary.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
temporary.replace(target)
PY

restart_baselines
echo "$(timestamp) supervisor complete; decision in $combined" >>"$log"

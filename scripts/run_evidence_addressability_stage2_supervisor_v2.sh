#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR

root=corrected_runs/evidence_addressability_gate_v2
log="$root/logs/stage2_supervisor_v2.log"
findings=(aortic_enlargement cardiomegaly lung_opacity nodule_mass pleural_effusion pleural_thickening pulmonary_fibrosis)
dev_huatuo=corrected_runs/vindr_v2/hidden_confirmation_huatuo_recoverability_v1
dev_hulu=corrected_runs/vindr_v2/hidden_confirmation_hulu_recoverability_v1
fresh_huatuo="$root/hidden_fresh_huatuo_v2"
fresh_hulu="$root/hidden_fresh_hulu_v3"
raw_huatuo="$root/raw_huatuo_v1"
raw_hulu="$root/raw_hulu_v1"
select_huatuo="$root/selection_huatuo_v1.json"
select_hulu="$root/selection_hulu_v1.json"
result_huatuo="$root/confirmation_huatuo_v1.json"
result_hulu="$root/confirmation_hulu_v1.json"
joint="$root/joint_gate_v1.json"
holdout_lock=/home/dbw/datasets/physionet/vindr-cxr/1.0.0/addressability_holdout_v2/lock_receipt.json
exposure_audit="$root/holdout_prior_exposure_audit_v2.json"
mkdir -p "$root/logs"

timestamp() { date -u +%FT%TZ; }

restart_baselines() {
  if ! tmux has-session -t baseline_llava_methods_v2 2>/dev/null; then
    tmux new-session -d -s baseline_llava_methods_v2 \
      "cd /home/dbw/ANCHOR && bash scripts/run_baseline_llava_methods_long_queue_v1.sh"
  fi
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

stop_exploration_jobs() {
  for session in addressability_fresh_hidden_huatuo_v2 addressability_fresh_hidden_hulu_v2 addressability_fresh_hidden_hulu_v3; do
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
    fi
  done
}

fail() {
  echo "$(timestamp) FAIL: $1" >>"$log"
  stop_exploration_jobs
  restart_baselines
  exit 1
}

wait_complete() {
  local session=$1 directory=$2
  while [[ ! -f "$directory/summary.json" ]]; do
    if ! tmux has-session -t "$session" 2>/dev/null; then
      return 1
    fi
    sleep 20
  done
  /opt/miniconda3/bin/python - "$directory/summary.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('status') == 'complete' else 1)
PY
}

run_raw() {
  local model=$1 dev=$2 fresh=$3 output=$4 model_dir=$5 log_name=$6
  local python=/opt/miniconda3/envs/huatuo/bin/python
  if [[ "$model" == hulu ]]; then
    python=/home/dbw/.venvs/hulumed/bin/python
  fi
  local resume=()
  if [[ -f "$output/summary.json" ]]; then
    return 0
  fi
  if [[ -d "$output" ]]; then
    resume=(--resume)
  fi
  PYTHONPATH=. "$python" \
    -m anchor.corrected_sgta.collect_evidence_addressability_visual_features_v1 \
    --model "$model" --dev "$dev" --confirmation "$fresh" \
    --findings "${findings[@]}" --image-root /workspace/vinbigdata/train \
    --output-dir "$output" --model-dir "$model_dir" --progress-every 64 \
    --wait-for-gpu-lock "${resume[@]}" >>"$root/logs/$log_name" 2>&1
}

echo "$(timestamp) supervisor v2 started" >>"$log"
if [[ -f "$joint" ]]; then
  /opt/miniconda3/bin/python - "$joint" "$result_huatuo" "$result_hulu" anchor/corrected_sgta/authorize_evidence_addressability_joint_gate_v1.py <<'PY' \
    || fail "stale or invalid existing joint result"
import hashlib,json,pathlib,sys
j,h,t,c=map(pathlib.Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads(j.read_text())
def attachments(result):
 x=json.loads(result.read_text()); p=pathlib.Path(x['predictions']); m=pathlib.Path(x['opened_marker'])
 return p.is_file() and m.is_file() and sha(p)==x['predictions_sha256'] and sha(m)==x['opened_marker_sha256']
mp=d.get('model_pass',{})
valid_mp=(set(mp)=={'huatuo','hulu'} and all(type(value) is bool for value in mp.values()))
ok=(d.get('protocol')=='evidence-addressability-raw-stage2-joint-authorizer-v1' and d.get('status')=='complete' and d.get('authorizer_code_sha256')==sha(c) and h.is_file() and t.is_file() and attachments(h) and attachments(t) and d.get('inputs',{}).get('huatuo',{}).get('sha256')==sha(h) and d.get('inputs',{}).get('hulu',{}).get('sha256')==sha(t) and valid_mp and d.get('both_models_pass')==all(mp.values()) and d.get('decision')==('GO_TO_LOCALIZATION_AND_CAUSALITY' if d.get('both_models_pass') else 'CLOSE_GLOBAL_SUMMARY_INTERNAL_DECODING_ROUTE'))
raise SystemExit(0 if ok else 1)
PY
  echo "$(timestamp) validated existing joint result; no confirmation rerun" >>"$log"
  restart_baselines
  exit 0
fi

wait_complete addressability_fresh_hidden_huatuo_v2 "$fresh_huatuo" \
  || fail "fresh Huatuo hidden collection failed"
echo "$(timestamp) fresh Huatuo margins complete" >>"$log"
wait_complete addressability_fresh_hidden_hulu_v3 "$fresh_hulu" \
  || fail "fresh Hulu hidden collection failed"
echo "$(timestamp) fresh Hulu margins complete" >>"$log"

run_raw huatuo "$dev_huatuo" "$fresh_huatuo" "$raw_huatuo" \
  /home/dbw/models/HuatuoGPT-Vision-7B raw_huatuo_v1.log \
  || fail "Huatuo raw/projector collection failed"
run_raw hulu "$dev_hulu" "$fresh_hulu" "$raw_hulu" \
  /home/dbw/models/Hulu-Med-4B raw_hulu_v1.log \
  || fail "Hulu raw/projector collection failed"
echo "$(timestamp) raw/projector features complete" >>"$log"

if [[ ! -f "$select_huatuo" ]]; then
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.corrected_sgta.analyze_evidence_addressability_raw_gate_v2 select \
    --model huatuo --dev "$dev_huatuo" --raw "$raw_huatuo" \
    --findings "${findings[@]}" --holdout-lock "$holdout_lock" \
    --exposure-audit "$exposure_audit" --output "$select_huatuo" >>"$log" 2>&1 \
    || fail "Huatuo development selection failed"
fi
if [[ ! -f "$select_hulu" ]]; then
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.corrected_sgta.analyze_evidence_addressability_raw_gate_v2 select \
    --model hulu --dev "$dev_hulu" --raw "$raw_hulu" \
    --findings "${findings[@]}" --holdout-lock "$holdout_lock" \
    --exposure-audit "$exposure_audit" --output "$select_hulu" >>"$log" 2>&1 \
    || fail "Hulu development selection failed"
fi
echo "$(timestamp) both development selections frozen" >>"$log"

if [[ -f "$result_huatuo" || -f "${result_huatuo%.json}.predictions.jsonl" ]]; then
  /opt/miniconda3/bin/python - "$result_huatuo" <<'PY' || fail "partial/stale Huatuo confirmation detected"
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); q=pathlib.Path(d['predictions']); m=pathlib.Path(d['opened_marker'])
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
ok=(d.get('protocol')=='evidence-addressability-raw-increment-gate-v2' and d.get('mode')=='fresh_confirmation_once' and d.get('status')=='complete' and d.get('model')=='huatuo' and q.is_file() and sha(q)==d['predictions_sha256'] and m.is_file() and sha(m)==d['opened_marker_sha256'])
raise SystemExit(0 if ok else 1)
PY
else
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.corrected_sgta.analyze_evidence_addressability_raw_gate_v2 confirm \
    --model huatuo --dev "$dev_huatuo" --confirmation "$fresh_huatuo" \
    --raw "$raw_huatuo" --findings "${findings[@]}" \
    --holdout-lock "$holdout_lock" --exposure-audit "$exposure_audit" \
    --selection "$select_huatuo" --output "$result_huatuo" >>"$log" 2>&1 \
    || fail "Huatuo one-shot confirmation failed"
fi
if [[ -f "$result_hulu" || -f "${result_hulu%.json}.predictions.jsonl" ]]; then
  /opt/miniconda3/bin/python - "$result_hulu" <<'PY' || fail "partial/stale Hulu confirmation detected"
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); q=pathlib.Path(d['predictions']); m=pathlib.Path(d['opened_marker'])
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
ok=(d.get('protocol')=='evidence-addressability-raw-increment-gate-v2' and d.get('mode')=='fresh_confirmation_once' and d.get('status')=='complete' and d.get('model')=='hulu' and q.is_file() and sha(q)==d['predictions_sha256'] and m.is_file() and sha(m)==d['opened_marker_sha256'])
raise SystemExit(0 if ok else 1)
PY
else
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.corrected_sgta.analyze_evidence_addressability_raw_gate_v2 confirm \
    --model hulu --dev "$dev_hulu" --confirmation "$fresh_hulu" \
    --raw "$raw_hulu" --findings "${findings[@]}" \
    --holdout-lock "$holdout_lock" --exposure-audit "$exposure_audit" \
    --selection "$select_hulu" --output "$result_hulu" >>"$log" 2>&1 \
    || fail "Hulu one-shot confirmation failed"
fi

PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.corrected_sgta.authorize_evidence_addressability_joint_gate_v1 \
  --huatuo "$result_huatuo" --hulu "$result_hulu" --output "$joint" \
  >>"$log" 2>&1 || fail "joint authorizer failed"

echo "$(timestamp) joint decision complete: $joint" >>"$log"
restart_baselines

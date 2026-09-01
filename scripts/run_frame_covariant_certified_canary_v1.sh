#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

jobs=corrected_runs/detached_jobs
lock="$jobs/locks/gpu0-vindr-v2.lock"
state="$jobs/frame-covariant-certified-canary-v1.state.jsonl"
log_root="$jobs/logs/frame-covariant-certified-canary-v1"
output_root=corrected_runs/daylong_idea_search_v1/frame_covariant_certified_canary_v2
certificate=configs/frame_covariant_orientation_cert_v1.json
mkdir -p "$jobs/locks" "$log_root" "$output_root"
exec 8>"$lock"

runtime_for() {
  case "$1" in
    hulu) echo /home/dbw/.venvs/hulumed/bin/python ;;
    huatuo) echo /opt/miniconda3/envs/huatuo/bin/python ;;
    *) return 2 ;;
  esac
}

record_state() {
  local status=$1 model=$2 reason=${3:-}
  /opt/miniconda3/bin/python - "$state" "$status" "$model" "$reason" <<'PY'
import datetime,json,sys
path,status,model,reason=sys.argv[1:]
row={"time":datetime.datetime.now(datetime.timezone.utc).isoformat(),"status":status,"model":model,"reason":reason}
with open(path,"a") as handle:
    handle.write(json.dumps(row)+"\n")
PY
}

for model in hulu huatuo; do
  python=$(runtime_for "$model") || exit 2
  output="$output_root/$model"
  log="$log_root/$model.log"
  resume=()
  [[ -f "$output/config.json" ]] && resume=(--resume)
  record_state waiting "$model" canonical_gpu_lock
  flock 8
  record_state running "$model" orientation_certified_n32
  PYTHONPATH=. "$python" -m anchor.corrected_sgta.run_frame_covariant_cross_model_v1 \
    --model "$model" \
    --output-dir "$output" \
    --limit 32 \
    --candidate-pool-size 64 \
    --orientation-certificate "$certificate" \
    --max-new-tokens 96 \
    --seed 20260813 \
    "${resume[@]}" >>"$log" 2>&1
  rc=$?
  flock -u 8
  if [[ "$rc" -eq 0 ]]; then
    record_state completed "$model" "$output/analysis.json"
  else
    record_state failed "$model" "rc=$rc; log=$log"
  fi
done

record_state queue_completed all "$output_root"

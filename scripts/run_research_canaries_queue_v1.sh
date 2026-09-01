#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

jobs=corrected_runs/detached_jobs
lock="$jobs/locks/gpu0-vindr-v2.lock"
state="$jobs/research-canaries-queue-v1.state.jsonl"
log_root="$jobs/logs/research-canaries-v1"
mkdir -p "$jobs/locks" "$log_root"
exec 8>"$lock"

runtime_for() {
  case "$1" in
    huatuo) echo /opt/miniconda3/envs/huatuo/bin/python ;;
    hulu) echo /home/dbw/.venvs/hulumed/bin/python ;;
    *) return 2 ;;
  esac
}

record_state() {
  local status=$1 model=$2 experiment=$3 reason=${4:-}
  /opt/miniconda3/bin/python - "$state" "$status" "$model" "$experiment" "$reason" <<'PY'
import datetime,json,sys
path,status,model,experiment,reason=sys.argv[1:]
row={"time":datetime.datetime.now(datetime.timezone.utc).isoformat(),"status":status,"model":model,"experiment":experiment,"reason":reason}
with open(path,"a") as handle: handle.write(json.dumps(row)+"\n")
PY
}

run_target_blind() {
  local model=$1 experiment=$2 manifest=$3 output=$4 image_root=$5
  local python log resume=() rc
  python=$(runtime_for "$model") || return 2
  log="$log_root/${model}-${experiment}.log"
  mkdir -p "$output"
  if [[ -f "$output/generation_config.json" ]]; then
    resume=(--resume)
  fi
  record_state waiting "$model" "$experiment" "canonical_gpu_lock"
  flock 8
  record_state running "$model" "$experiment" "manifest=$manifest"
  PYTHONPATH=. "$python" -m anchor.corrected_sgta.run_target_blind_canary_v1 \
    --model "$model" --manifest "$manifest" --image-root "$image_root" \
    --output-dir "$output" --max-new-tokens 128 "${resume[@]}" >>"$log" 2>&1
  rc=$?
  flock -u 8
  if [[ "$rc" -eq 0 ]]; then
    record_state completed "$model" "$experiment" "answers=$(wc -l < "$output/answers.jsonl")"
  else
    record_state failed "$model" "$experiment" "rc=$rc;log=$log"
  fi
  return "$rc"
}

# The first natural-donor polarity extractor failed a coordination-negation
# audit before GPU execution.  This queue is fail-closed until an independently
# rebuilt direct-polarity artifact explicitly passes; preserving the commands
# below documents the sealed plan without accidentally running invalid inputs.
semantic_gate=corrected_runs/matched_retrieval_polarity_pilot_v1/direct_polarity_semantic_gate.json
if ! /opt/miniconda3/bin/python - "$semantic_gate" <<'PY'
import json,sys
try:
    payload=json.load(open(sys.argv[1]))
except (FileNotFoundError,json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("passed") is True and payload.get("protocol")=="direct-polarity-semantic-gate-v1" else 1)
PY
then
  record_state sealed_invalid_input all all "natural donor polarity scope audit failed; corrected semantic gate absent"
  exit 0
fi

pilot_manifest=corrected_runs/matched_retrieval_polarity_pilot_v1/target_blind_pilot_v2.json
pilot_root=corrected_runs/matched_retrieval_polarity_pilot_v1/generated_answers
for model in huatuo hulu; do
  run_target_blind "$model" matched_polarity_pilot "$pilot_manifest" \
    "$pilot_root/$model" data/medheval/images || true
done

if [[ -f "$pilot_root/huatuo/answers.jsonl" && -f "$pilot_root/hulu/answers.jsonl" ]]; then
  /opt/miniconda3/envs/huatuo/bin/python -m \
    anchor.corrected_sgta.analyze_matched_retrieval_polarity_pilot_v1 \
    >>"$log_root/matched-polarity-analysis.log" 2>&1 || \
    record_state analysis_failed both matched_polarity_pilot "$log_root/matched-polarity-analysis.log"
fi

firewall_root=corrected_runs/polarity_firewall_canary_v1
for model in huatuo hulu; do
  for arm in depolarized_rag token_matched_neutral_rag query_term_only_neutral_rag; do
    manifest="$firewall_root/${arm}.json"
    if [[ "$arm" == token_matched_neutral_rag ]]; then
      manifest="$firewall_root/${model}_token_matched_neutral_rag.json"
    fi
    run_target_blind "$model" "firewall_${arm}" "$manifest" \
      "$firewall_root/generated_answers/$model/$arm" data/medheval/images || true
  done
done

if [[ -f "$firewall_root/generated_answers/huatuo/query_term_only_neutral_rag/answers.jsonl" && \
      -f "$firewall_root/generated_answers/hulu/query_term_only_neutral_rag/answers.jsonl" ]]; then
  /opt/miniconda3/envs/huatuo/bin/python -m \
    anchor.corrected_sgta.analyze_polarity_firewall_canary_v1 \
    >>"$log_root/firewall-analysis.log" 2>&1 || \
    record_state analysis_failed both firewall_five_arm "$log_root/firewall-analysis.log"
fi

record_state queue_completed all all "research canary queue exhausted"

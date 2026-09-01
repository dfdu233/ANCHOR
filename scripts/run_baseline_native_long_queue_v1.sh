#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

root=corrected_runs/paper_baselines_v1/full_matrix_v1/native
jobs=corrected_runs/detached_jobs
lock="$jobs/locks/gpu0-vindr-v2.lock"
state="$jobs/baseline-native-long-queue-v1.state.jsonl"
mkdir -p "$root" "$jobs/locks" "$jobs/logs"
bash scripts/wait_for_all_baseline_gates_v1.sh native
exec 8>"$lock"

runtime_for() {
  case "$1" in
    hulu) echo /home/dbw/.venvs/hulumed/bin/python ;;
    qwen) echo /home/dbw/.venvs/qwen25vl-v2/bin/python ;;
    huatuo|llava) echo /opt/miniconda3/envs/huatuo/bin/python ;;
    *) return 2 ;;
  esac
}

dataset_contract() {
  case "$1" in
    vqa_rad_official_oe) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json|/home/dbw/datasets/public/vqa_rad_hf/test_images|200" ;;
    visual_mimic_oe) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/visual_mimic_oe.json|/home/dbw/ANCHOR/data/medheval/images|490" ;;
    iu_xray_report) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/iu_xray_report.json|/home/dbw/ANCHOR/data/medheval/images/IU-Xray|590" ;;
    mimic_cxr_report) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/mimic_cxr_report.json|/home/dbw/ANCHOR/data/medheval/images|694" ;;
    *) return 2 ;;
  esac
}

record_state() {
  local status=$1 model=$2 dataset=$3 method=$4 reason=${5:-}
  /opt/miniconda3/bin/python - "$state" "$status" "$model" "$dataset" "$method" "$reason" <<'PY'
import datetime,json,sys
path,status,model,dataset,method,reason=sys.argv[1:]
row={"time":datetime.datetime.now(datetime.timezone.utc).isoformat(),"status":status,"model":model,"dataset":dataset,"method":method,"reason":reason}
with open(path,"a") as f: f.write(json.dumps(row)+"\n")
PY
}

run_job() {
  local model=$1 dataset=$2 method=$3 limit=${4:-0}
  local python contract manifest images expected beams out log attempt rc got
  python=$(runtime_for "$model") || return 2
  contract=$(dataset_contract "$dataset") || return 2
  IFS='|' read -r manifest images expected <<<"$contract"
  [[ "$limit" -gt 0 ]] && expected=$limit
  beams=1
  [[ "$method" == beam ]] && beams=4
  out="$root/$model/$dataset/$method"
  [[ "$limit" -gt 0 ]] && out="$root/gates/$model/$dataset/${method}_n${limit}"
  log="$jobs/logs/baseline-native-${model}-${dataset}-${method}-n${expected}.log"
  mkdir -p "$out"
  got=0
  [[ -f "$out/answers.jsonl" ]] && got=$(wc -l < "$out/answers.jsonl")
  if [[ "$got" -ne "$expected" ]]; then
    record_state running "$model" "$dataset" "$method" "${got}/${expected}"
    rc=1
    flock 8
    for attempt in 1 2 3; do
      PYTHONPATH=. "$python" -m anchor.medeval.run_native_oe_vqa \
        --model "$model" --manifest "$manifest" --image-root "$images" \
        --output-dir "$out" --limit "$limit" --max-new-tokens 256 --seed 42 \
        --decode-mode "$method" --num-beams "$beams" >>"$log" 2>&1
      rc=$?
      [[ "$rc" -eq 0 ]] && break
      if tail -80 "$log" | grep -Eq 'empty generation|refusing to resume an incompatible|manifest qids are not unique|not an exact unique manifest prefix'; then
        record_state semantic_failed "$model" "$dataset" "$method" "attempt=${attempt};rc=${rc};not_retried"
        break
      fi
      record_state retry "$model" "$dataset" "$method" "attempt=${attempt};rc=${rc}"
    done
    flock -u 8
    if [[ "$rc" -ne 0 ]]; then
      record_state failed "$model" "$dataset" "$method" "generation_rc=${rc};log=${log}"
      return 0
    fi
  fi
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_oe_generation \
    --manifest "$manifest" --answers "$out/answers.jsonl" --limit "$expected" \
    --max-new-tokens 256 --max-cap-hit-rate 0.05 \
    --output "$out/qualification.json" >>"$log" 2>&1
  rc=$?
  if [[ "$rc" -ne 0 ]]; then
    record_state quality_failed "$model" "$dataset" "$method" "qualification_rc=${rc};log=${log}"
    return 0
  fi
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.evaluate_oe_vqa \
    --manifest "$manifest" --answers "$out/answers.jsonl" \
    --output "$out/evaluation_lexical_auxiliary.json" \
    --bootstrap-replicates 5000 --seed 42 --max-new-tokens 256 >>"$log" 2>&1
  rc=$?
  if [[ "$rc" -eq 0 ]]; then
    record_state completed "$model" "$dataset" "$method" "${expected}/${expected}"
  else
    record_state scoring_failed "$model" "$dataset" "$method" "evaluation_rc=${rc};log=${log}"
  fi
}

# Missing native-control gates. Existing Hulu/LLaVA gates remain authoritative.
for model in huatuo qwen; do
  run_job "$model" vqa_rad_official_oe greedy 32
  run_job "$model" vqa_rad_official_oe beam 32
done

# Complete references first, then the length/search control. Long report/OE jobs
# are deliberately ahead of CE so they continue while method ports are audited.
model_authorized() {
  local model=$1 greedy beam
  case "$model" in
    huatuo|qwen)
      greedy="$root/gates/$model/vqa_rad_official_oe/greedy_n32/qualification.json"
      beam="$root/gates/$model/vqa_rad_official_oe/beam_n32/qualification.json"
      ;;
    hulu|llava)
      greedy="corrected_runs/unified_eval/smoke/native_oe_controls_t2_v1/$model/greedy256/qualification.json"
      beam="corrected_runs/unified_eval/smoke/native_oe_controls_t2_v1/$model/beam4_256/qualification.json"
      ;;
  esac
  /opt/miniconda3/bin/python - "$greedy" "$beam" <<'PY'
import json,sys
ok=True
for path in sys.argv[1:]:
    try: ok = ok and bool(json.load(open(path)).get("passed"))
    except (FileNotFoundError, json.JSONDecodeError): ok=False
raise SystemExit(0 if ok else 1)
PY
}

for method in greedy beam; do
  for model in huatuo hulu llava qwen; do
    if ! model_authorized "$model"; then
      record_state gate_warning "$model" all "$method" \
        "VQA-RAD n=32 format warning retained as diagnostic; each full dataset is qualified independently"
    fi
    for dataset in visual_mimic_oe iu_xray_report mimic_cxr_report vqa_rad_official_oe; do
      run_job "$model" "$dataset" "$method" 0
    done
  done
done

record_state queue_completed all all all "native greedy/beam matrix exhausted"

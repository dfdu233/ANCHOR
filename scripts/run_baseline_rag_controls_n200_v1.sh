#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

jobs=corrected_runs/detached_jobs
control_root=corrected_runs/paper_baselines_v1/full_matrix_v1/rag_controls_n200/knowledge_mimic_ce
output_root="$control_root/generation"
state="$jobs/baseline-rag-controls-n200-v1.state.jsonl"
images=data/medheval/images
mkdir -p "$output_root" "$jobs/locks" "$jobs/logs"
bash scripts/wait_for_all_baseline_gates_v1.sh rag

runtime_for() {
  case "$1" in
    hulu) echo /home/dbw/.venvs/hulumed/bin/python ;;
    qwen) echo /home/dbw/.venvs/qwen25vl-v2/bin/python ;;
    huatuo|llava) echo /opt/miniconda3/envs/huatuo/bin/python ;;
    *) return 2 ;;
  esac
}

exec 8>"$jobs/locks/gpu0-vindr-v2.lock"
for model in huatuo hulu llava qwen; do
  python=$(runtime_for "$model")
  for condition in shuffled_context image_swap; do
    manifest="$control_root/$condition.json"
    out="$output_root/$model/$condition"
    log="$jobs/logs/baseline-rag-control-${model}-${condition}-n200.log"
    mkdir -p "$out"
    rc=1
    flock 8
    for attempt in 1 2 3; do
      PYTHONPATH=. "$python" -m anchor.medeval.run_native_oe_vqa \
        --model "$model" --manifest "$manifest" --image-root "$images" \
        --output-dir "$out" --limit 0 --max-new-tokens 128 --seed 42 \
        --decode-mode greedy --num-beams 1 >>"$log" 2>&1
      rc=$?
      [[ "$rc" -eq 0 ]] && break
      tail -80 "$log" | grep -Eq 'refusing to resume an incompatible|manifest qids are not unique|not an exact unique manifest prefix' && break
    done
    flock -u 8
    if [[ "$rc" -eq 0 ]]; then
      PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_ce_generation \
        --manifest "$manifest" --answers "$out/answers.jsonl" --limit 200 \
        --max-new-tokens 128 --min-parse-rate 0.90 \
        --output "$out/qualification.json" >>"$log" 2>&1
      rc=$?
    fi
    if [[ "$rc" -eq 0 ]]; then
      PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
        -m anchor.corrected_sgta.evaluate_medheval_answers \
        --answers "$out/answers.jsonl" --questions "$manifest" \
        --output "$out/evaluation_ce_v7.json" >>"$log" 2>&1
      rc=$?
    fi
    /opt/miniconda3/bin/python - "$state" "$model" "$condition" "$rc" "$attempt" "$log" <<'PY'
import datetime,json,sys
path,model,condition,rc,attempt,log=sys.argv[1:]
row={"time":datetime.datetime.now(datetime.timezone.utc).isoformat(),"model":model,"condition":condition,"n":200,"status":"completed" if rc=="0" else "failed","returncode":int(rc),"attempts":int(attempt),"log":log}
with open(path,"a") as handle: handle.write(json.dumps(row)+"\n")
PY
  done
done

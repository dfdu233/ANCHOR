#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

jobs=corrected_runs/detached_jobs
root=corrected_runs/paper_baselines_v1/full_matrix_v1/native_ce
state="$jobs/baseline-native-ce-long-queue-v1.state.jsonl"
mkdir -p "$root" "$jobs/locks" "$jobs/logs"
bash scripts/wait_for_all_baseline_gates_v1.sh native
exec 8>"$jobs/locks/gpu0-vindr-v2.lock"

runtime_for() {
  case "$1" in
    hulu) echo /home/dbw/.venvs/hulumed/bin/python ;;
    qwen) echo /home/dbw/.venvs/qwen25vl-v2/bin/python ;;
    huatuo|llava) echo /opt/miniconda3/envs/huatuo/bin/python ;;
  esac
}

dataset_contract() {
  case "$1" in
    cxr_vishal) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/cxr_vishal.json|/home/dbw/ANCHOR/data/medheval/images|5587" ;;
    knowledge_mimic_ce) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/knowledge_mimic_ce.json|/home/dbw/ANCHOR/data/medheval/images|2000" ;;
    slake_fine_grained) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/slake_fine_grained.json|/home/dbw/ANCHOR/data/medheval/images/Slake|1536" ;;
  esac
}

generation_contract_compatible() {
  local dataset=$1 manifest=$2 config=$3 prior_sha old
  [[ -f "$config" ]] || return 1
  prior_sha=$(/opt/miniconda3/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("manifest_sha256", ""))' "$config")
  [[ -n "$prior_sha" ]] || return 1
  [[ "$(sha256sum "$manifest" | cut -d' ' -f1)" == "$prior_sha" ]] && return 0
  while IFS= read -r old; do
    [[ "$(sha256sum "$old" | cut -d' ' -f1)" == "$prior_sha" ]] || continue
    /opt/miniconda3/bin/python - "$old" "$manifest" <<'PY'
import json,sys
old,new=(json.load(open(path)) for path in sys.argv[1:])
fields=("qid","id","img_name","image","question","answer")
def contract(rows):
    return [{key:row.get(key) for key in fields} for row in rows]
raise SystemExit(0 if contract(old)==contract(new) else 1)
PY
    return $?
  done < <(find corrected_runs/unified_eval/inputs/baseline_matrix_v1/stale_artifacts -path "*/$dataset.json" -type f 2>/dev/null | sort -r)
  return 1
}

archive_incompatible_generation() {
  local out=$1 stamp archive file
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  archive="$out/stale_artifacts/$stamp"
  mkdir -p "$archive"
  for file in answers.jsonl generation_config.json qualification.json evaluation_ce_v7.json; do
    [[ -f "$out/$file" ]] && mv "$out/$file" "$archive/$file"
  done
}

for method in greedy beam; do
  for model in huatuo hulu llava qwen; do
    for dataset in cxr_vishal knowledge_mimic_ce slake_fine_grained; do
      contract=$(dataset_contract "$dataset")
      IFS='|' read -r manifest images expected <<<"$contract"
      python=$(runtime_for "$model")
      beams=1; [[ "$method" == beam ]] && beams=4
      out="$root/$model/$dataset/$method"
      log="$jobs/logs/baseline-native-ce-${model}-${dataset}-${method}.log"
      mkdir -p "$out"
      rc=1
      got=0; [[ -f "$out/answers.jsonl" ]] && got=$(wc -l < "$out/answers.jsonl")
      if [[ "$got" -eq "$expected" ]] && generation_contract_compatible "$dataset" "$manifest" "$out/generation_config.json"; then
        rc=0
        echo "reuse generation-equivalent output rows=$got dataset=$dataset" >>"$log"
      else
        [[ "$got" -gt 0 ]] && archive_incompatible_generation "$out"
        flock 8
        for attempt in 1 2 3; do
          PYTHONPATH=. "$python" -m anchor.medeval.run_native_oe_vqa \
            --model "$model" --manifest "$manifest" --image-root "$images" \
            --output-dir "$out" --limit 0 --max-new-tokens 128 --seed 42 \
            --decode-mode "$method" --num-beams "$beams" >>"$log" 2>&1
          rc=$?
          [[ "$rc" -eq 0 ]] && break
          if tail -80 "$log" | grep -Eq 'refusing to resume an incompatible|manifest qids are not unique|not an exact unique manifest prefix'; then break; fi
        done
        flock -u 8
      fi
      if [[ "$rc" -eq 0 ]]; then
        PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_ce_generation \
          --manifest "$manifest" --answers "$out/answers.jsonl" --limit "$expected" \
          --max-new-tokens 128 --min-parse-rate 0.90 --output "$out/qualification.json" \
          >>"$log" 2>&1
        rc=$?
      fi
      if [[ "$rc" -eq 0 ]]; then
        PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
          -m anchor.corrected_sgta.evaluate_medheval_answers \
          --answers "$out/answers.jsonl" --questions "$manifest" \
          --output "$out/evaluation_ce_v7.json" >>"$log" 2>&1
        rc=$?
      fi
      /opt/miniconda3/bin/python - "$state" "$model" "$dataset" "$method" "$rc" "$expected" "$log" <<'PY'
import datetime,json,sys
path,model,dataset,method,rc,expected,log=sys.argv[1:]
row={"time":datetime.datetime.now(datetime.timezone.utc).isoformat(),"model":model,"dataset":dataset,"method":method,"status":"completed" if rc=="0" else "failed","returncode":int(rc),"expected":int(expected),"log":log}
with open(path,"a") as f: f.write(json.dumps(row)+"\n")
PY
    done
  done
done

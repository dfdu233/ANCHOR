#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

jobs=corrected_runs/detached_jobs
root=corrected_runs/paper_baselines_v1/full_matrix_v1/trained_llava15
state="$jobs/baseline-trained-long-queue-v1.state.jsonl"
gate=corrected_runs/paper_baselines_v1/trained_llava_t2_v2/t2_audit.json
mkdir -p "$root" "$jobs/locks" "$jobs/logs"
while [[ ! -f "$gate" ]]; do sleep 30; done
bash scripts/wait_for_all_baseline_gates_v1.sh trained
exec 8>"$jobs/locks/gpu0-vindr-v2.lock"

dataset_contract() {
  case "$1" in
    cxr_vishal) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/cxr_vishal.json|/home/dbw/ANCHOR/data/medheval/images|5587|128" ;;
    knowledge_mimic_ce) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/knowledge_mimic_ce.json|/home/dbw/ANCHOR/data/medheval/images|2000|128" ;;
    slake_fine_grained) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/slake_fine_grained.json|/home/dbw/ANCHOR/data/medheval/images/Slake|1536|128" ;;
    vqa_rad_official_oe) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json|/home/dbw/datasets/public/vqa_rad_hf/test_images|200|256" ;;
    visual_mimic_oe) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/visual_mimic_oe.json|/home/dbw/ANCHOR/data/medheval/images|490|256" ;;
    iu_xray_report) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/iu_xray_report.json|/home/dbw/ANCHOR/data/medheval/images/IU-Xray|590|256" ;;
    mimic_cxr_report) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/mimic_cxr_report.json|/home/dbw/ANCHOR/data/medheval/images|694|256" ;;
  esac
}

for variant in base ha-dpo opa-dpo da-dpo sentinel less-is-more factmm-rag-generator; do
  if ! /opt/miniconda3/bin/python - "$gate" "$variant" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));raise SystemExit(0 if sys.argv[2] in d.get('passed_variants',[]) else 1)
PY
  then
    /opt/miniconda3/bin/python - "$state" "$variant" <<'PY'
import datetime,json,sys
with open(sys.argv[1],'a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'variant':sys.argv[2],'status':'N/A','reason':'official-entry T2 token identity failed'})+'\n')
PY
    continue
  fi
  for dataset in visual_mimic_oe iu_xray_report mimic_cxr_report vqa_rad_official_oe cxr_vishal knowledge_mimic_ce slake_fine_grained; do
    contract=$(dataset_contract "$dataset")
    IFS='|' read -r manifest images expected tokens <<<"$contract"
    out="$root/$variant/$dataset"
    log="$jobs/logs/baseline-trained-${variant}-${dataset}.log"
    mkdir -p "$out"
    rc=1
    flock 8
    for attempt in 1 2 3; do
      PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
        -m anchor.corrected_sgta.run_trained_llava_baseline_v1 \
        --variant "$variant" --manifest "$manifest" --image-root "$images" \
        --output-dir "$out" --limit 0 --max-new-tokens "$tokens" --seed 42 \
        >>"$log" 2>&1
      rc=$?
      [[ "$rc" -eq 0 ]] && break
    done
    flock -u 8
    /opt/miniconda3/bin/python - "$state" "$variant" "$dataset" "$rc" "$attempt" "$expected" "$log" <<'PY'
import datetime,json,sys
path,variant,dataset,rc,attempt,expected,log=sys.argv[1:]
row={"time":datetime.datetime.now(datetime.timezone.utc).isoformat(),"variant":variant,"dataset":dataset,"status":"generated" if rc=="0" else "failed","returncode":int(rc),"attempts":int(attempt),"expected":int(expected),"log":log}
with open(path,"a") as f: f.write(json.dumps(row)+"\n")
PY
  done
done

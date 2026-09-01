#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
env=/home/dbw/.venvs/vhr-official-4.45/bin/python
model=/home/dbw/models/llava-hf-llava-1.5-7b-hf-16952161b5e90aea6e332e36a6fe99024096dd0a
gate=corrected_runs/paper_baselines_v1/full_matrix_v1/trained_llava15/vhr_gates/t1_t2_audit.json
provenance=corrected_runs/paper_baselines_v1/full_matrix_v1/trained_llava15/vhr_gates/gate_provenance_v2.json
root=corrected_runs/paper_baselines_v1/full_matrix_v1/trained_llava15/vhr
state=corrected_runs/detached_jobs/baseline-vhr-full-queue-v1.state.jsonl
logroot=corrected_runs/detached_jobs/logs
while [[ ! -f "$gate" ]]; do sleep 30; done
if ! provenance_fp=$(PYTHONPATH=. "$env" -m anchor.medeval.build_baseline_gate_provenance_v1 \
  --output "$provenance" --model llava15-hf --method VHR \
  --checkpoint "$model" --config configs/unified_eval/baseline_matrix_v1.json \
  --manifest corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json \
  --source anchor/medeval/build_baseline_gate_provenance_v1.py \
  --source anchor/corrected_sgta/run_vhr_official_baseline_v1.py \
  --source anchor/corrected_sgta/run_trained_llava_baseline_v1.py \
  --source third_party/baselines/VHR/generation.py \
  --source third_party/baselines/VHR/vhr.py \
  --source third_party/baselines/VHR/main.py \
  --generation-json '{"limit":32,"max_new_tokens":256,"seed":42,"arms":["native","off","vhr","custom_base"],"vhr_aug_ratio":2.0,"vhr_last_layers":14,"vhr_layer1":true,"vhr_filter":true}'); then
  echo '{"status":"blocked","reason":"VHR checkpoint or provenance is incomplete"}' >>"$state"
  exit 70
fi
if ! /opt/miniconda3/bin/python - "$gate" "$provenance_fp" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));raise SystemExit(0 if d.get('passed') is True and d.get('gate_provenance_fingerprint')==sys.argv[2] else 1)
PY
then
  echo '{"status":"blocked","reason":"official VHR T1/T2 is missing, failed, or stale provenance; rerun gate v2"}' >>"$state"
  exit 70
fi
bash scripts/wait_for_all_baseline_gates_v1.sh vhr
contract() {
 case "$1" in
  cxr_vishal) echo "data/medheval/images|5587|128|ce";;
  knowledge_mimic_ce) echo "data/medheval/images|2000|128|ce";;
  slake_fine_grained) echo "data/medheval/images/Slake|1536|128|ce";;
  vqa_rad_official_oe) echo "/home/dbw/datasets/public/vqa_rad_hf/test_images|200|256|oe";;
  visual_mimic_oe) echo "data/medheval/images|490|256|oe";;
  iu_xray_report) echo "data/medheval/images/IU-Xray|590|256|report";;
  mimic_cxr_report) echo "data/medheval/images|694|256|report";;
 esac
}
exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
for dataset in visual_mimic_oe iu_xray_report mimic_cxr_report vqa_rad_official_oe cxr_vishal knowledge_mimic_ce slake_fine_grained; do
 IFS='|' read -r images expected tokens task <<<"$(contract "$dataset")"
 out="$root/$dataset"; log="$logroot/baseline-vhr-$dataset.log"; mkdir -p "$out"
 rc=1
 flock 8
 for attempt in 1 2 3; do
  PYTHONPATH=third_party/baselines/VHR:. "$env" -m anchor.corrected_sgta.run_vhr_official_baseline_v1 \
   --model-path "$model" --manifest "corrected_runs/unified_eval/inputs/baseline_matrix_v1/$dataset.json" \
   --image-root "$images" --output-dir "$out" --method vhr --limit 0 --max-new-tokens "$tokens" --seed 42 >>"$log" 2>&1
  rc=$?; [[ "$rc" -eq 0 ]] && break
 done
 flock -u 8
 if [[ "$rc" -eq 0 && "$task" == ce ]]; then
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.corrected_sgta.evaluate_medheval_answers --answers "$out/answers.jsonl" --questions "corrected_runs/unified_eval/inputs/baseline_matrix_v1/$dataset.json" --output "$out/evaluation_ce_v7.json" >>"$log" 2>&1; rc=$?
 elif [[ "$rc" -eq 0 && "$task" == oe ]]; then
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.evaluate_oe_vqa --manifest "corrected_runs/unified_eval/inputs/baseline_matrix_v1/$dataset.json" --answers "$out/answers.jsonl" --output "$out/evaluation_lexical_auxiliary.json" --bootstrap-replicates 5000 --seed 42 --max-new-tokens "$tokens" >>"$log" 2>&1; rc=$?
 fi
 /opt/miniconda3/bin/python - "$state" "$dataset" "$rc" "$expected" <<'PY'
import datetime,json,sys
p,d,r,n=sys.argv[1:]
with open(p,'a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':d,'status':'completed' if r=='0' else 'failed','returncode':int(r),'expected':int(n)})+'\n')
PY
done

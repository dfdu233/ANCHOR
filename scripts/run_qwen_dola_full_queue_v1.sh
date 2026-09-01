#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
gate=corrected_runs/paper_baselines_v1/full_matrix_v1/gates/qwen_dola/summary.json
provenance=corrected_runs/paper_baselines_v1/full_matrix_v1/gates/qwen_dola/gate_provenance_v2.json
root=corrected_runs/paper_baselines_v1/full_matrix_v1/cross_model_methods/qwen/dola
state=corrected_runs/detached_jobs/baseline-qwen-dola-full-v1.state.jsonl
if ! provenance_fp=$(PYTHONPATH=. /home/dbw/.venvs/qwen25vl-v2/bin/python -m anchor.medeval.build_baseline_gate_provenance_v1 \
 --output "$provenance" --model qwen --method DoLa \
 --checkpoint /home/dbw/models/Qwen2.5-VL-7B-Instruct \
 --config configs/unified_eval/baseline_matrix_v1.json \
 --manifest corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json \
 --source anchor/medeval/build_baseline_gate_provenance_v1.py \
 --source anchor/corrected_sgta/models_oe.py \
 --source anchor/corrected_sgta/cross_model_dola.py \
 --source anchor/corrected_sgta/run_cross_model_dola_gate_v1.py \
 --source anchor/corrected_sgta/run_cross_model_method_full_v1.py \
 --generation-json '{"limit":32,"max_new_tokens":256,"seed":42,"decode":"greedy","candidate_policy":"DoLa"}'); then
 echo '{"status":"blocked","reason":"Qwen DoLa provenance build failed"}' >>"$state"; exit 70
fi
while [[ ! -f "$gate" ]];do sleep 30;done
if ! /opt/miniconda3/bin/python - "$gate" "$provenance_fp" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));raise SystemExit(0 if d.get('passed') is True and d.get('gate_provenance_fingerprint')==sys.argv[2] else 1)
PY
then echo '{"status":"blocked","reason":"Qwen DoLa T1/T2 missing, failed, or stale provenance; rerun gate v2"}' >>"$state";exit 70;fi
bash scripts/wait_for_all_baseline_gates_v1.sh qwen_dola
contract(){ case "$1" in cxr_vishal)echo 'data/medheval/images|128|ce';;knowledge_mimic_ce)echo 'data/medheval/images|128|ce';;slake_fine_grained)echo 'data/medheval/images/Slake|128|ce';;vqa_rad_official_oe)echo '/home/dbw/datasets/public/vqa_rad_hf/test_images|256|oe';;visual_mimic_oe)echo 'data/medheval/images|256|oe';;iu_xray_report)echo 'data/medheval/images/IU-Xray|256|report';;mimic_cxr_report)echo 'data/medheval/images|256|report';;esac;}
exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
for dataset in visual_mimic_oe iu_xray_report mimic_cxr_report vqa_rad_official_oe cxr_vishal knowledge_mimic_ce slake_fine_grained;do
 IFS='|' read -r images tokens task<<<"$(contract "$dataset")";out="$root/$dataset";log=corrected_runs/detached_jobs/logs/baseline-qwen-dola-$dataset.log;mkdir -p "$out";rc=1
 flock 8
 for attempt in 1 2 3;do PYTHONPATH=. /home/dbw/.venvs/qwen25vl-v2/bin/python -m anchor.corrected_sgta.run_cross_model_method_full_v1 --model qwen --method dola --manifest "corrected_runs/unified_eval/inputs/baseline_matrix_v1/$dataset.json" --image-root "$images" --output-dir "$out" --max-new-tokens "$tokens" --seed 42 >>"$log" 2>&1;rc=$?;[[ "$rc" -eq 0 ]]&&break;done
 flock -u 8
 /opt/miniconda3/bin/python - "$state" "$dataset" "$rc" <<'PY'
import datetime,json,sys
with open(sys.argv[1],'a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':sys.argv[2],'status':'generated' if sys.argv[3]=='0' else 'failed','returncode':int(sys.argv[3])})+'\n')
PY
done

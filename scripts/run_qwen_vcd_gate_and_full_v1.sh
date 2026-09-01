#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR
mode=${1:-gate-and-full}
if [[ "$mode" != gate-and-full && "$mode" != --gate-only ]]; then
 echo "usage: $0 [--gate-only]" >&2
 exit 64
fi
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python=/home/dbw/.venvs/qwen25vl-v2/bin/python
root=corrected_runs/paper_baselines_v1/full_matrix_v1/cross_model_methods/qwen/vcd
audit="$root/t1_t2_audit.json"
provenance="$root/gate_provenance_v2.json"
state=corrected_runs/detached_jobs/baseline-qwen-vcd-v1.state.jsonl
mkdir -p "$root" corrected_runs/detached_jobs/logs
if ! provenance_fp=$(PYTHONPATH=. "$python" -m anchor.medeval.build_baseline_gate_provenance_v1 \
 --output "$provenance" --model qwen --method VCD \
 --checkpoint /home/dbw/models/Qwen2.5-VL-7B-Instruct \
 --config configs/unified_eval/baseline_matrix_v1.json \
 --manifest corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json \
 --source anchor/medeval/build_baseline_gate_provenance_v1.py \
 --source anchor/corrected_sgta/models_oe.py \
 --source anchor/corrected_sgta/cross_model_vcd.py \
 --source anchor/corrected_sgta/run_cross_model_vcd_gate_v1.py \
 --source anchor/corrected_sgta/run_cross_model_method_full_v1.py \
 --generation-json '{"limit":32,"max_new_tokens":256,"seed":42,"decode":"greedy","contrast":"VCD"}'); then
 echo '{"status":"blocked","reason":"Qwen VCD provenance build failed"}' >>"$state"
 exit 70
fi
gate="$root/gate_runs/$provenance_fp/gate_n32.jsonl"
mkdir -p "$(dirname "$gate")"
exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
flock 8
PYTHONPATH=. "$python" -m anchor.corrected_sgta.run_cross_model_vcd_gate_v1 \
 --model qwen --manifest corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json \
 --image-root /home/dbw/datasets/public/vqa_rad_hf/test_images --output "$gate" \
 --limit 32 --max-new-tokens 256 --seed 42 >corrected_runs/detached_jobs/logs/baseline-qwen-vcd-gate.log 2>&1
rc=$?
flock -u 8
if [[ "$rc" -eq 0 ]]; then
 /opt/miniconda3/bin/python - "$gate" "$audit" "$provenance" "$provenance_fp" <<'PY'
import hashlib,json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
exact=sum(x['off_token_exact'] for x in rows);changed=sum(x['vcd_changed'] for x in rows)
active=sum(x['vcd']['audit'].get('mean_contrast_l1',0)>0 for x in rows)
provenance=json.load(open(sys.argv[3])); expected=sys.argv[4]
r={'protocol':'qwen-vcd-t1-t2-v2','gate_provenance':sys.argv[3],'gate_provenance_sha256':hashlib.sha256(open(sys.argv[3],'rb').read()).hexdigest(),'gate_provenance_fingerprint':provenance.get('fingerprint'),'n':len(rows),'off_token_exact':exact,'changed_sequences':changed,'contrast_active_samples':active,'t1_passed':len(rows)==32 and exact==32,'t2_passed':changed>=1 and active==32}
r['provenance_passed']=provenance.get('fingerprint')==expected
r['passed']=r['t1_passed'] and r['t2_passed'] and r['provenance_passed'];open(sys.argv[2],'w').write(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed'] else 1)
PY
 rc=$?
fi
if [[ "$rc" -ne 0 ]]; then echo '{"status":"N/A","reason":"Qwen VCD faithful-port T1/T2 failed"}' >>"$state"; exit 0; fi
if ! /opt/miniconda3/bin/python - "$audit" "$provenance_fp" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));raise SystemExit(0 if d.get('passed') is True and d.get('gate_provenance_fingerprint')==sys.argv[2] else 1)
PY
then echo '{"status":"blocked","reason":"Qwen VCD gate provenance is stale"}' >>"$state"; exit 70; fi
if [[ "$mode" == --gate-only ]]; then
 /opt/miniconda3/bin/python - "$state" "$provenance_fp" <<'PY'
import datetime,json,sys
with open(sys.argv[1],'a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'gate_passed','provenance_fingerprint':sys.argv[2]})+'\n')
PY
 exit 0
fi
bash scripts/wait_for_all_baseline_gates_v1.sh qwen_vcd
contract() {
 case "$1" in
  cxr_vishal) echo "data/medheval/images|128|ce";; knowledge_mimic_ce) echo "data/medheval/images|128|ce";; slake_fine_grained) echo "data/medheval/images/Slake|128|ce";;
  vqa_rad_official_oe) echo "/home/dbw/datasets/public/vqa_rad_hf/test_images|256|oe";; visual_mimic_oe) echo "data/medheval/images|256|oe";;
  iu_xray_report) echo "data/medheval/images/IU-Xray|256|report";; mimic_cxr_report) echo "data/medheval/images|256|report";;
 esac
}
for dataset in visual_mimic_oe iu_xray_report mimic_cxr_report vqa_rad_official_oe cxr_vishal knowledge_mimic_ce slake_fine_grained; do
 IFS='|' read -r images tokens task <<<"$(contract "$dataset")"; out="$root/$dataset"; log=corrected_runs/detached_jobs/logs/baseline-qwen-vcd-$dataset.log; mkdir -p "$out"; rc=1
 flock 8
 for attempt in 1 2 3; do PYTHONPATH=. "$python" -m anchor.corrected_sgta.run_cross_model_method_full_v1 --model qwen --method vcd --manifest "corrected_runs/unified_eval/inputs/baseline_matrix_v1/$dataset.json" --image-root "$images" --output-dir "$out" --max-new-tokens "$tokens" --seed 42 >>"$log" 2>&1; rc=$?; [[ "$rc" -eq 0 ]] && break; done
 flock -u 8
 if [[ "$rc" -eq 0 && "$task" == ce ]]; then PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.corrected_sgta.evaluate_medheval_answers --answers "$out/answers.jsonl" --questions "corrected_runs/unified_eval/inputs/baseline_matrix_v1/$dataset.json" --output "$out/evaluation_ce_v7.json" >>"$log" 2>&1; rc=$?; fi
 if [[ "$rc" -eq 0 && "$task" == oe ]]; then PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.evaluate_oe_vqa --manifest "corrected_runs/unified_eval/inputs/baseline_matrix_v1/$dataset.json" --answers "$out/answers.jsonl" --output "$out/evaluation_lexical_auxiliary.json" --bootstrap-replicates 5000 --seed 42 --max-new-tokens "$tokens" >>"$log" 2>&1; rc=$?; fi
 /opt/miniconda3/bin/python - "$state" "$dataset" "$rc" <<'PY'
import datetime,json,sys
p,d,r=sys.argv[1:]
with open(p,'a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dataset':d,'status':'completed' if r=='0' else 'failed','returncode':int(r)})+'\n')
PY
done

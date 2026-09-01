#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
jobs=corrected_runs/detached_jobs
root=corrected_runs/paper_baselines_v1/full_matrix_v1/cross_model_methods
gate_root=corrected_runs/paper_baselines_v1/cross_model_gates_v2
state="$jobs/baseline-cross-model-methods-long-queue-v1.state.jsonl"
mkdir -p "$root" "$gate_root" "$jobs/locks" "$jobs/logs"

runtime_for() { [[ "$1" == hulu ]] && echo /home/dbw/.venvs/hulumed/bin/python || echo /opt/miniconda3/envs/huatuo/bin/python; }
contract() {
 case "$1" in
  visual_mimic_oe) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/visual_mimic_oe.json|data/medheval/images|256";;
  iu_xray_report) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/iu_xray_report.json|data/medheval/images/IU-Xray|256";;
  mimic_cxr_report) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/mimic_cxr_report.json|data/medheval/images|256";;
  vqa_rad_official_oe) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json|/home/dbw/datasets/public/vqa_rad_hf/test_images|256";;
  cxr_vishal) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/cxr_vishal.json|data/medheval/images|128";;
  knowledge_mimic_ce) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/knowledge_mimic_ce.json|data/medheval/images|128";;
  slake_fine_grained) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/slake_fine_grained.json|data/medheval/images/Slake|128";;
 esac
}

exec 8>"$jobs/locks/gpu0-vindr-v2.lock"

# Re-run every port gate against the exact source files used by Full. The
# source-tagged output avoids mixing a prior successful gate with later port
# repairs, even when those repairs target another architecture branch.
for method in vcd dola; do
 for model in huatuo hulu; do
  python=$(runtime_for "$model")
  provenance="$gate_root/$model/$method/gate_provenance.json"
  mkdir -p "$(dirname "$provenance")"
  tag=$("$python" - "$method" "$model" "$provenance" <<'PY'
import hashlib,importlib.metadata,json,sys
from pathlib import Path
method,model,target=sys.argv[1:]
root=Path('/home/dbw/ANCHOR')
paths=[root/'anchor/corrected_sgta/models_oe.py',root/f'anchor/corrected_sgta/cross_model_{method}.py',root/f'anchor/corrected_sgta/run_cross_model_{method}_gate_v1.py',root/'anchor/corrected_sgta/run_cross_model_method_full_v1.py']
config=root/'configs/unified_eval/baseline_matrix_v1.json'; manifest=root/'corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json'
checkpoint={'huatuo':'HuatuoGPT-Vision-7B','hulu':'Hulu-Med-4B'}[model]
checkpoint=Path('/home/dbw/models')/checkpoint
small=sorted(p for p in checkpoint.iterdir() if p.is_file() and p.suffix not in {'.safetensors','.bin'})
weights=sorted(({'path':p.name,'bytes':p.stat().st_size} for p in checkpoint.iterdir() if p.is_file() and p.suffix in {'.safetensors','.bin'}),key=lambda row:row['path'])
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def version(name):
 try:return importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:return None
deps={name:version(name) for name in ('torch','transformers','accelerate','numpy')}
payload={'protocol':'baseline-gate-provenance-v1','model':model,'method':method,'sources':[{'path':str(p.resolve()),'sha256':sha(p)} for p in paths],'frozen_config':{'path':str(config.resolve()),'sha256':sha(config)},'gate_manifest':{'path':str(manifest.resolve()),'sha256':sha(manifest)},'checkpoint':{'path':str(checkpoint.resolve()),'metadata_files':[{'path':p.name,'sha256':sha(p)} for p in small],'weight_inventory':weights},'dependencies':deps,'generation':{'limit':32,'max_new_tokens':256,'seed':42}}
payload['fingerprint']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
Path(target).write_text(json.dumps(payload,indent=2)+'\n');print(payload['fingerprint'][:16])
PY
)
  gate="$gate_root/$model/$method/t1_t2_audit.json"
  current=0
  if [[ -f "$gate" ]]; then
   /opt/miniconda3/bin/python - "$gate" "$tag" <<'PY' && current=1
import json,sys
d=json.load(open(sys.argv[1]));raise SystemExit(0 if d.get('source_tag')==sys.argv[2] else 1)
PY
  fi
  if [[ "$current" -ne 1 ]]; then
   rows="$gate_root/runs/$tag/${model}_${method}_n32.jsonl"
   mkdir -p "$(dirname "$rows")" "$(dirname "$gate")"
   flock 8
   PYTHONPATH=. "$python" -m "anchor.corrected_sgta.run_cross_model_${method}_gate_v1" \
    --model "$model" --manifest corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json \
    --image-root /home/dbw/datasets/public/vqa_rad_hf/test_images --output "$rows" \
    --limit 32 --max-new-tokens 256 --seed 42 \
    >"$jobs/logs/baseline-${model}-${method}-current-gate.log" 2>&1
   gate_rc=$?
   flock -u 8
   /opt/miniconda3/bin/python - "$rows" "$gate" "$tag" "$method" "$model" "$gate_rc" "$provenance" <<'PY'
import hashlib,json,sys
from pathlib import Path
source,target=map(Path,sys.argv[1:3]);tag,method,model=sys.argv[3:6];rc=int(sys.argv[6]);provenance=Path(sys.argv[7])
rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()] if source.is_file() else []
changed_key=f'{method}_changed'; exact=sum(bool(x.get('off_token_exact')) for x in rows);changed=sum(bool(x.get(changed_key)) for x in rows)
if method=='vcd': active=sum((x.get('vcd') or {}).get('audit',{}).get('mean_contrast_l1',0)>0 for x in rows)
else: active=sum(bool((x.get('dola') or {}).get('audit',{}).get('selected_candidate_layers')) for x in rows)
result={'protocol':'cross-model-current-source-t1-t2-v2','source_tag':tag,'gate_provenance':str(provenance),'gate_provenance_sha256':hashlib.sha256(provenance.read_bytes()).hexdigest(),'model':model,'method':method,'runner_returncode':rc,'n':len(rows),'method_off_token_exact':exact,'changed_sequences':changed,'active_samples':active,'t1_passed':len(rows)==32 and exact==32,'t2_passed':rc==0 and changed>=1 and active==32,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None}
result['passed']=result['t1_passed'] and result['t2_passed'];target.write_text(json.dumps(result,indent=2)+'\n')
PY
  fi
 done
done

bash scripts/wait_for_all_baseline_gates_v1.sh cross

for method in vcd dola; do
 for model in huatuo hulu; do
  gate="$gate_root/$model/$method/t1_t2_audit.json"
  if ! /opt/miniconda3/bin/python - "$gate" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('passed') else 1)
PY
  then
   /opt/miniconda3/bin/python - "$state" "$model" "$method" "$gate" <<'PY'
import datetime,json,sys
p,m,k,g=sys.argv[1:]
with open(p,'a') as f: f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':m,'method':k,'status':'N/A','reason':f'T1/T2 failed; evidence={g}'})+'\n')
PY
   continue
  fi
  python=$(runtime_for "$model")
  for dataset in visual_mimic_oe iu_xray_report mimic_cxr_report vqa_rad_official_oe cxr_vishal knowledge_mimic_ce slake_fine_grained; do
   IFS='|' read -r manifest images tokens <<<"$(contract "$dataset")"
   out="$root/$model/$method/$dataset"; log="$jobs/logs/baseline-${model}-${method}-${dataset}.log"; mkdir -p "$out"
   rc=1
   flock 8
   for attempt in 1 2 3; do
    PYTHONPATH=. "$python" -m anchor.corrected_sgta.run_cross_model_method_full_v1 \
      --model "$model" --method "$method" --manifest "$manifest" --image-root "$images" \
      --output-dir "$out" --max-new-tokens "$tokens" --seed 42 >>"$log" 2>&1
    rc=$?; [[ "$rc" -eq 0 ]] && break
    tail -80 "$log" | grep -q 'refusing to resume an incompatible' && break
   done
   flock -u 8
   /opt/miniconda3/bin/python - "$state" "$model" "$method" "$dataset" "$rc" "$log" <<'PY'
import datetime,json,sys
p,m,k,d,r,l=sys.argv[1:]
with open(p,'a') as f: f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':m,'method':k,'dataset':d,'status':'generated' if r=='0' else 'failed','returncode':int(r),'log':l})+'\n')
PY
  done
 done
done

#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

jobs=corrected_runs/detached_jobs
out=corrected_runs/paper_baselines_v1/full_matrix_v1/llava_methods
state="$jobs/baseline-llava-methods-long-queue-v1.state.jsonl"
mkdir -p "$jobs/locks" "$jobs/logs" "$out"
exec 8>"$jobs/locks/gpu0-vindr-v2.lock"

gate_root=corrected_runs/paper_baselines_v1/full_matrix_v1/gates/llava_methods
gate_provenance="$gate_root/gate_provenance.json"
mkdir -p "$gate_root"
gate_tag=$(/opt/miniconda3/envs/huatuo/bin/python - "$gate_provenance" <<'PY'
import hashlib,importlib.metadata,json,sys
from pathlib import Path
root=Path('/home/dbw/ANCHOR')
paths = [
    root/'anchor/medeval/run_native_oe_vqa.py',
    root/'anchor/corrected_sgta/run_llava_med_generation_matrix.py',
    root/'data/medheval/code/baselines/Mitigation/llava-med-1.5/llava/eval/model_vqa.py',
]
config=root/'configs/unified_eval/baseline_matrix_v1.json'
manifest=root/'corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json'
checkpoint=Path('/home/dbw/models/LLaVA-Med-v1.5-mistral-7b')
small=sorted(p for p in checkpoint.iterdir() if p.is_file() and p.suffix not in {'.safetensors','.bin'})
weights=sorted(({'path':p.name,'bytes':p.stat().st_size} for p in checkpoint.iterdir() if p.is_file() and p.suffix in {'.safetensors','.bin'}),key=lambda row:row['path'])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def version(name):
 try:return importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:return None
deps={name:version(name) for name in ('torch','transformers','accelerate','numpy')}
payload={
 'protocol':'baseline-gate-provenance-v1',
 'sources':[{'path':str(p.resolve()),'sha256':sha(p)} for p in paths],
 'frozen_config':{'path':str(config.resolve()),'sha256':sha(config)},
 'gate_manifest':{'path':str(manifest.resolve()),'sha256':sha(manifest)},
 'checkpoint':{'path':str(checkpoint.resolve()),'metadata_files':[{'path':p.name,'sha256':sha(p)} for p in small],'weight_inventory':weights},
 'dependencies':deps,
 'generation':{'limit':32,'max_new_tokens':256,'seed':42,'conv_mode':'mistral_instruct','keyword_stopping':False},
}
payload['fingerprint']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
target=Path(sys.argv[1]); target.write_text(json.dumps(payload,indent=2)+'\n')
print(payload['fingerprint'][:16])
PY
)
gate_run="$gate_root/runs/$gate_tag"
gate_summary="$gate_root/t1_t2_audit.json"
gate_current=0
if [[ -f "$gate_summary" ]]; then
  /opt/miniconda3/bin/python - "$gate_summary" "$gate_tag" <<'PY' && gate_current=1
import json,sys
d=json.load(open(sys.argv[1]))
raise SystemExit(0 if d.get('source_tag') == sys.argv[2] and d.get('all_evaluated') else 1)
PY
fi
if [[ "$gate_current" -ne 1 ]]; then
  flock 8
  mkdir -p "$gate_run" "$gate_root"
  gate_manifest=corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json
  gate_images=/home/dbw/datasets/public/vqa_rad_hf/test_images
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.run_native_oe_vqa \
    --model llava --manifest "$gate_manifest" --image-root "$gate_images" \
    --output-dir "$gate_run/canonical" --limit 32 --max-new-tokens 256 --seed 42 \
    >"$jobs/logs/baseline-llava-current-gate-canonical.log" 2>&1
  canonical_rc=$?
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.corrected_sgta.run_llava_med_generation_matrix \
    --question-file "$gate_manifest" --image-folder "$gate_images" --out "$gate_run/common" \
    --source vqa_rad --dataset current_source_gate --task open_vqa \
    --methods greedy VCD DoLa opera PAI avisc VISTA \
    --chunk-size 32 --limit 32 --max-new-tokens 256 --seed 42 \
    --conv-mode mistral_instruct --disable-keyword-stopping --qualification-run --continue-on-error \
    >"$jobs/logs/baseline-llava-current-method-gates.log" 2>&1
  common_rc=$?
  /opt/miniconda3/bin/python - "$gate_root" "$gate_run" "$gate_tag" "$canonical_rc" "$common_rc" "$gate_provenance" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,run=map(Path,sys.argv[1:3]); tag=sys.argv[3]; canonical_rc,common_rc=map(int,sys.argv[4:6]); provenance=Path(sys.argv[6])
methods=['VCD','DoLa','opera','PAI','avisc','VISTA']
def load(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def ids(row): return list(row.get('metadata',{}).get('generated_token_ids',[]))
canonical=run/'canonical/answers.jsonl'
base=run/'common/vqa_rad/current_source_gate/open_vqa/greedy/chunk_0000.answers.jsonl'
summary={'protocol':'llava-current-source-t1-t2-v1','source_tag':tag,'gate_provenance':str(provenance),'gate_provenance_sha256':hashlib.sha256(provenance.read_bytes()).hexdigest(),'canonical_returncode':canonical_rc,'common_returncode':common_rc,'canonical':str(canonical),'common_greedy':str(base),'methods':{}}
try:
    native,greedy=load(canonical),load(base)
    aligned=len(native)==len(greedy)==32 and [str(x['question_id']) for x in native]==[str(x['question_id']) for x in greedy]
    exact=sum(ids(a)==ids(b) and bool(ids(a)) for a,b in zip(native,greedy)) if aligned else 0
    summary['t1']={'n_native':len(native),'n_common':len(greedy),'qid_aligned':aligned,'token_exact':exact,'passed':canonical_rc==0 and aligned and exact==32,'canonical_sha256':hashlib.sha256(canonical.read_bytes()).hexdigest(),'common_sha256':hashlib.sha256(base.read_bytes()).hexdigest()}
except Exception as error:
    greedy=[]; summary['t1']={'passed':False,'error':f'{type(error).__name__}: {error}'}
for method in methods:
    answer=run/f'common/vqa_rad/current_source_gate/open_vqa/{method}/chunk_0000.answers.jsonl'
    meta=answer.with_name('chunk_0000.meta.json')
    result={'answers':str(answer),'meta':str(meta),'passed':False}
    try:
        rows,metadata=load(answer),json.loads(meta.read_text()); audit=metadata.get('output_audit') or {}
        aligned=len(greedy)==len(rows)==32 and [str(x['question_id']) for x in greedy]==[str(x['question_id']) for x in rows]
        changed=sum(ids(a)!=ids(b) for a,b in zip(greedy,rows)) if aligned else 0
        result.update({'n':len(rows),'qid_aligned':aligned,'changed_sequences':changed,'generated_token_ids_available':all(bool(ids(x)) for x in rows),'output_aligned':audit.get('aligned'),'degenerate_reasons':audit.get('degenerate_reasons',[]),'answers_sha256':hashlib.sha256(answer.read_bytes()).hexdigest(),'passed':summary['t1'].get('passed') is True and aligned and changed>=1 and all(bool(ids(x)) for x in rows) and audit.get('aligned') is True and not audit.get('degenerate_reasons')})
    except Exception as error: result['error']=f'{type(error).__name__}: {error}'
    summary['methods'][method]=result
    target=root/method/'t1_t2_audit.json'; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps({'protocol':summary['protocol'],'source_tag':tag,'gate_provenance':str(provenance),'gate_provenance_sha256':summary['gate_provenance_sha256'],'t1':summary.get('t1'),'t2':result,'passed':result['passed']},indent=2)+'\n')
summary['all_evaluated']=all('error' not in row for row in summary['methods'].values())
summary['passed_methods']=[m for m,r in summary['methods'].items() if r['passed']]
summary['failed_methods']=[m for m,r in summary['methods'].items() if not r['passed']]
(root/'t1_t2_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
PY
  flock -u 8
fi

bash scripts/wait_for_all_baseline_gates_v1.sh llava

llava_methods=()
for method in VCD DoLa opera PAI avisc VISTA; do
  if /opt/miniconda3/bin/python - "$gate_root/$method/t1_t2_audit.json" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('passed') else 1)
PY
  then
    llava_methods+=("$method")
  else
    /opt/miniconda3/bin/python - "$state" "$method" "$gate_root/$method/t1_t2_audit.json" <<'PY'
import datetime,json,sys
with open(sys.argv[1],'a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':sys.argv[2],'status':'N/A','reason':f'current-source T1/T2 failed; evidence={sys.argv[3]}'})+'\n')
PY
  fi
done
if [[ "${#llava_methods[@]}" -eq 0 ]]; then
  /opt/miniconda3/bin/python - "$state" "$gate_summary" <<'PY'
import datetime,json,sys
with open(sys.argv[1],'a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'queue_stopped','reason':f'no LLaVA methods passed current-source T1/T2; evidence={sys.argv[2]}'})+'\n')
PY
  exit 0
fi

run_dataset() {
  local dataset=$1 task=$2 manifest=$3 images=$4 source=$5 tokens=${6:-256}
  local log="$jobs/logs/baseline-llava-methods-${dataset}.log" rc=1 attempt
  flock 8
  for attempt in 1 2 3; do
    PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
      -m anchor.corrected_sgta.run_llava_med_generation_matrix \
      --question-file "$manifest" --image-folder "$images" --out "$out" \
      --source "$source" --dataset "$dataset" --task "$task" \
      --methods "${llava_methods[@]}" \
      --chunk-size 64 --max-new-tokens "$tokens" --seed 42 \
      --conv-mode mistral_instruct --disable-keyword-stopping \
      --continue-on-error >>"$log" 2>&1
    rc=$?
    [[ "$rc" -eq 0 ]] && break
  done
  flock -u 8
  /opt/miniconda3/bin/python - "$state" "$dataset" "$rc" "$attempt" "$log" <<'PY'
import datetime,json,sys
path,dataset,rc,attempt,log=sys.argv[1:]
row={"time":datetime.datetime.now(datetime.timezone.utc).isoformat(),"dataset":dataset,"status":"completed" if rc=="0" else "failed","returncode":int(rc),"attempts":int(attempt),"log":log}
with open(path,"a") as f: f.write(json.dumps(row)+"\n")
PY
}

run_dataset visual_mimic_oe open_vqa \
  corrected_runs/unified_eval/inputs/baseline_matrix_v1/visual_mimic_oe.json \
  /home/dbw/ANCHOR/data/medheval/images medheval
run_dataset iu_xray_report report_generation \
  corrected_runs/unified_eval/inputs/baseline_matrix_v1/iu_xray_report.json \
  /home/dbw/ANCHOR/data/medheval/images/IU-Xray mmedrag
run_dataset mimic_cxr_report report_generation \
  corrected_runs/unified_eval/inputs/baseline_matrix_v1/mimic_cxr_report.json \
  /home/dbw/ANCHOR/data/medheval/images mmedrag
run_dataset vqa_rad_official_oe open_vqa \
  corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json \
  /home/dbw/datasets/public/vqa_rad_hf/test_images vqa_rad
run_dataset cxr_vishal close_vqa \
  corrected_runs/unified_eval/inputs/baseline_matrix_v1/cxr_vishal.json \
  /home/dbw/ANCHOR/data/medheval/images medheval 128
run_dataset knowledge_mimic_ce close_vqa \
  corrected_runs/unified_eval/inputs/baseline_matrix_v1/knowledge_mimic_ce.json \
  /home/dbw/ANCHOR/data/medheval/images medheval 128
run_dataset slake_fine_grained close_vqa \
  corrected_runs/unified_eval/inputs/baseline_matrix_v1/slake_fine_grained.json \
  /home/dbw/ANCHOR/data/medheval/images/Slake medheval 128

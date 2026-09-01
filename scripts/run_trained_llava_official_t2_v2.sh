#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/dbw/.cache/huggingface
root=corrected_runs/paper_baselines_v1/trained_llava_t2_v2
manifest=corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json
official_manifest="$root/official_questions.jsonl"
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
base=/home/dbw/models/llava-v1.5-7b
log=corrected_runs/detached_jobs/logs/baseline-trained-official-t2-v2.log
mkdir -p "$root"
PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.prepare_official_llava_t2_manifest_v1 --manifest "$manifest" --output "$official_manifest" --limit 32 >>"$log" 2>&1
exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
variants=(base ha-dpo opa-dpo da-dpo sentinel less-is-more factmm-rag-generator)
for variant in "${variants[@]}"; do
 flock 8
 PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.corrected_sgta.run_trained_llava_baseline_v1 --variant "$variant" --manifest "$manifest" --image-root "$images" --output-dir "$root/unified/$variant" --limit 32 --max-new-tokens 256 --seed 42 >>"$log" 2>&1
 variant_rc=$?
 flock -u 8
 [[ "$variant_rc" -ne 0 ]] && continue
done
ha_root=third_party/training_baselines/HA-DPO/ha_dpo/models/llava-v1_5
declare -A paths=( [base]="$base" [ha-dpo]="/home/dbw/models/hadpo-llava-1.5" [da-dpo]="/home/dbw/models/da-dpo-llava-v1.5-7b" [sentinel]="/home/dbw/models/sentinel-llava-v1.5-7b" [less-is-more]="/home/dbw/models/less-is-more-llava-v1.5-7b" [factmm-rag-generator]="/home/dbw/models/factmm-rag-generator-v1" )
for variant in base ha-dpo da-dpo sentinel less-is-more factmm-rag-generator; do
 out="$root/official/$variant/answers.jsonl"; mkdir -p "$(dirname "$out")"
 [[ -f "$out" && "$(wc -l < "$out")" -ne 32 ]] && mv "$out" "$out.incomplete_v2"
 [[ -f "$out" && "$(wc -l < "$out")" -eq 32 ]] && continue
 model_base="$base"; [[ "$variant" == base || "$variant" == factmm-rag-generator ]] && model_base=""
 cmd=(/opt/miniconda3/envs/huatuo/bin/python -m llava.eval.model_vqa_loader --model-path "${paths[$variant]}" --image-folder "$images" --question-file "$official_manifest" --answers-file "$out" --conv-mode llava_v1 --num-chunks 1 --chunk-idx 0 --temperature 0 --top_p 1 --num_beams 1 --max_new_tokens 256)
 [[ -n "$model_base" ]] && cmd+=(--model-base "$model_base")
flock 8
PYTHONPATH="anchor/medeval/llava_legacy_sitecustomize:$ha_root" "${cmd[@]}" >>"$log" 2>&1 || true
flock -u 8
done
opa_root=third_party/training_baselines/OPA-DPO/eval_llava_rlhf_coco
opa_llava=third_party/training_baselines/OPA-DPO/llava_setup/LLaVA
opa_repo=third_party/training_baselines/OPA-DPO
opa_out="$root/official/opa-dpo/answers.jsonl"; mkdir -p "$(dirname "$opa_out")"
[[ -f "$opa_out" && "$(wc -l < "$opa_out")" -ne 32 ]] && mv "$opa_out" "$opa_out.incomplete_v2"
if [[ ! -f "$opa_out" ]]; then
flock 8
PYTHONPATH="anchor/medeval/llava_legacy_sitecustomize:$opa_repo:$opa_llava" /opt/miniconda3/envs/huatuo/bin/python "$opa_root/model_vqa.py" --model-path "$base" --qlora-path /home/dbw/models/opadpo-lora-llava-v1.5-7b --use-qlora True --image-folder "$images" --question-file "$official_manifest" --answers-file "$opa_out" --conv-mode llava_v1 --temperature 0 --top_p 1 --num_beams 1 --max-new-tokens 256 --test-prompt "" >>"$log" 2>&1 || true
flock -u 8
fi
/opt/miniconda3/bin/python - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);variants=['base','ha-dpo','opa-dpo','da-dpo','sentinel','less-is-more','factmm-rag-generator']; rows=[]
def load(path):return [json.loads(x) for x in open(path) if x.strip()]
def canon(row):
 ids=list(row.get('metadata',{}).get('generated_token_ids',[]))
 while ids and ids[-1] in {2}:ids.pop()
 return ids
for v in variants:
 u=root/'unified'/v/'answers.jsonl';o=root/'official'/v/'answers.jsonl'
 if not u.is_file() or not o.is_file(): rows.append({'variant':v,'status':'failed_missing_artifact'});continue
 ur,orr=load(u),load(o); aligned=len(ur)==len(orr)==32 and [str(x['question_id']) for x in ur]==[str(x['question_id']) for x in orr]
 exact=sum(canon(a)==canon(b) for a,b in zip(ur,orr)) if aligned else 0
 rows.append({'variant':v,'n_unified':len(ur),'n_official':len(orr),'qid_aligned':aligned,'content_token_exact':exact,'passed':aligned and exact==32,'unified_sha256':hashlib.sha256(u.read_bytes()).hexdigest(),'official_sha256':hashlib.sha256(o.read_bytes()).hexdigest()})
def digest(path):
 p=Path(path);return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
result={'protocol':'trained-llava-official-t2-v2','n':32,'rows':rows,'passed_variants':[x['variant'] for x in rows if x.get('passed')],'failed_variants':[x['variant'] for x in rows if not x.get('passed')],
 'runtime_provenance':{
  'unified_runner_sha256':digest('anchor/corrected_sgta/run_trained_llava_baseline_v1.py'),
  'legacy_import_shim_sha256':digest('anchor/medeval/llava_legacy_sitecustomize/sitecustomize.py'),
  'ha_dpo_official_entry_sha256':digest('third_party/training_baselines/HA-DPO/ha_dpo/models/llava-v1_5/llava/eval/model_vqa_loader.py'),
  'opa_dpo_official_entry_sha256':digest('third_party/training_baselines/OPA-DPO/eval_llava_rlhf_coco/model_vqa.py'),
  'opa_dpo_llava_commit':'817a4af4e7323dd392b9bcc723cf5844c1272896',
  'opa_dpo_released_patch_sha256':digest('third_party/training_baselines/OPA-DPO/llava_setup/llava_modifications.patch')}}
(root/'t2_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
PY

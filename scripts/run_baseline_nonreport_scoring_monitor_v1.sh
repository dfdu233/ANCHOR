#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR
root=corrected_runs/paper_baselines_v1/full_matrix_v1
inputs=corrected_runs/unified_eval/inputs/baseline_matrix_v1
state=corrected_runs/detached_jobs/baseline-nonreport-scoring-monitor-v1.state.jsonl
archive_stale() {
 local out=$1; shift; local stamp archive file
 stamp=$(date -u +%Y%m%dT%H%M%SZ); archive="$out/stale_artifacts/$stamp"; mkdir -p "$archive"
 for file in "$@"; do [[ -f "$file" ]] && mv "$file" "$archive/"; done
}
score() {
 local key=$1 dataset=$2 task=$3 expected=$4 tokens=$5; shift 5; local files=("$@") n=0 f combined out rc
 for f in "${files[@]}";do [[ -f "$f" ]]||return 0;n=$((n+$(wc -l <"$f")));done
 [[ "$n" -eq "$expected" ]]||return 0
 out="$root/derived_scores/$key";mkdir -p "$out";combined="$out/answers.jsonl"
 [[ "$task" == ce ]]&&metric="$out/evaluation_ce_v7.json"||metric="$out/evaluation_lexical_auxiliary.json"
 if [[ -f "$metric" ]] && PYTHONPATH=. /opt/miniconda3/bin/python -m anchor.medeval.validate_score_artifact_binding_v1 \
   --score "$metric" --qualification "$out/qualification.json" \
   --task "$([[ "$task" == ce ]] && echo mixed_ce || echo open_vqa)" --expected "$expected" \
   >/dev/null 2>&1; then return 0; fi
 [[ -f "$metric" ]] && archive_stale "$out" "$metric" "$out/qualification.json"
 PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.combine_answer_chunks_v1 --manifest "$inputs/$dataset.json" --answers "${files[@]}" --output "$combined" >"$out/score.log" 2>&1;rc=$?
 if [[ "$rc" -eq 0 ]]; then
  if [[ "$task" == ce ]]; then
   PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_ce_generation \
    --manifest "$inputs/$dataset.json" --answers "$combined" --limit "$expected" \
    --max-new-tokens "$tokens" --min-parse-rate 0.90 --output "$out/qualification.json" >>"$out/score.log" 2>&1; rc=$?
  else
   PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_oe_generation \
    --manifest "$inputs/$dataset.json" --answers "$combined" --limit "$expected" \
    --max-new-tokens "$tokens" --max-cap-hit-rate 0.05 --output "$out/qualification.json" >>"$out/score.log" 2>&1; rc=$?
  fi
 fi
 if [[ "$rc" -eq 0 && "$task" == ce ]];then PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.corrected_sgta.evaluate_medheval_answers --answers "$combined" --questions "$inputs/$dataset.json" --output "$metric" >>"$out/score.log" 2>&1;rc=$?;fi
 if [[ "$rc" -eq 0 && "$task" == oe ]];then PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.evaluate_oe_vqa --manifest "$inputs/$dataset.json" --answers "$combined" --output "$metric" --bootstrap-replicates 5000 --seed 42 --max-new-tokens "$tokens" >>"$out/score.log" 2>&1;rc=$?;fi
 /opt/miniconda3/bin/python - "$state" "$key" "$rc" "$n" <<'PY'
import datetime,json,sys
p,k,r,n=sys.argv[1:]
with open(p,'a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cell':k,'status':'completed' if r=='0' else 'failed','returncode':int(r),'rows':int(n)})+'\n')
PY
}
score_trained() {
 local variant=$1 dataset=$2 task=$3 expected=$4 tokens=$5
 local out="$root/trained_llava15/$variant/$dataset" answers="$root/trained_llava15/$variant/$dataset/answers.jsonl" checked rc n=0 metric
 [[ -f "$answers" ]] || return 0
 n=$(wc -l < "$answers"); [[ "$n" -eq "$expected" ]] || return 0
 [[ "$task" == ce ]] && metric="$out/evaluation_ce_v7.json" || metric="$out/evaluation_lexical_auxiliary.json"
 if [[ -f "$metric" ]] && PYTHONPATH=. /opt/miniconda3/bin/python -m anchor.medeval.validate_score_artifact_binding_v1 \
   --score "$metric" --qualification "$out/qualification.json" \
   --task "$([[ "$task" == ce ]] && echo mixed_ce || echo open_vqa)" --expected "$expected" \
   >/dev/null 2>&1; then return 0; fi
 [[ -f "$metric" ]] && archive_stale "$out" "$metric" "$out/qualification.json"
 checked="$out/scoring_answers.jsonl"
 PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.combine_answer_chunks_v1 \
   --manifest "$inputs/$dataset.json" --answers "$answers" --output "$checked" >"$out/score.log" 2>&1; rc=$?
 if [[ "$rc" -eq 0 ]]; then
  if [[ "$task" == ce ]]; then
   PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_ce_generation \
    --manifest "$inputs/$dataset.json" --answers "$checked" --limit "$expected" \
    --max-new-tokens "$tokens" --min-parse-rate 0.90 --output "$out/qualification.json" >>"$out/score.log" 2>&1; rc=$?
  else
   PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_oe_generation \
    --manifest "$inputs/$dataset.json" --answers "$checked" --limit "$expected" \
    --max-new-tokens "$tokens" --max-cap-hit-rate 0.05 --output "$out/qualification.json" >>"$out/score.log" 2>&1; rc=$?
  fi
 fi
 if [[ "$rc" -eq 0 && "$task" == ce ]]; then
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.corrected_sgta.evaluate_medheval_answers \
   --answers "$checked" --questions "$inputs/$dataset.json" --output "$metric" >>"$out/score.log" 2>&1; rc=$?
 fi
 if [[ "$rc" -eq 0 && "$task" == oe ]]; then
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.evaluate_oe_vqa \
   --manifest "$inputs/$dataset.json" --answers "$checked" --output "$metric" \
   --bootstrap-replicates 5000 --seed 42 --max-new-tokens "$tokens" >>"$out/score.log" 2>&1; rc=$?
 fi
 /opt/miniconda3/bin/python - "$state" "trained/$variant/$dataset" "$rc" "$n" <<'PY'
import datetime,json,sys
p,k,r,n=sys.argv[1:]
with open(p,'a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cell':k,'status':'completed' if r=='0' else 'failed','returncode':int(r),'rows':int(n)})+'\n')
PY
}
while true;do
 for spec in 'cxr_vishal|ce|5587|128' 'knowledge_mimic_ce|ce|2000|128' 'slake_fine_grained|ce|1536|128' 'vqa_rad_official_oe|oe|200|256' 'visual_mimic_oe|oe|490|256';do
  IFS='|' read -r dataset task expected tokens<<<"$spec"
  source=medheval;taskdir=close_vqa;[[ "$task" == oe ]]&&taskdir=open_vqa;[[ "$dataset" == vqa_rad_official_oe ]]&&source=vqa_rad
  for method in VCD DoLa opera PAI avisc VISTA;do
   paper_method=$method;[[ "$method" == opera ]]&&paper_method=OPERA;[[ "$method" == avisc ]]&&paper_method=AvisC
   folder="$root/llava_methods/$source/$dataset/$taskdir/$method";mapfile -t chunks < <(find "$folder" -maxdepth 1 -name 'chunk_*.answers.jsonl' -type f 2>/dev/null|sort);[[ ${#chunks[@]} -gt 0 ]]&&score "llava/$paper_method/$dataset" "$dataset" "$task" "$expected" "$tokens" "${chunks[@]}"
  done
  for model in huatuo hulu qwen;do for method in vcd dola;do paper_method=VCD;[[ "$method" == dola ]]&&paper_method=DoLa;f="$root/cross_model_methods/$model/$method/$dataset/answers.jsonl";[[ -f "$f" ]]&&score "$model/$paper_method/$dataset" "$dataset" "$task" "$expected" "$tokens" "$f";done;done
  for variant in base ha-dpo opa-dpo da-dpo sentinel less-is-more factmm-rag-generator vhr; do
   score_trained "$variant" "$dataset" "$task" "$expected" "$tokens"
  done
 done
 sleep 300
done

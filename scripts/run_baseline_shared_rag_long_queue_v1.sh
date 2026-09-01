#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
jobs=corrected_runs/detached_jobs
prompt_root=corrected_runs/paper_baselines_v1/full_matrix_v1/rag/bm25
root=corrected_runs/paper_baselines_v1/full_matrix_v1/shared_rag_generation
state="$jobs/baseline-shared-rag-long-queue-v1.state.jsonl"
mkdir -p "$root" "$jobs/locks" "$jobs/logs"

for dataset in cxr_vishal knowledge_mimic_ce slake_fine_grained vqa_rad_official_oe visual_mimic_oe; do
  while [[ ! -f "$prompt_root/$dataset/prompt_manifest.json" || ! -f "$prompt_root/$dataset/no_context.json" ]]; do sleep 30; done
done
bash scripts/wait_for_all_baseline_gates_v1.sh rag
exec 8>"$jobs/locks/gpu0-vindr-v2.lock"

runtime_for() { case "$1" in hulu) echo /home/dbw/.venvs/hulumed/bin/python;; qwen) echo /home/dbw/.venvs/qwen25vl-v2/bin/python;; *) echo /opt/miniconda3/envs/huatuo/bin/python;; esac; }
contract() {
 case "$1" in
  cxr_vishal) echo "data/medheval/images|5587|128|ce";;
  knowledge_mimic_ce) echo "data/medheval/images|2000|128|ce";;
  slake_fine_grained) echo "data/medheval/images/Slake|1536|128|ce";;
  vqa_rad_official_oe) echo "/home/dbw/datasets/public/vqa_rad_hf/test_images|200|256|oe";;
  visual_mimic_oe) echo "data/medheval/images|490|256|oe";;
 esac
}

for model in huatuo hulu llava qwen; do
 python=$(runtime_for "$model")
 for dataset in visual_mimic_oe vqa_rad_official_oe cxr_vishal knowledge_mimic_ce slake_fine_grained; do
  IFS='|' read -r images expected tokens task <<<"$(contract "$dataset")"
  for condition in no_context rag; do
   manifest="$prompt_root/$dataset/${condition}.json"; out="$root/$model/$dataset/$condition"; log="$jobs/logs/baseline-rag-${model}-${dataset}-${condition}.log"; mkdir -p "$out"
   rc=1
   flock 8
   for attempt in 1 2 3; do
    PYTHONPATH=. "$python" -m anchor.medeval.run_native_oe_vqa --model "$model" \
      --manifest "$manifest" --image-root "$images" --output-dir "$out" --limit 0 \
      --max-new-tokens "$tokens" --seed 42 --decode-mode greedy --num-beams 1 >>"$log" 2>&1
    rc=$?; [[ "$rc" -eq 0 ]] && break
    tail -80 "$log" | grep -q 'refusing to resume an incompatible' && break
   done
   flock -u 8
   if [[ "$rc" -eq 0 ]]; then
    if [[ "$task" == ce ]]; then
      PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_ce_generation \
        --manifest "$manifest" --answers "$out/answers.jsonl" --limit "$expected" \
        --max-new-tokens "$tokens" --min-parse-rate 0.90 \
        --output "$out/qualification.json" >>"$log" 2>&1; rc=$?
    else
      PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_oe_generation \
        --manifest "$manifest" --answers "$out/answers.jsonl" --limit "$expected" \
        --max-new-tokens "$tokens" --max-cap-hit-rate 0.05 \
        --output "$out/qualification.json" >>"$log" 2>&1; rc=$?
    fi
   fi
   if [[ "$rc" -eq 0 && "$task" == ce ]]; then
    PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.corrected_sgta.evaluate_medheval_answers --answers "$out/answers.jsonl" --questions "$manifest" --output "$out/evaluation_ce_v7.json" >>"$log" 2>&1; rc=$?
   elif [[ "$rc" -eq 0 ]]; then
    PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.evaluate_oe_vqa --manifest "$manifest" --answers "$out/answers.jsonl" --output "$out/evaluation_lexical_auxiliary.json" --bootstrap-replicates 5000 --seed 42 --max-new-tokens "$tokens" >>"$log" 2>&1; rc=$?
   fi
   /opt/miniconda3/bin/python - "$state" "$model" "$dataset" "$condition" "$rc" "$expected" "$log" <<'PY'
import datetime,json,sys
p,m,d,c,r,n,l=sys.argv[1:]
with open(p,'a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':m,'dataset':d,'condition':c,'status':'completed' if r=='0' else 'failed','returncode':int(r),'expected':int(n),'log':l})+'\n')
PY
  done
 done
done

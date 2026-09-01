#!/usr/bin/env bash
# Resumable post-baseline matrix for every native model and inference-only arm.
# It takes the shared GPU lock, so starting early cannot pre-empt Baseline.
set -uo pipefail
cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
jobs=corrected_runs/detached_jobs
root=corrected_runs/paper_baselines_v1/full_matrix_v1/cross_model_complete_v1
state="$jobs/cross-model-complete-v1.state.jsonl"
mkdir -p "$root" "$jobs/locks" "$jobs/logs"
# Baseline queues own the GPU first.  Waiting on their supervisor processes
# prevents this optional completion matrix from interleaving with references.
while pgrep -f '^bash scripts/run_baseline_(native_ce_long_queue_v1|native_long_queue_v1|cross_model_methods_long_queue_v1)\.sh$' >/dev/null; do
  sleep 30
done
exec 8>"$jobs/locks/gpu0-vindr-v2.lock"

runtime_for() {
  case "$1" in
    hulu) echo /home/dbw/.venvs/hulumed/bin/python ;;
    qwen|llava16) echo /home/dbw/.venvs/qwen25vl-v2/bin/python ;;
    huatuo|llava) echo /opt/miniconda3/envs/huatuo/bin/python ;;
  esac
}
contract() {
  case "$1" in
    omnimedvqa) echo "data/omnimedvqa/eight_modality_smoke64_v1.json|/home/dbw/datasets/public/OmniMedVQA/extracted/OmniMedVQA|64" ;;
    pmcvqa) echo "data/pmcvqa/smoke64_v1.json|/home/dbw/datasets/public/PMC-VQA|64" ;;
    pathvqa) echo "data/pathvqa/smoke64_v1.json|/home/dbw/datasets/public/path-vqa|64" ;;
    mmmu_medical) echo "data/mmmu/smoke37_v1.json|/home/dbw/datasets/public/MMMU/medical_images|37" ;;
    vqa_rad_official_oe) echo "corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json|/home/dbw/datasets/public/vqa_rad_hf/test_images|200" ;;
  esac
}
record() {
  /opt/miniconda3/bin/python - "$state" "$@" <<'PY'
import datetime,json,sys
p,status,model,method,dataset,rc=sys.argv[1:]
with open(p,'a') as f: f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'model':model,'method':method,'dataset':dataset,'returncode':int(rc)})+'\n')
PY
}

datasets=(omnimedvqa pmcvqa pathvqa mmmu_medical vqa_rad_official_oe)
models=(huatuo hulu llava llava16 qwen)
methods=(greedy vcd icd cve agla avisc clearsight dola)
# Build one BLIP-ITM prompt-matched image per question, shared by all target
# models. This preserves AGLA semantics without mixing incompatible
# Transformers versions into the VLM runtimes.
for dataset in "${datasets[@]}"; do
  IFS='|' read -r manifest images expected <<<"$(contract "$dataset")"
  agla_root="corrected_runs/paper_baselines_v1/agla_prompt_match_v1/$dataset"
  if [[ ! -f "$agla_root/manifest.json" ]] || [[ "$(/opt/miniconda3/bin/python -c 'import json,sys;print(len(json.load(open(sys.argv[1]))))' "$agla_root/manifest.json" 2>/dev/null || echo 0)" -ne "$expected" ]]; then
    flock 8
    PYTHONPATH=. /home/dbw/.venvs/agla-blip/bin/python -m anchor.medeval.prepare_agla_augmented \
      --manifest "$manifest" --image-root "$images" --output-dir "$agla_root" \
      >>"$jobs/logs/agla-prompt-match-${dataset}.log" 2>&1
    rc=$?
    flock -u 8
    if [[ "$rc" -ne 0 ]]; then
      record failed blip_itm agla "$dataset" "$rc"
    fi
  fi
done
for dataset in "${datasets[@]}"; do
  IFS='|' read -r manifest images expected <<<"$(contract "$dataset")"
  for model in "${models[@]}"; do
    python=$(runtime_for "$model")
    for method in "${methods[@]}"; do
      run_manifest="$manifest"
      if [[ "$method" == agla && -f "corrected_runs/paper_baselines_v1/agla_prompt_match_v1/$dataset/manifest.json" ]]; then
        run_manifest="corrected_runs/paper_baselines_v1/agla_prompt_match_v1/$dataset/manifest.json"
      fi
      out="$root/$model/$dataset/$method"; log="$jobs/logs/cross-complete-${model}-${dataset}-${method}.log"
      mkdir -p "$out"
      got=0; [[ -f "$out/answers.jsonl" ]] && got=$(wc -l < "$out/answers.jsonl")
      rc=0
      if [[ "$got" -ne "$expected" ]]; then
        record running "$model" "$method" "$dataset" 0
        flock 8
        PYTHONPATH=. "$python" -m anchor.corrected_sgta.run_cross_model_method_full_v1 \
          --model "$model" --method "$method" --manifest "$run_manifest" --image-root "$images" \
          --output-dir "$out" --max-new-tokens 64 --seed 42 >>"$log" 2>&1
        rc=$?
        flock -u 8
      fi
      if [[ "$rc" -eq 0 && -f "$out/answers.jsonl" ]]; then
        PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.corrected_sgta.evaluate_medheval_answers \
          --answers "$out/answers.jsonl" --questions "$run_manifest" --output "$out/evaluation_strict.json" \
          --bootstrap-replicates 500 --bootstrap-seed 42 >>"$log" 2>&1 || rc=$?
      fi
      if [[ "$rc" -eq 0 ]]; then status=completed; else status=failed; fi
      record "$status" "$model" "$method" "$dataset" "$rc"
    done
  done
done
record queue_completed all all all 0

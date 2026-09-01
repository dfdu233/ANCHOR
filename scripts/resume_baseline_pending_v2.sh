#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HOME=/home/dbw/.cache/huggingface

root=corrected_runs/paper_baselines_v1/full_matrix_v1
jobs=corrected_runs/detached_jobs
lock="$jobs/locks/gpu0-vindr-v2.lock"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$jobs/locks" "$jobs/logs"

exec 8>"$lock"
flock 8

run_one() {
  local model manifest images out expected tokens beams log count prior_tokens archive
  model=$1
  manifest=$2
  images=$3
  out=$4
  expected=$5
  tokens=$6
  beams=$7
  log="$jobs/logs/baseline-recovery-${model}-$(basename "$(dirname "$out")")-$(basename "$out").log"
  mkdir -p "$out"
  count=0
  [[ -f "$out/answers.jsonl" ]] && count=$(wc -l <"$out/answers.jsonl")
  if [[ "$count" -eq "$expected" ]]; then
    echo "already complete rows=$count out=$out" >>"$log"
    return 0
  fi
  if [[ "$count" -eq 0 && -f "$out/generation_config.json" ]]; then
    prior_tokens=$(/opt/miniconda3/bin/python - "$out/generation_config.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get("max_new_tokens", ""))
PY
)
    if [[ "$prior_tokens" != "$tokens" ]]; then
      archive="$out/stale_artifacts/$stamp"
      mkdir -p "$archive"
      mv "$out/generation_config.json" "$archive/generation_config.json"
    fi
  fi
  echo "start model=$model manifest=$manifest rows=$count/$expected tokens=$tokens" >>"$log"
  if ! PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.run_native_oe_vqa \
      --model "$model" --manifest "$manifest" --image-root "$images" \
      --output-dir "$out" --limit 0 --max-new-tokens "$tokens" --seed 42 \
      --decode-mode "$([[ "$beams" -gt 1 ]] && echo beam || echo greedy)" \
      --num-beams "$beams" >>"$log" 2>&1; then
    echo "generation failed out=$out" >>"$log"
    return 0
  fi
  if ! PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_oe_generation \
      --manifest "$manifest" --answers "$out/answers.jsonl" --limit "$expected" \
      --max-new-tokens "$tokens" --output "$out/qualification.json" >>"$log" 2>&1; then
    echo "structural qualification failed out=$out" >>"$log"
    return 0
  fi
  echo "complete out=$out" >>"$log"
}

# Missing native short/report generation.  Open/report generation receives a
# 1024-token context-safety ceiling; reaching it is diagnostic and never a validity
# failure.  Obvious repeated spans are rejected by structural qualification.
run_one llava \
  "$root/../../unified_eval/inputs/baseline_matrix_v1/visual_mimic_oe.json" \
  data/medheval/images "$root/native/llava/visual_mimic_oe/greedy" 490 1024 1

for dataset in cxr_vishal knowledge_mimic_ce slake_fine_grained vqa_rad_official_oe visual_mimic_oe; do
  case "$dataset" in
    cxr_vishal) images=data/medheval/images; expected=5587; tokens=128 ;;
    knowledge_mimic_ce) images=data/medheval/images; expected=2000; tokens=128 ;;
    slake_fine_grained) images=data/medheval/images/Slake; expected=1536; tokens=128 ;;
    vqa_rad_official_oe) images=/home/dbw/datasets/public/vqa_rad_hf/test_images; expected=200; tokens=1024 ;;
    visual_mimic_oe) images=data/medheval/images; expected=490; tokens=1024 ;;
  esac
  for condition in no_context rag; do
    run_one llava "$root/rag/bm25/$dataset/$condition.json" "$images" \
      "$root/shared_rag_generation/llava/$dataset/$condition" "$expected" "$tokens" 1
  done
done

for dataset in iu_xray_report mimic_cxr_report; do
  if [[ "$dataset" == iu_xray_report ]]; then expected=590; else expected=694; fi
  for condition in no_context rag; do
    run_one llava "$root/rag/biomedclip_report/$dataset/$condition.json" data/medheval/images \
      "$root/shared_rag_report_generation/llava/$dataset/$condition" "$expected" 1024 1
  done
done

flock -u 8

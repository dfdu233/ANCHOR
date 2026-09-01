#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/second-vqa-rad-oe.lock
if ! flock -n 9; then
  echo "Another SECOND VQA-RAD OE pipeline owns the run lock; experiment not started"
  # A lock collision means this invocation did not run the experiment.  Use a
  # distinct non-zero status so the detached-job ledger cannot call it done.
  exit 75
fi

upstream=corrected_runs/detached_jobs/hulu-report-dependency-audit-v4.json
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done

second_root=/home/dbw/third_party/SECOND
second_eval=$second_root/lmms-eval-vicuna
second_python=/home/dbw/envs/second/bin/python
task_path=/home/dbw/ANCHOR/configs/unified_eval/lmms_tasks/vqa_rad_oe
manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
smoke=corrected_runs/unified_eval/smoke/second_vqa_rad_oe_v1
full=corrected_runs/unified_eval/full/second_vqa_rad_oe_v1
mkdir -p "$smoke" "$full"

export PYTHONPATH=$second_eval:$second_eval/LLaVA-NeXT:/home/dbw/ANCHOR
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

run_second() {
  local output_root=$1
  shift
  cd "$second_eval"
  "$second_python" -m lmms_eval \
    --model llava \
    --model_args pretrained=/home/dbw/models/LLaVA-Med-v1.5-mistral-7b,conv_template=mistral_instruct,device_map=custom \
    --include_path "$task_path" \
    --tasks vqa_rad_official_test_oe \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix second_vqa_rad_oe \
    --output_path "$output_root/lmms" \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type attn_topk \
    --attention_threshold 1.0 \
    --positional_embedding_type bilinear_interpolation \
    --stages -2 -1 0 1 \
    --contrastive_alphas 0.8 0.8 0.8 \
    --seed 42 \
    --verbosity INFO \
    "$@"
  cd /home/dbw/ANCHOR
}

smoke_passed=false
if [[ -f "$smoke/qualification.json" ]]; then
  smoke_passed=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(str(bool(json.load(open(sys.argv[1])).get("passed",False))).lower())' \
    "$smoke/qualification.json")
fi
if [[ "$smoke_passed" != "true" ]]; then
  run_second "$smoke" --limit 32
  mapfile -t smoke_samples < <(find "$smoke/lmms" -type f -name '*_samples_vqa_rad_official_test_oe.jsonl' | sort)
  if [[ ${#smoke_samples[@]} -eq 0 ]]; then
    echo "SECOND smoke produced no per-sample log"
    exit 2
  fi
  smoke_sample=${smoke_samples[$((${#smoke_samples[@]} - 1))]}
  "$second_python" -m anchor.medeval.import_lmms_samples \
    --samples "$smoke_sample" \
    --output "$smoke/answers.jsonl"
  "$second_python" -m anchor.medeval.qualify_oe_generation \
    --manifest "$manifest" \
    --answers "$smoke/answers.jsonl" \
    --limit 32 \
    --output "$smoke/qualification.json"
fi

if [[ ! -f "$full/answers.jsonl" ]] || [[ $(wc -l < "$full/answers.jsonl") -ne 200 ]]; then
  run_second "$full"
  mapfile -t full_samples < <(find "$full/lmms" -type f -name '*_samples_vqa_rad_official_test_oe.jsonl' | sort)
  if [[ ${#full_samples[@]} -eq 0 ]]; then
    echo "SECOND full run produced no per-sample log"
    exit 2
  fi
  full_sample=${full_samples[$((${#full_samples[@]} - 1))]}
  "$second_python" -m anchor.medeval.import_lmms_samples \
    --samples "$full_sample" \
    --output "$full/answers.jsonl"
fi

mapfile -t greedy_answers < <(find corrected_runs/unified_eval/full/vqa_rad_oe_mitigation_v3/vqa_rad/official_test_oe/open_vqa/greedy -type f -name 'chunk_*.answers.jsonl' | sort)
if [[ ${#greedy_answers[@]} -eq 0 ]]; then
  echo "Greedy full answers unavailable; cannot perform paired evaluation"
  exit 2
fi
"$second_python" -m anchor.medeval.evaluate_oe_vqa \
  --manifest "$manifest" \
  --answers "$full/answers.jsonl" \
  --baseline-answers "${greedy_answers[@]}" \
  --output "$full/evaluation.json" \
  --bootstrap-replicates 5000 \
  --seed 42

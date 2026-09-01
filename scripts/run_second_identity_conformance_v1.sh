#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/second-identity-conformance.lock
if ! flock -n 9; then
  echo "Another SECOND identity audit owns the run lock; experiment not started"
  exit 75
fi

upstream=corrected_runs/detached_jobs/llava-port-diagnostic-v4.json
while true; do
  status=$(/opt/miniconda3/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status","missing"))' "$upstream")
  [[ "$status" == "done" || "$status" == "failed" ]] && break
  sleep 30
done

out=corrected_runs/unified_eval/sanity/second_identity_conformance_v1
manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
second_root=/home/dbw/third_party/SECOND
second_eval=$second_root/lmms-eval-vicuna
second_python=/home/dbw/envs/second/bin/python
task_path=/home/dbw/ANCHOR/configs/unified_eval/lmms_tasks/vqa_rad_oe
mkdir -p "$out"

export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

export PYTHONPATH=/home/dbw/ANCHOR/anchor:/home/dbw/ANCHOR/data/medheval/code/baselines/Med-LVLMs/llava-med-1.5
canonical_ready=false
if [[ -f "$out/canonical.answers.jsonl" && -f "$out/canonical.qualification.json" ]]; then
  canonical_ready=$(/opt/miniconda3/bin/python - "$out/canonical.answers.jsonl" "$out/canonical.qualification.json" <<'PY'
import json
import sys

answers = sum(1 for line in open(sys.argv[1]) if line.strip())
qualification = json.load(open(sys.argv[2]))
print(str(answers == 32 and qualification.get("passed") is True).lower())
PY
  )
fi
if [[ "$canonical_ready" != "true" ]]; then
  /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.diagnose_llava_port canonical \
    --manifest "$manifest" --image-root "$images" --output "$out/canonical.answers.jsonl" \
    --limit 32 --max-new-tokens 64 --seed 42
  /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_oe_generation \
    --manifest "$manifest" --answers "$out/canonical.answers.jsonl" --limit 32 \
    --output "$out/canonical.qualification.json"
else
  echo "Reusing qualified frozen canonical 32 answers"
fi

export PYTHONPATH=$second_eval:$second_eval/LLaVA-NeXT:/home/dbw/ANCHOR
cd "$second_eval"
"$second_python" -m lmms_eval \
  --model llava \
  --model_args pretrained=/home/dbw/models/LLaVA-Med-v1.5-mistral-7b,conv_template=mistral_instruct,device_map=custom \
  --include_path "$task_path" \
  --tasks vqa_rad_official_test_oe \
  --batch_size 1 \
  --limit 32 \
  --log_samples \
  --log_samples_suffix second_identity \
  --output_path /home/dbw/ANCHOR/$out/lmms \
  --seed 42 \
  --verbosity INFO
cd /home/dbw/ANCHOR

mapfile -t samples < <(find "$out/lmms" -type f -name '*_samples_vqa_rad_official_test_oe.jsonl' | sort)
if [[ ${#samples[@]} -eq 0 ]]; then
  echo "SECOND identity run produced no per-sample log"
  exit 2
fi
sample=${samples[$((${#samples[@]} - 1))]}
"$second_python" -m anchor.medeval.import_lmms_samples \
  --samples "$sample" --output "$out/second_standard.answers.jsonl"
"$second_python" -m anchor.medeval.qualify_oe_generation \
  --manifest "$manifest" --answers "$out/second_standard.answers.jsonl" --limit 32 \
  --output "$out/second_standard.qualification.json"
"$second_python" -m anchor.medeval.evaluate_backend_conformance \
  --canonical "$out/canonical.answers.jsonl" \
  --candidate "$out/second_standard.answers.jsonl" \
  --output "$out/conformance.json"

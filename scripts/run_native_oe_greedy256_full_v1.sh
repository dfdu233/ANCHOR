#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

jobs=corrected_runs/detached_jobs
upstream="$jobs/native-oe-controls-t2-v1.json"
while true; do
  status=$(/opt/miniconda3/bin/python - "$upstream" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
print(json.load(p.open()).get("status", "missing") if p.is_file() else "missing")
PY
)
  [[ "$status" == "done" ]] && break
  if [[ "$status" == "failed" ]]; then
    echo "T2 controls failed; full OE generation is not authorized" >&2
    exit 2
  fi
  sleep 30
done

mkdir -p "$jobs/locks"
exec 8>"$jobs/locks/gpu0-vindr-v2.lock"
flock 8

manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
registry=corrected_runs/unified_eval/provenance/artifact_registry_v1.jsonl

run_model() {
  local model=$1 python=$2
  local output=corrected_runs/unified_eval/full/${model}_native_vqa_rad_oe_greedy256_v1
  PYTHONPATH=anchor "$python" -m anchor.medeval.run_native_oe_vqa \
    --model "$model" --manifest "$manifest" --image-root "$images" \
    --output-dir "$output" --max-new-tokens 256 --seed 42 \
    --decode-mode greedy --num-beams 1
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.medeval.qualify_oe_generation \
    --manifest "$manifest" --answers "$output/answers.jsonl" --limit 200 \
    --max-new-tokens 256 --max-cap-hit-rate 0.05 \
    --require-terminal-completeness --terminal-question-policy explicit_sentence_instruction \
    --min-terminal-completeness-rate 0.95 --output "$output/qualification.json"
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.medeval.evaluate_oe_vqa \
    --manifest "$manifest" --answers "$output/answers.jsonl" \
    --output "$output/evaluation_lexical_auxiliary.json" \
    --bootstrap-replicates 5000 --seed 42
  PYTHONPATH=. /opt/miniconda3/bin/python -m anchor.medeval.artifact_registry \
    --registry "$registry" --artifact "$output/answers.jsonl" \
    --status admissible --evaluator-version oe-generation-qualification-v3-response-form \
    --evidence-scope "qualified raw OE generation; vqa-rad; $model; greedy256; clinical claim evaluation pending" \
    --reason 'identity, nonempty, diversity, cap-hit, and response-form gates passed; lexical metrics auxiliary' \
    --qualification "$output/qualification.json"
}

run_model hulu /home/dbw/.venvs/hulumed/bin/python
run_model llava /opt/miniconda3/envs/huatuo/bin/python
flock -u 8

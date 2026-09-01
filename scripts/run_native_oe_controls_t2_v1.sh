#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
registry=corrected_runs/unified_eval/provenance/artifact_registry_v1.jsonl
root=corrected_runs/unified_eval/smoke/native_oe_controls_t2_v1
mkdir -p "$root" corrected_runs/detached_jobs/locks

exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
flock 8

run_model() {
  local model=$1 python=$2 identity_manifest=$3 identity_images=$4 canonical=$5 identity_tokens=$6
  local identity="$root/$model/identity_trace_certified"
  local greedy="$root/$model/greedy256"
  local beam="$root/$model/beam4_256"

  PYTHONPATH=anchor "$python" -m anchor.medeval.run_native_oe_vqa \
    --model "$model" --manifest "$identity_manifest" --image-root "$identity_images" \
    --output-dir "$identity" --limit 32 --max-new-tokens "$identity_tokens" --seed 42 \
    --decode-mode greedy --num-beams 1
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.medeval.evaluate_backend_conformance \
    --canonical "$canonical" --candidate "$identity/answers.jsonl" \
    --min-normalized-exact 1 --min-token-f1 1 --require-token-exact \
    --output "$identity/conformance.json"

  PYTHONPATH=anchor "$python" -m anchor.medeval.run_native_oe_vqa \
    --model "$model" --manifest "$manifest" --image-root "$images" \
    --output-dir "$greedy" --limit 32 --max-new-tokens 256 --seed 42 \
    --decode-mode greedy --num-beams 1
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.medeval.qualify_oe_generation \
    --manifest "$manifest" --answers "$greedy/answers.jsonl" --limit 32 \
    --max-new-tokens 256 --max-cap-hit-rate 0.05 \
    --require-terminal-completeness --terminal-question-policy explicit_sentence_instruction \
    --min-terminal-completeness-rate 0.95 --output "$greedy/qualification.json"

  PYTHONPATH=anchor "$python" -m anchor.medeval.run_native_oe_vqa \
    --model "$model" --manifest "$manifest" --image-root "$images" \
    --output-dir "$beam" --limit 32 --max-new-tokens 256 --seed 42 \
    --decode-mode beam --num-beams 4
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.medeval.qualify_oe_generation \
    --manifest "$manifest" --answers "$beam/answers.jsonl" --limit 32 \
    --max-new-tokens 256 --max-cap-hit-rate 0.05 \
    --require-terminal-completeness --terminal-question-policy explicit_sentence_instruction \
    --min-terminal-completeness-rate 0.95 --output "$beam/qualification.json"

  for arm in greedy256 beam4_256; do
    PYTHONPATH=. /opt/miniconda3/bin/python -m anchor.medeval.artifact_registry \
      --registry "$registry" --artifact "$root/$model/$arm/answers.jsonl" \
      --status admissible --evaluator-version oe-generation-qualification-v3-response-form \
      --evidence-scope "canonical OE-VQA functional smoke; vqa-rad; $model; $arm; T2_n32" \
      --reason 'identity, nonempty, diversity, cap-hit, and response-form gates passed' \
      --qualification "$root/$model/$arm/qualification.json"
  done
}

run_model \
  hulu /home/dbw/.venvs/hulumed/bin/python \
  corrected_runs/unified_eval/rag/common_protocol_v1/mimic/visual_ce_v2/t3_n200_top3/prompts/no_context.json \
  /home/dbw/ANCHOR/data/medheval/images \
  corrected_runs/unified_eval/rag/common_protocol_v1/mimic/visual_ce_v2/ladder_v3/T2_n32/hulu/no_context/answers.jsonl \
  128
run_model \
  llava /opt/miniconda3/envs/huatuo/bin/python \
  "$manifest" "$images" \
  corrected_runs/unified_eval/sanity/llava_canonical_runtime_gate_v2/n32/canonical/answers.jsonl \
  64

flock -u 8

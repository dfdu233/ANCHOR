#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/common-rag-ce-ladder-v3.lock
if ! flock -n 9; then
  echo "Another common RAG CE v3 ladder owns the lock" >&2
  exit 75
fi

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
eval_python=/opt/miniconda3/envs/huatuo/bin/python
hulu_python=/home/dbw/.venvs/hulumed/bin/python
root=corrected_runs/unified_eval/rag/common_protocol_v1
max_tokens=128

run_generation() {
  local model=$1 manifest=$2 images=$3 out=$4 limit=$5
  # Native runners instantiate the full VLM before checking their append-only
  # prefix.  Avoid that cost when a prior arm is already complete, but only
  # after validating the exact contract and ordered qids.
  if "$eval_python" - "$model" "$manifest" "$images" "$out" "$limit" "$max_tokens" <<'PY'
import json, sys
from pathlib import Path

model, manifest_path, image_root, out_path, limit, max_tokens = sys.argv[1:]
limit, max_tokens = int(limit), int(max_tokens)
manifest_path = Path(manifest_path).resolve()
image_root = Path(image_root).resolve()
out = Path(out_path)
config_path, answers_path = out / "generation_config.json", out / "answers.jsonl"
if not config_path.is_file() or not answers_path.is_file():
    raise SystemExit(1)
config = json.load(config_path.open())
checks = (
    config.get("model") == model,
    Path(config.get("manifest", "")).resolve() == manifest_path,
    Path(config.get("image_root", "")).resolve() == image_root,
    config.get("limit") == limit,
    config.get("max_new_tokens") == max_tokens,
    config.get("seed") == 42,
)
if not all(checks):
    raise SystemExit(1)
manifest = json.load(manifest_path.open())
if isinstance(manifest, dict):
    manifest = manifest.get("questions", manifest.get("data", []))
def qid(row):
    for key in ("question_id", "qid", "id", "sample_id"):
        if key in row:
            return str(row[key])
    return "None"
expected = [qid(row) for row in manifest[:limit]]
answers = [json.loads(line) for line in answers_path.read_text().splitlines() if line.strip()]
observed = [qid(row) for row in answers]
tokens_ok = all(isinstance(row.get("metadata", {}).get("generated_token_ids"), list) for row in answers)
raise SystemExit(0 if observed == expected and len(observed) == limit and tokens_ok else 1)
PY
  then
    echo "Reusing complete prefix without model load: model=$model out=$out n=$limit"
    return 0
  fi
  if [[ "$model" == "huatuo" ]]; then
    PYTHONPATH=anchor:/home/dbw/HuatuoGPT-Vision "$eval_python" \
      -m anchor.medeval.run_huatuo_native_oe_vqa \
      --manifest "$manifest" --image-root "$images" --output-dir "$out" \
      --limit "$limit" --max-new-tokens "$max_tokens" --seed 42
  elif [[ "$model" == "hulu" ]]; then
    PYTHONPATH=anchor "$hulu_python" -m anchor.medeval.run_native_oe_vqa \
      --model hulu --manifest "$manifest" --image-root "$images" \
      --output-dir "$out" --limit "$limit" --max-new-tokens "$max_tokens" --seed 42
  else
    PYTHONPATH=anchor "$eval_python" -m anchor.medeval.run_native_oe_vqa \
      --model llava --manifest "$manifest" --image-root "$images" \
      --output-dir "$out" --limit "$limit" --max-new-tokens "$max_tokens" --seed 42
  fi
}

qualify_and_score() {
  local manifest=$1 out=$2 limit=$3
  PYTHONPATH=. "$eval_python" -m anchor.medeval.qualify_ce_generation \
    --manifest "$manifest" --answers "$out/answers.jsonl" --limit "$limit" \
    --max-new-tokens "$max_tokens" --output "$out/qualification.json" || return 1
  PYTHONPATH=anchor "$eval_python" -m corrected_sgta.evaluate_medheval_answers \
    --answers "$out/answers.jsonl" --questions "$manifest" \
    --output "$out/evaluation.json" || return 1
  "$eval_python" - "$out/evaluation.json" <<'PY' || return 1
import json, sys
report = json.load(open(sys.argv[1]))
if report["protocol_version"] != "medheval-decoded-eval-v6-leading-ceg-reference":
    raise SystemExit("wrong CE-G evaluator version")
if report["invalid_ground_truth"]:
    raise SystemExit("visual CE manifest contains an invalid binary reference")
PY
}

failed=0
for dataset in iuxray mimic; do
  if [[ "$dataset" == "iuxray" ]]; then
    images=/home/dbw/ANCHOR/data/medheval/images/IU-Xray
  else
    images=/home/dbw/ANCHOR/data/medheval/images
  fi
  prompts="$root/$dataset/visual_ce_v2/t3_n200_top3/prompts"
  for model in huatuo hulu llava; do
    smoke_ok=1
    for arm in no_context rag; do
      manifest="$prompts/$arm.json"
      out="$root/$dataset/visual_ce_v2/ladder_v3/T2_n32/$model/$arm"
      if ! run_generation "$model" "$manifest" "$images" "$out" 32 || \
         ! qualify_and_score "$manifest" "$out" 32; then
        echo "T2 failed: dataset=$dataset model=$model arm=$arm" >&2
        smoke_ok=0
        failed=1
      fi
    done
    [[ "$smoke_ok" == 1 ]] || continue
    pilot_ok=1
    for arm in no_context rag; do
      manifest="$prompts/$arm.json"
      out="$root/$dataset/visual_ce_v2/ladder_v3/T3_n200/$model/$arm"
      if ! run_generation "$model" "$manifest" "$images" "$out" 200 || \
         ! qualify_and_score "$manifest" "$out" 200; then
        echo "T3 failed: dataset=$dataset model=$model arm=$arm" >&2
        pilot_ok=0
        failed=1
      fi
    done
    [[ "$pilot_ok" == 1 ]] || continue
    PYTHONPATH=. "$eval_python" -m anchor.medeval.compare_ce_arms \
      --manifest "$prompts/no_context.json" \
      --baseline "$root/$dataset/visual_ce_v2/ladder_v3/T3_n200/$model/no_context/answers.jsonl" \
      --candidate "$root/$dataset/visual_ce_v2/ladder_v3/T3_n200/$model/rag/answers.jsonl" \
      --output "$root/$dataset/visual_ce_v2/ladder_v3/T3_n200/$model/comparison.json" \
      --bootstrap-draws 5000 --seed 42 || failed=1
  done
done

exit "$failed"

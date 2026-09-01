#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks

# One causal-control ladder at a time.  The second lock is shared with the
# VinDr collectors so a queued experiment cannot silently contend for GPU 0.
exec 9>corrected_runs/detached_jobs/locks/llava-common-rag-controls-v1.lock
if ! flock -n 9; then
  echo "Another LLaVA common-RAG causal-control ladder owns the lock" >&2
  exit 75
fi
exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
flock 8

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

eval_python=/opt/miniconda3/envs/huatuo/bin/python
root=corrected_runs/unified_eval/rag/common_protocol_v1
ladder_summary="$root/visual_ce_ladder_v3_summary.json"
max_tokens=128
failed=0

run_generation() {
  local manifest=$1 images=$2 out=$3 limit=$4
  mkdir -p "$out"

  # Reuse is fail-closed: configuration, ordered qids, and token provenance
  # must all match before avoiding a model load.
  if "$eval_python" - "$manifest" "$images" "$out" "$limit" "$max_tokens" <<'PY'
import json, sys
from pathlib import Path

manifest_path, image_root, out_path, limit, max_tokens = sys.argv[1:]
manifest_path = Path(manifest_path).resolve()
image_root = Path(image_root).resolve()
out = Path(out_path)
limit, max_tokens = int(limit), int(max_tokens)
config_path, answers_path = out / "generation_config.json", out / "answers.jsonl"
if not config_path.is_file() or not answers_path.is_file():
    raise SystemExit(1)
config = json.load(config_path.open())
checks = (
    config.get("model") == "llava",
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
    echo "Reusing complete control prefix: out=$out n=$limit"
    return 0
  fi

  PYTHONPATH=anchor "$eval_python" -m anchor.medeval.run_native_oe_vqa \
    --model llava --manifest "$manifest" --image-root "$images" \
    --output-dir "$out" --limit "$limit" \
    --max-new-tokens "$max_tokens" --seed 42
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
    raise SystemExit("visual CE control manifest contains an invalid binary reference")
PY
}

authorized_dataset() {
  local dataset=$1
  "$eval_python" - "$ladder_summary" "$dataset" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
dataset = sys.argv[2]
authorized = {
    (row.get("dataset"), row.get("model"))
    for row in summary.get("full_authorized", [])
}
raise SystemExit(0 if (dataset, "llava") in authorized else 1)
PY
}

comparison_authorized() {
  local path=$1
  "$eval_python" - "$path" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("full_run_authorized") is True else 1)
PY
}

for dataset in iuxray mimic; do
  if ! authorized_dataset "$dataset"; then
    echo "Skipping non-authorized raw RAG gain: dataset=$dataset model=llava"
    continue
  fi
  if [[ "$dataset" == "iuxray" ]]; then
    images=/home/dbw/ANCHOR/data/medheval/images/IU-Xray
  else
    images=/home/dbw/ANCHOR/data/medheval/images
  fi

  base="$root/$dataset/visual_ce_v2"
  prompts="$base/t3_n200_top3/prompts"
  controls="$base/t3_n200_top3/controls_v1"
  out_root="$base/ladder_v3/causal_controls_v1"
  real_answers="$base/ladder_v3/T3_n200/llava/rag/answers.jsonl"
  mkdir -p "$out_root"

  relevance_ok=1
  for tier in T2_n32 T3_n200; do
    if [[ "$tier" == "T2_n32" ]]; then
      limit=32
    else
      limit=200
    fi
    out="$out_root/$tier/llava/shuffled_context"
    if ! run_generation "$controls/shuffled_context.json" "$images" "$out" "$limit" || \
       ! qualify_and_score "$controls/shuffled_context.json" "$out" "$limit"; then
      echo "Shuffled-context control failed: dataset=$dataset tier=$tier" >&2
      relevance_ok=0
      failed=1
      break
    fi
  done
  [[ "$relevance_ok" == 1 ]] || continue

  PYTHONPATH=. "$eval_python" -m anchor.medeval.compare_ce_arms \
    --manifest "$prompts/rag.json" \
    --baseline "$out_root/T3_n200/llava/shuffled_context/answers.jsonl" \
    --candidate "$real_answers" \
    --output "$out_root/rag_vs_shuffled_context.json" \
    --bootstrap-draws 5000 --seed 42 || { failed=1; continue; }

  # Relevance is a prerequisite.  Do not spend GPU time on image swapping if
  # the retrieved content itself does not outperform a disjoint permutation.
  if ! comparison_authorized "$out_root/rag_vs_shuffled_context.json"; then
    echo "Relevant retrieval failed its causal gate; skipping image control: dataset=$dataset"
    continue
  fi

  image_ok=1
  for tier in T2_n32 T3_n200; do
    if [[ "$tier" == "T2_n32" ]]; then
      limit=32
    else
      limit=200
    fi
    out="$out_root/$tier/llava/image_swap"
    if ! run_generation "$controls/image_swap.json" "$images" "$out" "$limit" || \
       ! qualify_and_score "$controls/image_swap.json" "$out" "$limit"; then
      echo "Image-identity control failed: dataset=$dataset tier=$tier" >&2
      image_ok=0
      failed=1
      break
    fi
  done
  [[ "$image_ok" == 1 ]] || continue

  PYTHONPATH=. "$eval_python" -m anchor.medeval.compare_ce_arms \
    --manifest "$prompts/rag.json" \
    --baseline "$out_root/T3_n200/llava/image_swap/answers.jsonl" \
    --candidate "$real_answers" \
    --output "$out_root/rag_vs_image_swap.json" \
    --bootstrap-draws 5000 --seed 42 || failed=1
done

PYTHONPATH=. "$eval_python" -m anchor.medeval.summarize_rag_controls \
  --root "$root" --datasets iuxray mimic \
  --output "$root/visual_ce_ladder_v3_causal_controls.json" || failed=1

exit "$failed"

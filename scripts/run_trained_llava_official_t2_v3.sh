#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR

root=corrected_runs/paper_baselines_v1/trained_llava_t2_v3
log=corrected_runs/detached_jobs/logs/baseline-trained-official-t2-v3.log
lock=corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
python=/home/dbw/.venvs/llava15-official-431/bin/python
manifest=corrected_runs/unified_eval/inputs/baseline_matrix_v1/vqa_rad_official_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
base=/home/dbw/models/llava-v1.5-7b
ha_root=third_party/training_baselines/HA-DPO/ha_dpo/models/llava-v1_5
da_root=third_party/training_baselines/DA-DPO
opa_root=third_party/training_baselines/OPA-DPO
opa_llava="$opa_root/llava_setup/LLaVA"
sentinel_root=third_party/MedHEval/code/baselines/Med-LVLMs/llava_1.6/LLaVA
official_entry="$ha_root/llava/eval/model_vqa_loader.py"
variants=(base ha-dpo opa-dpo da-dpo sentinel less-is-more factmm-rag-generator)

mkdir -p "$root" "$(dirname "$log")" "$(dirname "$lock")"
PYTHONPATH=. "$python" -m anchor.medeval.prepare_official_llava_t2_manifest_v1 \
  --manifest "$manifest" --output "$root/official_questions_n1.jsonl" --limit 1 >>"$log" 2>&1
PYTHONPATH=. "$python" -m anchor.medeval.prepare_official_llava_t2_manifest_v1 \
  --manifest "$manifest" --output "$root/official_questions_n32.jsonl" --limit 32 >>"$log" 2>&1

# The lock is held for the entire GPU-bearing queue, not one subprocess at a
# time.  This prevents another generator from entering between unified and
# official runs and invalidating their conformance comparison.
exec 8>"$lock"
flock 8
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$python" - <<'PY' >>"$log" 2>&1
import inspect, json, pathlib, torch, transformers, tokenizers, peft
assert transformers.__version__ == "4.31.0", transformers.__version__
assert tokenizers.__version__ == "0.13.3", tokenizers.__version__
out = {
    "python": inspect.getfile(transformers),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "peft": peft.__version__,
}
path = pathlib.Path("corrected_runs/paper_baselines_v1/trained_llava_t2_v3/environment.json")
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY

checkpoint_path() {
  case "$1" in
    base) echo "$base" ;;
    ha-dpo) echo /home/dbw/models/hadpo-llava-1.5 ;;
    opa-dpo) echo /home/dbw/models/opadpo-lora-llava-v1.5-7b ;;
    da-dpo) echo /home/dbw/models/da-dpo-llava-v1.5-7b ;;
    sentinel) echo /home/dbw/models/sentinel-llava-v1.5-7b ;;
    less-is-more) echo /home/dbw/models/less-is-more-llava-v1.5-7b ;;
    factmm-rag-generator) echo /home/dbw/models/factmm-rag-generator-v1 ;;
    *) return 1 ;;
  esac
}

official_model_path() {
  case "$1" in
    ha-dpo|da-dpo|sentinel|less-is-more)
      echo "/home/dbw/model_cache/trained_llava_t2_v3_aliases/$1-llava-lora"
      ;;
    *) checkpoint_path "$1" ;;
  esac
}

archive_stage_variant() {
  local stage=$1 variant=$2 stamp archive side
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  archive="corrected_runs/paper_baselines_v1/full_matrix_v1/invalid_archives/trained_t2_v3_${stage}_${variant}_${stamp}"
  for side in unified official; do
    if [[ -e "$root/$stage/$side/$variant" ]]; then
      mkdir -p "$archive/$side"
      mv "$root/$stage/$side/$variant" "$archive/$side/$variant"
    fi
  done
}

run_unified() {
  local stage=$1 limit=$2 variant=$3
  PYTHONPATH=. "$python" -m anchor.corrected_sgta.run_trained_llava_baseline_v1 \
    --variant "$variant" --manifest "$manifest" --image-root "$images" \
    --output-dir "$root/$stage/unified/$variant" --limit "$limit" \
    --max-new-tokens 256 --seed 42 >>"$log" 2>&1
}

run_official() {
  local stage=$1 limit=$2 variant=$3 questions model_path model_base source_root entry_path
  questions="$root/official_questions_${stage}.jsonl"
  model_path=$(official_model_path "$variant")
  if [[ "$variant" == opa-dpo ]]; then
    PYTHONPATH=".:$opa_root:$opa_llava" "$python" -m anchor.medeval.run_official_opa_t2_v3 \
      --official-entry "$opa_root/eval_llava_rlhf_coco/model_vqa.py" \
      --model-path "$base" --adapter-path "$(checkpoint_path "$variant")" \
      --image-folder "$images" --question-file "$questions" \
      --answers-file "$root/$stage/official/$variant/answers.jsonl" \
      --evidence-file "$root/$stage/official/$variant/evidence.json" \
      --max-new-tokens 256 >>"$log" 2>&1
    return
  fi
  model_base=""
  source_root=$ha_root
  entry_path=$official_entry
  case "$variant" in
    ha-dpo|less-is-more) model_base=$base ;;
    da-dpo) model_base=$base; source_root=$da_root ;;
    sentinel)
      model_base=$base
      source_root=$sentinel_root
      entry_path="$sentinel_root/llava/eval/model_vqa_loader.py"
      ;;
  esac
  command=("$python" -m anchor.medeval.run_official_llava_t2_v3
    --variant "$variant" --official-entry "$entry_path"
    --model-path "$model_path" --image-folder "$images"
    --question-file "$questions"
    --answers-file "$root/$stage/official/$variant/answers.jsonl"
    --evidence-file "$root/$stage/official/$variant/evidence.json"
    --conv-mode llava_v1 --max-new-tokens 256)
  [[ -n "$model_base" ]] && command+=(--model-base "$model_base")
  PYTHONPATH=".:$source_root" "${command[@]}" >>"$log" 2>&1
}

audit_one() {
  local stage=$1 limit=$2 variant=$3 extra=()
  [[ "$stage" == n32 ]] && extra=(--prerequisite-audit "$root/n1_audit.json")
  PYTHONPATH=. /opt/miniconda3/bin/python -m anchor.medeval.audit_trained_llava_t2_v3 \
    --root "$root" --stage "$stage" --expected "$limit" --variants "$variant" \
    --output "$root/$stage/gates/$variant.json" "${extra[@]}" >>"$log" 2>&1
}

run_stage() {
  local stage=$1 limit=$2 variant failures=0
  for variant in "${variants[@]}"; do
    if [[ -f "$root/$stage/gates/$variant.json" ]] && \
       /opt/miniconda3/bin/python - "$root/$stage/gates/$variant.json" "$variant" <<'PY' >/dev/null 2>&1
import json,sys
x=json.load(open(sys.argv[1])); raise SystemExit(0 if sys.argv[2] in x.get('passed_variants',[]) else 1)
PY
    then
      continue
    fi
    archive_stage_variant "$stage" "$variant"
    if run_unified "$stage" "$limit" "$variant" && \
       run_official "$stage" "$limit" "$variant" && \
       audit_one "$stage" "$limit" "$variant"; then
      echo "[$(date -u +%FT%TZ)] $stage $variant PASS" >>"$log"
    else
      echo "[$(date -u +%FT%TZ)] $stage $variant FAIL" >>"$log"
      failures=$((failures + 1))
    fi
  done
  return "$failures"
}

run_stage n1 1
n1_rc=$?
PYTHONPATH=. /opt/miniconda3/bin/python -m anchor.medeval.audit_trained_llava_t2_v3 \
  --root "$root" --stage n1 --expected 1 --variants "${variants[@]}" \
  --output "$root/n1_audit.json" >>"$log" 2>&1
n1_audit_rc=$?
if [[ "$n1_rc" -ne 0 || "$n1_audit_rc" -ne 0 ]]; then
  echo "[$(date -u +%FT%TZ)] n1 incomplete; n32 not admitted" >>"$log"
  exit 1
fi

run_stage n32 32
n32_rc=$?
PYTHONPATH=. /opt/miniconda3/bin/python -m anchor.medeval.audit_trained_llava_t2_v3 \
  --root "$root" --stage n32 --expected 32 --variants "${variants[@]}" \
  --prerequisite-audit "$root/n1_audit.json" --output "$root/t2_audit.json" >>"$log" 2>&1
audit_rc=$?
[[ "$n32_rc" -eq 0 && "$audit_rc" -eq 0 ]]

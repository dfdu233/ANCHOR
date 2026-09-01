#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

jobs=corrected_runs/detached_jobs
root=corrected_runs/paper_baselines_v1/full_matrix_v1/rag/biomedclip_report
generation=corrected_runs/paper_baselines_v1/full_matrix_v1/shared_rag_report_generation
corpus=corrected_runs/paper_baselines_v1/full_matrix_v1/rag/combined_corpus/corpus.jsonl
inputs=corrected_runs/unified_eval/inputs/baseline_matrix_v1
model_root=/home/dbw/models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
weights="$model_root/open_clip_pytorch_model.bin"
state="$jobs/baseline-report-rag-long-queue-v1.state.jsonl"
mkdir -p "$root" "$generation" "$jobs/locks" "$jobs/logs"

# The official checkpoint is 783,705,670 bytes.  Never try to load a partial
# resumable wget file.
while [[ ! -f "$weights" || "$(stat -c %s "$weights" 2>/dev/null || echo 0)" -ne 783705670 ]]; do
  sleep 30
done

# Retrieval is deterministic CPU work and is intentionally parallel with the
# single-GPU generation queue. Hiding CUDA here prevents OpenCLIP from
# contending with the active VLM while preserving the identical checkpoint,
# preprocessing, embeddings, and ranking rule.
for spec in \
  "iu_xray_report|iuxray|data/medheval/images/IU-Xray|iuxray|data/medheval/images/IU-Xray|590" \
  "mimic_cxr_report|mimic|data/medheval/images|iuxray|data/medheval/images/IU-Xray|694"; do
  IFS='|' read -r dataset source image_root corpus_source corpus_image_root expected <<<"$spec"
  out="$root/$dataset"; log="$jobs/logs/baseline-report-rag-retrieve-${dataset}.log"
  mkdir -p "$out"
  rc=1
  for attempt in 1 2 3; do
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.retrieve_report_biomedclip_v1 \
      --queries "$inputs/$dataset.json" --corpus "$corpus" --dataset "$source" \
      --image-root "$image_root" --corpus-dataset "$corpus_source" \
      --corpus-image-root "$corpus_image_root" --model-root "$model_root" \
      --output "$out/retrieval.jsonl" --cache-dir "$root/feature_cache" \
      --top-k 3 --batch-size 32 >>"$log" 2>&1
    rc=$?; [[ "$rc" -eq 0 ]] && break
  done
  if [[ "$rc" -eq 0 ]]; then
    PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.prepare_shared_rag_prompts_v1 \
      --queries "$inputs/$dataset.json" --retrieval "$out/retrieval.jsonl" \
      --output "$out/rag.json" >>"$log" 2>&1
    rc=$?
  fi
  /opt/miniconda3/bin/python - "$state" "$dataset" "$rc" "$expected" "$log" <<'PY'
import datetime,json,sys
p,d,r,n,l=sys.argv[1:]
with open(p,'a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'retrieval','dataset':d,'status':'completed' if r=='0' else 'failed','returncode':int(r),'expected':int(n),'log':l})+'\n')
PY
done

runtime_for() { case "$1" in hulu) echo /home/dbw/.venvs/hulumed/bin/python;; qwen) echo /home/dbw/.venvs/qwen25vl-v2/bin/python;; *) echo /opt/miniconda3/envs/huatuo/bin/python;; esac; }

bash scripts/wait_for_all_baseline_gates_v1.sh rag
exec 8>"$jobs/locks/gpu0-vindr-v2.lock"
for model in huatuo hulu llava qwen; do
  python=$(runtime_for "$model")
  for spec in \
    "iu_xray_report|data/medheval/images/IU-Xray|590" \
    "mimic_cxr_report|data/medheval/images|694"; do
    IFS='|' read -r dataset image_root expected <<<"$spec"
    for condition in no_context rag; do
      manifest="$root/$dataset/$condition.json"
      out="$generation/$model/$dataset/$condition"
      log="$jobs/logs/baseline-report-rag-${model}-${dataset}-${condition}.log"
      mkdir -p "$out"
      if [[ ! -f "$manifest" ]]; then rc=66
      else
        rc=1
        flock 8
        for attempt in 1 2 3; do
          PYTHONPATH=. "$python" -m anchor.medeval.run_native_oe_vqa --model "$model" \
            --manifest "$manifest" --image-root "$image_root" --output-dir "$out" \
            --limit 0 --max-new-tokens 256 --seed 42 --decode-mode greedy --num-beams 1 >>"$log" 2>&1
          rc=$?; [[ "$rc" -eq 0 ]] && break
          tail -80 "$log" | grep -q 'refusing to resume an incompatible' && break
        done
        flock -u 8
      fi
      if [[ "$rc" -eq 0 ]]; then
        PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_oe_generation \
          --manifest "$manifest" --answers "$out/answers.jsonl" --limit "$expected" \
          --max-new-tokens 256 --max-cap-hit-rate 0.05 \
          --output "$out/qualification.json" >>"$log" 2>&1
        rc=$?
      fi
      if [[ "$rc" -eq 0 ]]; then
        PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.evaluate_oe_vqa \
          --manifest "$manifest" --answers "$out/answers.jsonl" \
          --output "$out/evaluation_lexical_auxiliary.json" --bootstrap-replicates 5000 --seed 42 --max-new-tokens 256 >>"$log" 2>&1
        rc=$?
      fi
      /opt/miniconda3/bin/python - "$state" "$model" "$dataset" "$condition" "$rc" "$expected" "$log" <<'PY'
import datetime,json,sys
p,m,d,c,r,n,l=sys.argv[1:]
with open(p,'a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'generation','model':m,'dataset':d,'condition':c,'status':'completed' if r=='0' else 'failed','returncode':int(r),'expected':int(n),'log':l})+'\n')
PY
    done
  done
done

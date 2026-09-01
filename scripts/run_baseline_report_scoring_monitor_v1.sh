#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
root=corrected_runs/paper_baselines_v1/full_matrix_v1
inputs=corrected_runs/unified_eval/inputs/baseline_matrix_v1
out="$root/report_scores"
weights=/home/dbw/model_cache/report_metrics/download_audit.json
manifest=/home/dbw/model_cache/report_metrics/metric_manifest.json
state=corrected_runs/detached_jobs/baseline-report-scoring-monitor-v1.state.jsonl
clinical_mode=${BASELINE_CLINICAL_MODE:-required}
mkdir -p "$out"

archive_stale() {
  local target=$1 stamp archive file
  stamp=$(date -u +%Y%m%dT%H%M%SZ); archive="$target/stale_artifacts/$stamp"; mkdir -p "$archive"
  for file in "$target/summary.json" "$target/qualification.json"; do
    [[ -f "$file" ]] && mv "$file" "$archive/"
  done
}

while [[ ! -f "$weights" ]] || ! /opt/miniconda3/bin/python - "$weights" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('passed') else 1)
PY
do sleep 60; done

score_cell() {
  local key=$1 model=$2 method=$3 dataset=$4 expected=$5; shift 5
  local answers=("$@") lines=0 f pair target rc combined
  for f in "${answers[@]}"; do [[ -f "$f" ]] || return 0; lines=$((lines+$(wc -l < "$f"))); done
  [[ "$lines" -eq "$expected" ]] || return 0
  target="$out/$key"; pair="$target/pairs.jsonl"; combined="$target/answers.jsonl"
  if [[ -f "$target/summary.json" ]] && PYTHONPATH=. /opt/miniconda3/bin/python \
    -m anchor.medeval.validate_score_artifact_binding_v1 \
    --score "$target/summary.json" --qualification "$target/qualification.json" \
    --task report_generation --expected "$expected" >/dev/null 2>&1; then return 0; fi
  [[ -f "$target/summary.json" ]] && archive_stale "$target"
  mkdir -p "$target"
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.combine_answer_chunks_v1 \
    --manifest "$inputs/${dataset}_report.json" --answers "${answers[@]}" --output "$combined" \
    >"$target/prepare.log" 2>&1
  rc=$?
  if [[ "$rc" -eq 0 ]]; then
    PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_oe_generation \
      --manifest "$inputs/${dataset}_report.json" --answers "$combined" --limit "$expected" \
      --max-new-tokens 256 --max-cap-hit-rate 0.05 --output "$target/qualification.json" \
      >>"$target/prepare.log" 2>&1
    rc=$?
  fi
  if [[ "$rc" -eq 0 ]]; then
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.prepare_report_evaluation_pairs_v1 \
    --manifest "$inputs/${dataset}_report.json" --answers "$combined" --output "$pair" \
    --dataset "$dataset_source" --method "$method" --model "$model" >>"$target/prepare.log" 2>&1
    rc=$?
  fi
  if [[ "$rc" -eq 0 ]]; then
    PYTHONPATH=anchor .venv-full/bin/python -m corrected_sgta.evaluate_oe_reports \
      --input "$pair" --output-dir "$target" --clinical "$clinical_mode" \
      --clinical-python .venv-full/bin/python --metric-manifest "$manifest" \
      --clinical-cache /home/dbw/model_cache/report_metrics --validate-directions \
      >>"$target/score.log" 2>&1
    rc=$?
  fi
  /opt/miniconda3/bin/python - "$state" "$key" "$rc" "$lines" <<'PY'
import datetime,json,sys
p,k,r,n=sys.argv[1:]
with open(p,'a') as f:f.write(json.dumps({'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cell':k,'status':'completed' if r=='0' else 'failed','returncode':int(r),'rows':int(n)})+'\n')
PY
}

while true; do
  for dataset in iu_xray mimic_cxr; do
    if [[ "$dataset" == iu_xray ]]; then expected=590; dataset_source=iuxray; else expected=694; dataset_source=mimic; fi
    for model in huatuo hulu llava qwen; do
      for method in greedy beam; do
        score_cell "native/$model/$dataset/$method" "$model" "$method" "$dataset" "$expected" "$root/native/$model/${dataset}_report/$method/answers.jsonl"
      done
      for condition in no_context rag; do
        score_cell "shared_rag/$model/$dataset/$condition" "$model" "shared_rag_$condition" "$dataset" "$expected" "$root/shared_rag_report_generation/$model/${dataset}_report/$condition/answers.jsonl"
      done
    done
    for model in huatuo hulu qwen; do
      for method in vcd dola; do
        score_cell "cross/$model/$dataset/$method" "$model" "$method" "$dataset" "$expected" "$root/cross_model_methods/$model/$method/${dataset}_report/answers.jsonl"
      done
    done
    for method in VCD DoLa opera PAI avisc VISTA; do
      folder="$root/llava_methods/mmedrag/${dataset}_report/report_generation/$method"
      mapfile -t chunks < <(find "$folder" -maxdepth 1 -name 'chunk_*.answers.jsonl' -type f 2>/dev/null | sort)
      [[ "${#chunks[@]}" -gt 0 ]] && score_cell "llava_methods/$dataset/$method" llava "$method" "$dataset" "$expected" "${chunks[@]}"
    done
    for variant in base ha-dpo opa-dpo da-dpo sentinel less-is-more factmm-rag-generator vhr; do
      score_cell "trained/$variant/$dataset" llava15 "$variant" "$dataset" "$expected" "$root/trained_llava15/$variant/${dataset}_report/answers.jsonl"
    done
  done
  sleep 300
done

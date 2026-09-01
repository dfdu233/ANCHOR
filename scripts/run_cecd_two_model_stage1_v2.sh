#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks corrected_runs/vindr_v2/cecd_two_model_stage1_v2

admission=${CECD_ADMISSION_RESULT:-corrected_runs/vindr_v2/cecd_human_admission_v2/analysis.json}
PYTHONPATH=anchor .venv-full/bin/python -m corrected_sgta.cecd_admission_gate "$admission"

exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
flock 8

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

huatuo_python=/opt/miniconda3/envs/huatuo/bin/python
hulu_python=/home/dbw/.venvs/hulumed/bin/python

run_factorial() {
  local family=$1
  local python_bin=$2
  local output_dir=$3
  shift 3
  local resume_args=()
  if [[ -f "$output_dir/config.json" ]]; then
    resume_args=(--resume)
  fi
  PYTHONPATH=anchor:/home/dbw/HuatuoGPT-Vision "$python_bin" -u \
    -m corrected_sgta.run_cecd_factorial_v1 \
    --output-dir "$output_dir" \
    --model-family "$family" \
    --admission-result "$admission" \
    "${resume_args[@]}" \
    "$@"
}

huatuo_canary=corrected_runs/vindr_v2/cecd_huatuo_canary_admitted_v2
huatuo_full=corrected_runs/vindr_v2/cecd_huatuo_factorial_admitted_v2
hulu_canary=corrected_runs/vindr_v2/cecd_hulu_canary_admitted_v2
hulu_full=corrected_runs/vindr_v2/cecd_hulu_factorial_admitted_v2

run_factorial huatuo "$huatuo_python" "$huatuo_canary" --max-claims 1
run_factorial huatuo "$huatuo_python" "$huatuo_full"
run_factorial hulu "$hulu_python" "$hulu_canary" --max-claims 1
run_factorial hulu "$hulu_python" "$hulu_full"

PYTHONPATH=anchor .venv-full/bin/python -u \
  -m corrected_sgta.verify_cecd_two_model_stage1_v2 \
  --admission "$admission" \
  --huatuo-dir "$huatuo_full" \
  --hulu-dir "$hulu_full" \
  --output corrected_runs/vindr_v2/cecd_two_model_stage1_v2/input_gate.json

PYTHONPATH=anchor .venv-full/bin/python -u \
  anchor/corrected_sgta/analyze_clinical_equivalence_composition_defect_v1.py \
  --input "$huatuo_full/factorial_rows.jsonl" \
  --input "$hulu_full/factorial_rows.jsonl" \
  --output corrected_runs/vindr_v2/cecd_two_model_stage1_v2/analysis.json \
  --folds 5 \
  --bootstrap-draws 5000 \
  --seed 42

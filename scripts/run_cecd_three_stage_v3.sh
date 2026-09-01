#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks corrected_runs/vindr_v2/cecd_three_stage_v3

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
analysis_root=corrected_runs/vindr_v2/cecd_three_stage_v3

run_factorial() {
  local family=$1
  local python_bin=$2
  local stage=$3
  local split=$4
  local per_bin=$5
  local output_dir=corrected_runs/vindr_v2/cecd_${family}_${stage}_v3
  local resume_args=()
  if [[ -f "$output_dir/config.json" ]]; then
    resume_args=(--resume)
  fi
  PYTHONPATH=anchor:/home/dbw/HuatuoGPT-Vision "$python_bin" -u \
    -m corrected_sgta.run_cecd_factorial_v1 \
    --output-dir "$output_dir" \
    --model-family "$family" \
    --admission-result "$admission" \
    --stage-label "$stage" \
    --manifest-split "$split" \
    --per-bin "$per_bin" \
    "${resume_args[@]}"
}

run_stage_both_models() {
  local stage=$1
  local split=$2
  local per_bin=$3
  run_factorial huatuo "$huatuo_python" "$stage" "$split" "$per_bin"
  run_factorial hulu "$hulu_python" "$stage" "$split" "$per_bin"
}

# 160/model: operational canary only.  Its output cannot stop or authorize the
# mechanism on a scientific metric.
run_stage_both_models pilot_screen pilot 10
PYTHONPATH=anchor .venv-full/bin/python -u \
  anchor/corrected_sgta/analyze_clinical_equivalence_composition_defect_v1.py \
  --mode pilot_screen \
  --input corrected_runs/vindr_v2/cecd_huatuo_pilot_screen_v3/factorial_rows.jsonl \
  --input corrected_runs/vindr_v2/cecd_hulu_pilot_screen_v3/factorial_rows.jsonl \
  --output "$analysis_root/pilot_screen.json" --seed 42

# Independent dev split: fit every transform and coefficient exactly once.
run_stage_both_models dev_fit dev 20
PYTHONPATH=anchor .venv-full/bin/python -u \
  anchor/corrected_sgta/analyze_clinical_equivalence_composition_defect_v1.py \
  --mode dev_fit \
  --input corrected_runs/vindr_v2/cecd_huatuo_dev_fit_v3/factorial_rows.jsonl \
  --input corrected_runs/vindr_v2/cecd_hulu_dev_fit_v3/factorial_rows.jsonl \
  --output "$analysis_root/dev_fit.json" \
  --folds 5 --bootstrap-draws 5000 --seed 42

# Locked confirmation always uses the already serialized dev predictor.  The
# analyzer has no confirmation refit path.
run_stage_both_models confirmation_locked confirmation 60
PYTHONPATH=anchor .venv-full/bin/python -u \
  anchor/corrected_sgta/analyze_clinical_equivalence_composition_defect_v1.py \
  --mode confirmation_locked \
  --frozen-dev-fit "$analysis_root/dev_fit.json" \
  --input corrected_runs/vindr_v2/cecd_huatuo_confirmation_locked_v3/factorial_rows.jsonl \
  --input corrected_runs/vindr_v2/cecd_hulu_confirmation_locked_v3/factorial_rows.jsonl \
  --output "$analysis_root/confirmation_locked.json" \
  --bootstrap-draws 5000 --seed 42

PYTHONPATH=anchor .venv-full/bin/python -u \
  -m corrected_sgta.verify_cecd_three_stage_v3 \
  --admission "$admission" \
  --huatuo-pilot-screen-dir corrected_runs/vindr_v2/cecd_huatuo_pilot_screen_v3 \
  --huatuo-dev-fit-dir corrected_runs/vindr_v2/cecd_huatuo_dev_fit_v3 \
  --huatuo-confirmation-locked-dir corrected_runs/vindr_v2/cecd_huatuo_confirmation_locked_v3 \
  --hulu-pilot-screen-dir corrected_runs/vindr_v2/cecd_hulu_pilot_screen_v3 \
  --hulu-dev-fit-dir corrected_runs/vindr_v2/cecd_hulu_dev_fit_v3 \
  --hulu-confirmation-locked-dir corrected_runs/vindr_v2/cecd_hulu_confirmation_locked_v3 \
  --dev-fit "$analysis_root/dev_fit.json" \
  --confirmation "$analysis_root/confirmation_locked.json" \
  --output "$analysis_root/input_gate.json"


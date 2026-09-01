#!/usr/bin/env bash
set -uo pipefail
cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

jobs=corrected_runs/detached_jobs
lock="$jobs/locks/gpu0-vindr-v2.lock"
out=corrected_runs/dg_paired_validation_v1/huatuo_visual_mimic_n64
manifest=corrected_runs/unified_eval/inputs/baseline_matrix_v1/visual_mimic_oe.json
baseline=corrected_runs/paper_baselines_v1/full_matrix_v1/native/huatuo/visual_mimic_oe/greedy/answers.jsonl
bank=/home/dbw/data/modality_centers/PubMedVision/train/ct__chest.npy
mkdir -p "$jobs/logs" "$out"

# Baseline owns the GPU.  This process intentionally blocks here and is safe to
# leave detached while the matrix continues.
exec 8>"$lock"
flock 8
/opt/miniconda3/envs/huatuo/bin/python -m anchor.corrected_sgta.run_dg_paired_validation_v1 \
  --manifest "$manifest" --baseline-answers "$baseline" \
  --image-root data/medheval/images --source-bank "$bank" \
  --output-dir "$out" --variant feddg --alpha 0.01 --source-ratio 0.8 \
  --limit 64 --seed 42 --max-new-tokens 256 \
  >"$jobs/logs/dg-paired-validation-visual-mimic-n64.log" 2>&1
rc=$?
if [[ "$rc" -eq 0 ]]; then
  /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.qualify_oe_generation \
    --manifest "$manifest" --answers "$out/answers.jsonl" --limit 64 \
    --max-new-tokens 256 --max-cap-hit-rate 0.05 \
    --output "$out/qualification.json" \
    >>"$jobs/logs/dg-paired-validation-visual-mimic-n64.log" 2>&1
  qrc=$?
  [[ "$qrc" -ne 0 ]] && rc=$qrc
  /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.evaluate_oe_vqa \
    --manifest "$manifest" --answers "$out/answers.jsonl" \
    --baseline-answers "$baseline" \
    --limit 64 --output "$out/evaluation_lexical_auxiliary.json" \
    --bootstrap-replicates 5000 --seed 42 --max-new-tokens 256 \
    >>"$jobs/logs/dg-paired-validation-visual-mimic-n64.log" 2>&1
  erc=$?
  [[ "$erc" -ne 0 ]] && rc=$erc
  /opt/miniconda3/envs/huatuo/bin/python -m anchor.corrected_sgta.visualize_dg_alignment_v1 \
    --input "$out/answers.jsonl" --output-dir "$out/dg_visual_audit" \
    --image-root data/medheval/images --max-records 64 \
    >>"$jobs/logs/dg-paired-validation-visual-mimic-n64.log" 2>&1
fi
flock -u 8
echo "RC:$rc" >"$jobs/logs/dg-paired-validation-visual-mimic-n64.rc"
exit "$rc"

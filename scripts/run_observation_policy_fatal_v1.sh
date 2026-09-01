#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR

run_dir=corrected_runs/daylong_idea_search_v1/observation_policy_huatuo_v1
log_dir=corrected_runs/daylong_idea_search_v1/logs
mkdir -p "$log_dir"

resume_baselines() {
  bash scripts/resume_baselines_after_daylong_idea_v1.sh \
    >>"$log_dir/observation_policy_baseline_resume.log" 2>&1 || true
}
trap resume_baselines EXIT

CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/home/dbw/.cache/huggingface \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=. \
/opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.corrected_sgta.run_claim_universe_scoring \
  --model huatuo \
  --questions "$run_dir/manifest.json" \
  --image-root "$run_dir/images" \
  --output-dir "$run_dir/scores" \
  --skip-null \
  >"$log_dir/observation_policy_huatuo_score.log" 2>&1
score_status=$?

if [[ $score_status -ne 0 ]]; then
  exit "$score_status"
fi

PYTHONPATH=anchor \
/opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.observation_policy_probe_v1 analyze \
  --selections "$run_dir/selections.jsonl" \
  --raw "$run_dir/scores/raw.jsonl" \
  --output "$run_dir/analysis.json" \
  --model huatuo \
  --bootstrap-draws 5000 \
  --seed 20260812 \
  >"$log_dir/observation_policy_huatuo_analysis.log" 2>&1

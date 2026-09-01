#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
base=corrected_runs/unified_eval/physician_review/vqa_rad_mitigation_t3_n120_v1
inbox=/home/dbw/datasets/public/vqa_rad_hf/physician_review_returns/vqa_rad_mitigation_t3_n120_v1
output=$base/clinical_returns_v1

.venv-full/bin/python scripts/start_detached_job.py \
  --name llava-mitigation-t3-physician-monitor-v1 \
  --log corrected_runs/detached_jobs/llava-mitigation-t3-physician-monitor-v1.log \
  --state corrected_runs/detached_jobs/llava-mitigation-t3-physician-monitor-v1.json \
  -- .venv-full/bin/python scripts/monitor_physician_oe_pipeline.py \
    --base "$base" \
    --inbox "$inbox" \
    --output "$output" \
    --heartbeat "$output/monitor.heartbeat.json" \
    --interval 30

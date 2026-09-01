#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
for model in huatuo hulu; do
  base=corrected_runs/unified_eval/physician_review/internal_controls_t3_v2/$model
  inbox=/home/dbw/datasets/public/vqa_rad_hf/physician_review_returns/internal_controls_t3_v2/$model
  output=$base/clinical_returns_v1
  .venv-full/bin/python scripts/start_detached_job.py \
    --name "internal-controls-t3-v2-${model}-physician-monitor-v1" \
    --log "corrected_runs/detached_jobs/internal-controls-t3-v2-${model}-physician-monitor-v1.log" \
    --state "corrected_runs/detached_jobs/internal-controls-t3-v2-${model}-physician-monitor-v1.json" \
    -- .venv-full/bin/python scripts/monitor_physician_oe_pipeline.py \
      --base "$base" \
      --inbox "$inbox" \
      --output "$output" \
      --heartbeat "$output/monitor.heartbeat.json" \
      --interval 30
done

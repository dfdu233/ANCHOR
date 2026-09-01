#!/usr/bin/env bash
set -euo pipefail
cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
jobs=corrected_runs/detached_jobs
out=corrected_runs/factorial_interaction_probe_v1/pilot_n16/result.json
log="$jobs/logs/factorial-interaction-probe-v1.log"
mkdir -p "$(dirname "$out")" "$(dirname "$log")" "$jobs/locks"
if [[ -f "$out" ]]; then
  exit 0
fi
exec 8>"$jobs/locks/gpu0-vindr-v2.lock"
flock 8
PYTHONPATH=.:anchor /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.corrected_sgta.run_huatuo_factorial_interaction_probe_v1 \
  --manifest /home/dbw/datasets/physionet/vindr-cxr/1.0.0/manifests_v2/reader_vote_manifest_v2.jsonl \
  --bboxes /home/dbw/datasets/physionet/vindr-cxr/1.0.0/manifests_v2/bbox_annotations_v2.jsonl \
  --image-root /workspace/vinbigdata/train \
  --output "$out" --split pilot --per-bin 4 --max-cases 16 --seed 42 \
  >"$log" 2>&1
flock -u 8
PYTHONPATH=. /home/dbw/.venvs/hulumed/bin/python \
  -m anchor.corrected_sgta.analyze_factorial_interaction_probe_v1 \
  --input "$out" \
  --output corrected_runs/factorial_interaction_probe_v1/pilot_n16/analysis.json \
  >>"$log" 2>&1

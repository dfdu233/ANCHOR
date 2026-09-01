#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR

runtime=/home/dbw/.venvs/qwen25vl-v2/bin/python
manifest=corrected_runs/metric_calibration_probe_v2/prompt_manifest.jsonl
common=(
  --manifest "$manifest"
  --max-images 8
  --question-contract structured_neutral_v2 clinical_direct_v2
  --arm oracle_coordinate vision_coordinate
  --condition certified_x0p5 certified_x1 certified_x2 certified_cm missing detector_only header_unknown
)

qwen_output=corrected_runs/metric_calibration_probe_v2/qwen_parent_pilot_n8_v1
PYTHONPATH=. "$runtime" anchor/corrected_sgta/run_qwen25vl_metric_calibration_v1.py \
  --model /home/dbw/models/Qwen2.5-VL-7B-Instruct \
  --output "$qwen_output" "${common[@]}"
PYTHONPATH=. .venv-full/bin/python anchor/corrected_sgta/analyze_metric_calibration_v2.py \
  --answers "$qwen_output/answers.jsonl" --output "$qwen_output/analysis.json"

huatuo_output=corrected_runs/metric_calibration_probe_v2/huatuo_medical_pilot_n8_v1
PYTHONPATH=. "$runtime" anchor/corrected_sgta/run_qwen25vl_metric_calibration_v1.py \
  --model /home/dbw/models/HuatuoGPT-Vision-7B-Qwen2.5VL \
  --output "$huatuo_output" "${common[@]}"
PYTHONPATH=. .venv-full/bin/python anchor/corrected_sgta/analyze_metric_calibration_v2.py \
  --answers "$huatuo_output/answers.jsonl" --output "$huatuo_output/analysis.json"

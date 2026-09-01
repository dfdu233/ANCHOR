#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR

runtime=/home/dbw/.venvs/qwen25vl-v2/bin/python
analysis_runtime=.venv-full/bin/python
manifest=corrected_runs/metric_calibration_probe_v2/prompt_manifest.jsonl
root=corrected_runs/metric_calibration_probe_v2/full_v2
runner=anchor/corrected_sgta/run_qwen25vl_metric_calibration_v1.py
analyzer=anchor/corrected_sgta/analyze_metric_calibration_v2.py

run_and_analyze() {
  local model=$1
  local output=$2
  shift 2
  PYTHONPATH=. "$runtime" "$runner" \
    --model "$model" --manifest "$manifest" --output "$output" "$@"
  PYTHONPATH=. "$analysis_runtime" "$analyzer" \
    --answers "$output/answers.jsonl" --output "$output/analysis.json"
}

structured=(
  --max-images 97
  --question-contract structured_neutral_v2
  --arm oracle_coordinate vision_coordinate
  --condition certified_x0p5 certified_x1 certified_x2 certified_cm missing detector_only header_unknown
  --max-new-tokens 160
)

direct=(
  --max-images 97
  --question-contract clinical_direct_v2
  --arm vision_coordinate
  --condition certified_x1 missing detector_only header_unknown
  --max-new-tokens 320
)

run_and_analyze \
  /home/dbw/models/HuatuoGPT-Vision-7B-Qwen2.5VL \
  "$root/huatuo_medical_structured_n97_v1" "${structured[@]}"
run_and_analyze \
  /home/dbw/models/Qwen2.5-VL-7B-Instruct \
  "$root/qwen_parent_structured_n97_v1" "${structured[@]}"
run_and_analyze \
  /home/dbw/models/HuatuoGPT-Vision-7B-Qwen2.5VL \
  "$root/huatuo_medical_direct_n97_v1" "${direct[@]}"
run_and_analyze \
  /home/dbw/models/Qwen2.5-VL-7B-Instruct \
  "$root/qwen_parent_direct_n97_v1" "${direct[@]}"

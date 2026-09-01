#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${ANCHOR_PYTHON:-"$ROOT/.venv-full/bin/python"}
GPU=${GPU:-0}
OUT=${OUT:-"$ROOT/corrected_runs/high_efficiency"}

cd "$ROOT"

"$PY" -m corrected_sgta.audit_experiment_matrix --out "$OUT"
"$PY" -m corrected_sgta.prepare_generation_inputs --out "$OUT/inputs" --tasks open_vqa report_generation --sources mmedrag medheval chexpert_report

"$PY" -m corrected_sgta.run_llava_med_generation_matrix \
  --question-file "$OUT/inputs/mmedrag.mimic.report_generation.json" \
  --image-folder "$ROOT/data/medheval/images" \
  --out "$OUT/full_generation_mmedrag_mimic_report_20260726" \
  --source mmedrag --dataset mimic --task report_generation \
  --methods greedy beam DoLa PAI opera m3id VCD \
  --chunk-size 1000 --max-new-tokens 256 --conv-mode vicuna_v1 --gpu "$GPU" --continue-on-error

"$PY" -m corrected_sgta.summarize_high_efficiency --root "$OUT"

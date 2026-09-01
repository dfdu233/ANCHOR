#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

vindr=/home/dbw/datasets/physionet/vindr-cxr/1.0.0
manifest="$vindr/manifests/reader_vote_manifest.jsonl"
adjusted="$vindr/manifests/reader_adjusted_support/reader_adjusted_manifest.jsonl"
audit="$vindr/manifests/dicom_download_audit.json"
images="$vindr/train"
root=corrected_runs/vindr_layer_boundary/formal_v1
eval_python=/opt/miniconda3/envs/huatuo/bin/python
hulu_python=/home/dbw/.venvs/hulumed/bin/python

mkdir -p "$root" corrected_runs/detached_jobs/locks

# The data download and RAG pilot are independent persistent jobs.  Formal GPU
# work starts only after the complete DICOM audit and after the shared GPU is
# free; waiting is expected and survives a VS Code/SSH disconnect in tmux.
while true; do
  if [[ -s "$audit" ]] && "$eval_python" - "$audit" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
raise SystemExit(
    0
    if report.get("protocol_version") == "vindr-selective-dicom-audit-v2"
    and report.get("passed") is True
    else 1
)
PY
  then
    break
  fi
  sleep 60
done

for required in "$manifest" "$adjusted"; do
  [[ -s "$required" ]] || { echo "missing formal prerequisite: $required" >&2; exit 4; }
done

while "$eval_python" - <<'PY'
import json
from pathlib import Path
p = Path("corrected_runs/detached_jobs/common-rag-ce-ladder-v3.json")
if not p.exists():
    raise SystemExit(1)
raise SystemExit(0 if json.load(p.open()).get("status") in {"starting", "running"} else 1)
PY
do
  sleep 60
done

exec 9>corrected_runs/detached_jobs/locks/gpu0-formal-vindr.lock
flock 9

run_probe() {
  local model=$1 module=$2 python_bin=$3
  local model_root="$root/$model"
  local null="$model_root/global_null.npy"
  mkdir -p "$model_root"

  if [[ ! -s "$null" || ! -s "${null%.npy}.json" ]]; then
    local resume_args=()
    [[ -d "$model_root/null_calibration" ]] && resume_args+=(--resume)
    PYTHONPATH=anchor:/home/dbw/HuatuoGPT-Vision "$python_bin" -m "$module" \
      --manifest "$manifest" --image-root "$images" \
      --output-dir "$model_root/null_calibration" \
      --experiment-split dev --calibrate-global-null-output "$null" \
      "${resume_args[@]}"
  fi

  for split in dev test; do
    local out="$model_root/$split"
    local resume_args=()
    [[ -d "$out" ]] && resume_args+=(--resume)
    PYTHONPATH=anchor:/home/dbw/HuatuoGPT-Vision "$python_bin" -m "$module" \
      --manifest "$manifest" --image-root "$images" --output-dir "$out" \
      --experiment-split "$split" --global-null-npy "$null" \
      --bootstrap-draws 5000 "${resume_args[@]}"
  done

  PYTHONPATH=anchor "$eval_python" -m corrected_sgta.fit_reader_agreement_gate \
    --manifest "$manifest" --reader-adjusted-manifest "$adjusted" \
    --raw "$model_root/dev/raw.jsonl" "$model_root/test/raw.jsonl" \
    --model-id "$model" --steps 1000 --bootstrap-draws 5000 \
    --min-test-per-class 10 \
    --output "$model_root/reader_gate.json" \
    --boundary-output "$model_root/boundary_records.jsonl"
}

run_probe huatuo corrected_sgta.run_huatuo_vindr_commitment_probe "$eval_python"
run_probe hulu corrected_sgta.run_hulu_vindr_commitment_probe "$hulu_python"

cat "$root/huatuo/boundary_records.jsonl" "$root/hulu/boundary_records.jsonl" \
  > "$root/boundary_records.two_models.jsonl"
PYTHONPATH=. "$eval_python" -m anchor.medeval.classify_layer_boundary \
  --input "$root/boundary_records.two_models.jsonl" \
  --prereg configs/unified_eval/vindr_layer_boundary_prereg_v1.json \
  --output "$root/boundary_classification.json"

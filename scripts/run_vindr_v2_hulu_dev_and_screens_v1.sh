#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
eval_python=/opt/miniconda3/envs/huatuo/bin/python
hulu_python=/home/dbw/.venvs/hulumed/bin/python
jobs=corrected_runs/detached_jobs
manifest=/home/dbw/datasets/physionet/vindr-cxr/1.0.0/manifests_v2/reader_vote_manifest_v2.jsonl
sampling=/home/dbw/datasets/physionet/vindr-cxr/1.0.0/manifests_v2/summary_v2.json
audit=/home/dbw/datasets/physionet/vindr-cxr/1.0.0/manifests_v2/external_mount_audit_v1.json
images=/workspace/vinbigdata/train
findings=(aortic_enlargement cardiomegaly lung_opacity nodule_mass other_lesion pleural_effusion pleural_thickening pulmonary_fibrosis)

wait_for_success() {
  local state=$1
  while true; do
    "$eval_python" - "$state" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(3)
status = json.load(p.open()).get("status")
if status == "done":
    raise SystemExit(0)
if status == "failed":
    raise SystemExit(2)
raise SystemExit(3)
PY
    status=$?
    [[ "$status" == 0 ]] && return 0
    if [[ "$status" == 2 ]]; then
      echo "Required predecessor failed: $state" >&2
      return 2
    fi
    sleep 30
  done
}

wait_for_success "$jobs/vindr-v2-external-mount-audit-v2.json" || exit $?
wait_for_success "$jobs/llava-common-rag-causal-controls-v1.json" || exit $?

"$eval_python" - "$audit" "$manifest" "$images" <<'PY'
import json, sys
from pathlib import Path
audit_path, manifest_path, image_root = map(Path, sys.argv[1:])
row = json.load(audit_path.open())
checks = (
    row.get("protocol_version") == "vindr-readonly-external-subset-audit-v1",
    row.get("passed") is True,
    row.get("mount", {}).get("read_only") is True,
    row.get("selected_images") == 2341,
    row.get("validated_selected_images") == 2341,
    Path(row.get("manifest", "")).resolve() == manifest_path.resolve(),
    Path(row.get("image_root", "")).resolve() == image_root.resolve(),
    bool(row.get("ordered_selected_dicom_sha256")),
)
if not all(checks):
    raise SystemExit("external VinDr mount did not pass the exact frozen admission gate")
PY

mkdir -p "$jobs/locks"
exec 8>"$jobs/locks/gpu0-vindr-v2.lock"
flock 8

complete_collection() {
  local directory=$1 expected=$2
  "$eval_python" - "$directory" "$expected" <<'PY'
import json, sys
from pathlib import Path
directory, expected = Path(sys.argv[1]), int(sys.argv[2])
summary = directory / "summary.json"
metadata = directory / "metadata.jsonl"
features = directory / "hidden_states.npz"
if not all(path.is_file() for path in (summary, metadata, features)):
    raise SystemExit(1)
row = json.load(summary.open())
count = sum(bool(line.strip()) for line in metadata.read_text().splitlines())
raise SystemExit(0 if row.get("status") == "complete" and row.get("n") == expected and count == expected else 1)
PY
}

collect() {
  local output=$1 max_samples=${2:-}
  if complete_collection "$output" "${max_samples:-640}"; then
    echo "Reusing complete Hulu collection: $output"
    return 0
  fi
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite incomplete Hulu collection: $output" >&2
    return 2
  fi
  local extra=()
  [[ -n "$max_samples" ]] && extra=(--max-samples "$max_samples")
  PYTHONPATH=anchor "$hulu_python" -m corrected_sgta.collect_vindr_hidden_states_v2 \
    --model hulu --manifest "$manifest" --image-root "$images" \
    --output-dir "$output" --split dev --findings "${findings[@]}" \
    --votes 0 1 2 3 --layers 9 18 27 36 \
    --model-dir /home/dbw/models/Hulu-Med-4B --max-visual-tokens 1024 \
    --seed 42 "${extra[@]}"
}

collect corrected_runs/vindr_v2/hidden_canary_hulu_dev_v1 1 || exit $?
collect corrected_runs/vindr_v2/hidden_dev_hulu_all_findings_v1 || exit $?

# Release GPU 0 before nested CV and bootstrap analyses.
flock -u 8

env OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 PYTHONPATH=anchor \
  "$eval_python" -m corrected_sgta.screen_reader_residual_v1 \
  --features-dir corrected_runs/vindr_v2/hidden_dev_hulu_all_findings_v1 \
  --sampling-summary "$sampling" \
  --output corrected_runs/vindr_v2/hidden_dev_hulu_all_findings_v1/reader_residual_dev_screen_v1.json \
  --bootstrap-draws 5000 --seed 42

env OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 PYTHONPATH=anchor \
  "$eval_python" -m corrected_sgta.screen_virtual_reader_panel_dev_v1 \
  --dev-features-dir corrected_runs/vindr_v2/hidden_dev_hulu_all_findings_v1 \
  --sampling-summary "$sampling" \
  --output corrected_runs/vindr_v2/hidden_dev_hulu_all_findings_v1/virtual_reader_panel_dev_screen_v1.json \
  --bootstrap-draws 5000 --seed 42

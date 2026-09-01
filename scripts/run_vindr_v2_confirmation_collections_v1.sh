#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
eval_python=/opt/miniconda3/envs/huatuo/bin/python
hulu_python=/home/dbw/.venvs/hulumed/bin/python
jobs=corrected_runs/detached_jobs
manifest=/home/dbw/datasets/physionet/vindr-cxr/1.0.0/manifests_v2/reader_vote_manifest_v2.jsonl
audit=/home/dbw/datasets/physionet/vindr-cxr/1.0.0/manifests_v2/external_mount_audit_v1.json
images=/workspace/vinbigdata/train
findings=(aortic_enlargement cardiomegaly lung_opacity nodule_mass other_lesion pleural_effusion pleural_thickening pulmonary_fibrosis)

complete_collection() {
  local directory=$1 expected=$2
  "$eval_python" - "$directory" "$expected" <<'PY'
import json, sys
from pathlib import Path
directory, expected = Path(sys.argv[1]), int(sys.argv[2])
paths = (directory / "summary.json", directory / "metadata.jsonl", directory / "hidden_states.npz")
if not all(path.is_file() for path in paths):
    raise SystemExit(1)
summary = json.load(paths[0].open())
count = sum(bool(line.strip()) for line in paths[1].read_text().splitlines())
raise SystemExit(0 if summary.get("status") == "complete" and summary.get("n") == expected and count == expected else 1)
PY
}

valid_frozen_lock() {
  local lock=$1 model=$2 dev=$3
  "$eval_python" - "$lock" "$model" "$dev" <<'PY'
import hashlib, json, sys
from pathlib import Path

lock, model, dev = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
if not lock.is_file():
    raise SystemExit(1)
row = json.load(lock.open())
fingerprint = row.pop("fingerprint", None)
expected = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
hidden_hash = hashlib.sha256((dev / "hidden_states.npz").read_bytes()).hexdigest()
valid = (
    fingerprint == expected
    and row.get("status") == "frozen_before_confirmation_features"
    and row.get("model_id") == model
    and row.get("provenance", {}).get("dev_hidden_states_sha256") == hidden_hash
    and row.get("provenance", {}).get("confirmation_dir_absent_at_freeze") is True
)
raise SystemExit(0 if valid else 1)
PY
}

freeze_or_validate_lock() {
  local lock=$1 model=$2 screen=$3 dev=$4 confirmation=$5
  if valid_frozen_lock "$lock" "$model" "$dev"; then
    echo "Reusing integrity-validated pre-confirmation lock: $lock"
    return 0
  fi
  # The freezer itself refuses to run after any confirmation feature exists.
  # Thus a missing/invalid lock can never be reconstructed post hoc.
  PYTHONPATH=anchor "$eval_python" -m corrected_sgta.freeze_reader_residual_specs_v1 \
    --screen "$screen" --features-dir "$dev" \
    --confirmation-dir "$confirmation" --output "$lock" \
    --model-id "$model" --seed 42
  valid_frozen_lock "$lock" "$model" "$dev"
}

# The Hulu dev feature file is the dependency, not successful completion of
# every downstream CPU screen.  A killed analyzer must not force regeneration
# of the already valid 640-case GPU artifact.
while ! complete_collection corrected_runs/vindr_v2/hidden_dev_hulu_all_findings_v1 640; do
  "$eval_python" - "$jobs/vindr-v2-hulu-dev-and-screens-v1.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if p.is_file() and json.load(p.open()).get("status") == "failed":
    raise SystemExit(2)
raise SystemExit(0)
PY
  [[ "$?" == 2 ]] && { echo "Hulu dev collection failed" >&2; exit 2; }
  sleep 30
done

huatuo_screen=corrected_runs/vindr_v2/hidden_dev_huatuo_all_findings_v3/reader_residual_dev_unanimity_v1.json
hulu_screen=corrected_runs/vindr_v2/hidden_dev_hulu_all_findings_v1/reader_residual_dev_screen_v1.json
while [[ ! -s "$huatuo_screen" || ! -s "$hulu_screen" ]]; do
  "$eval_python" - "$jobs/vindr-v2-huatuo-reader-residual-unanimity-v1.json" "$jobs/vindr-v2-hulu-dev-and-screens-v3.json" <<'PY'
import json, sys
from pathlib import Path
for value in sys.argv[1:]:
    path = Path(value)
    if path.is_file() and json.load(path.open()).get("status") == "failed":
        raise SystemExit(2)
raise SystemExit(0)
PY
  [[ "$?" == 2 ]] && { echo "A required dev screen failed" >&2; exit 2; }
  sleep 30
done

mkdir -p corrected_runs/vindr_v2/reader_residual_locks
freeze_or_validate_lock \
  corrected_runs/vindr_v2/reader_residual_locks/huatuo_v1.json huatuo \
  "$huatuo_screen" corrected_runs/vindr_v2/hidden_dev_huatuo_all_findings_v3 \
  corrected_runs/vindr_v2/hidden_confirmation_huatuo_all_findings_v1 || exit $?
freeze_or_validate_lock \
  corrected_runs/vindr_v2/reader_residual_locks/hulu_v1.json hulu \
  "$hulu_screen" corrected_runs/vindr_v2/hidden_dev_hulu_all_findings_v1 \
  corrected_runs/vindr_v2/hidden_confirmation_hulu_all_findings_v1 || exit $?

"$eval_python" - "$audit" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
if not (
    row.get("passed") is True
    and row.get("mount", {}).get("read_only") is True
    and row.get("validated_selected_images") == 2341
    and bool(row.get("ordered_selected_dicom_sha256"))
):
    raise SystemExit("VinDr external mount admission is absent or invalid")
PY
[[ "$?" == 0 ]] || exit $?

mkdir -p "$jobs/locks"
exec 8>"$jobs/locks/gpu0-vindr-v2.lock"
flock 8
[[ "$?" == 0 ]] || exit $?

collect() {
  local model=$1 python=$2 output=$3 expected=$4 max_samples=${5:-}
  if complete_collection "$output" "$expected"; then
    echo "Reusing complete confirmation collection: $output"
    return 0
  fi
  local resume=()
  [[ -d "$output" ]] && resume=(--resume)
  local extra=()
  [[ -n "$max_samples" ]] && extra=(--max-samples "$max_samples")
  local layers model_dir model_extra=()
  if [[ "$model" == "huatuo" ]]; then
    layers=(7 14 21 28)
    model_dir=/home/dbw/models/HuatuoGPT-Vision-7B
    model_extra=(--huatuo-root /home/dbw/HuatuoGPT-Vision)
  else
    layers=(9 18 27 36)
    model_dir=/home/dbw/models/Hulu-Med-4B
    model_extra=(--max-visual-tokens 1024)
  fi
  PYTHONPATH=anchor "$python" -m corrected_sgta.collect_vindr_hidden_states_v2 \
    --model "$model" --manifest "$manifest" --image-root "$images" \
    --output-dir "$output" --split confirmation --findings "${findings[@]}" \
    --votes 0 1 2 3 --layers "${layers[@]}" --model-dir "$model_dir" \
    --seed 42 --checkpoint-every 64 "${model_extra[@]}" \
    "${resume[@]}" "${extra[@]}"
}

# One-case v3 canaries validate the resumable implementation and exact
# same-shape final-norm hook separately for both model families.
collect huatuo "$eval_python" corrected_runs/vindr_v2/hidden_canary_huatuo_confirmation_v3 1 1 || exit $?
collect hulu "$hulu_python" corrected_runs/vindr_v2/hidden_canary_hulu_confirmation_v3 1 1 || exit $?

collect huatuo "$eval_python" corrected_runs/vindr_v2/hidden_confirmation_huatuo_all_findings_v1 1920 || exit $?
collect hulu "$hulu_python" corrected_runs/vindr_v2/hidden_confirmation_hulu_all_findings_v1 1920 || exit $?

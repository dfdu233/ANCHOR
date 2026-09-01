#!/usr/bin/env bash
set -uo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

manifest=corrected_runs/unified_eval/inputs/baseline_matrix_v1/cxr_vishal.json
images=/home/dbw/ANCHOR/data/medheval/images
out=corrected_runs/paper_baselines_v1/full_matrix_v1/native_ce/huatuo/cxr_vishal/greedy
log=corrected_runs/detached_jobs/logs/baseline-native-ce-huatuo-cxr_vishal-greedy-repair-v1.log
state=corrected_runs/detached_jobs/baseline-native-ce-repair-v1.state.jsonl
python=/opt/miniconda3/envs/huatuo/bin/python

mkdir -p "$out" "$(dirname "$log")" corrected_runs/detached_jobs/locks
exec 8>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
flock 8
rc=1
for attempt in 1 2 3; do
  PYTHONPATH=. "$python" -m anchor.medeval.run_native_oe_vqa \
    --model huatuo --manifest "$manifest" --image-root "$images" \
    --output-dir "$out" --limit 0 --max-new-tokens 128 --seed 42 \
    --decode-mode greedy --num-beams 1 >>"$log" 2>&1
  rc=$?
  [[ "$rc" -eq 0 ]] && break
  if tail -80 "$log" | grep -Eq \
    'refusing to resume an incompatible|manifest qids are not unique|not an exact unique manifest prefix'; then
    break
  fi
done
flock -u 8

if [[ "$rc" -eq 0 ]]; then
  PYTHONPATH=. "$python" -m anchor.medeval.qualify_oe_generation \
    --manifest "$manifest" --answers "$out/answers.jsonl" --limit 5587 \
    --min-unique-rate 0 \
    --max-new-tokens 128 --max-cap-hit-rate 0.05 \
    --output "$out/qualification.json" >>"$log" 2>&1
  rc=$?
fi
if [[ "$rc" -eq 0 ]]; then
  PYTHONPATH=. "$python" -m anchor.corrected_sgta.evaluate_medheval_answers \
    --answers "$out/answers.jsonl" --questions "$manifest" \
    --output "$out/evaluation_ce_v7.json" >>"$log" 2>&1
  rc=$?
fi

/opt/miniconda3/bin/python - "$state" "$rc" "$log" <<'PY'
import datetime
import json
import sys

path, returncode, log = sys.argv[1:]
row = {
    "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model": "huatuo",
    "dataset": "cxr_vishal",
    "method": "greedy",
    "status": "completed" if returncode == "0" else "failed",
    "returncode": int(returncode),
    "expected": 5587,
    "log": log,
    "reason": "rerun after incompatible pre-beam-fix generation fingerprint",
}
with open(path, "a") as handle:
    handle.write(json.dumps(row) + "\n")
PY
exit "$rc"

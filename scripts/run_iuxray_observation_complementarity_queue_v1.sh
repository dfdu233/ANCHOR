#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=.

root=corrected_runs/daylong_idea_search_v1
pilot="$root/iuxray_observation_pilot64_v1"
full="$root/iuxray_observation_v1"
image_root=data/medheval/images/IU-Xray
lock=corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
state="$root/iuxray_observation_queue_state.jsonl"
python=/opt/miniconda3/envs/huatuo/bin/python

mkdir -p "$root/logs"
exec 8>"$lock"

stamp() {
  "$python" - "$state" "$1" "$2" <<'PY'
import datetime,json,sys
with open(sys.argv[1], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": sys.argv[2],
        "status": sys.argv[3],
    }) + "\n")
PY
}

score() {
  local model=$1 manifest=$2 output=$3
  score_at_root "$model" "$manifest" "$output" "$image_root"
}

score_at_root() {
  local model=$1 manifest=$2 output=$3 root=$4
  local resume=()
  [[ -d "$output" ]] && resume=(--resume)
  "$python" -m anchor.corrected_sgta.run_claim_universe_scoring \
    --model "$model" \
    --questions "$manifest" \
    --image-root "$root" \
    --output-dir "$output" \
    --skip-null \
    "${resume[@]}"
}

analyze() {
  local left=$1 right=$2 output=$3
  "$python" -m anchor.corrected_sgta.iuxray_observation_complementarity_v1 analyze \
    --view0-raw "$left/raw.jsonl" \
    --view1-raw "$right/raw.jsonl" \
    --output "$output" \
    --bootstrap-draws 5000 \
    --permutations 2000 \
    --seed 42
}

decision() {
  "$python" - "$1" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])
PY
}

lesion="$root/lesion_transplant_n128_v2"
stamp lesion_transplant queued
flock 8
stamp lesion_transplant running
if [[ ! -f "$lesion/raw.jsonl" ]]; then
  lesion_resume=()
  [[ -d "$lesion" ]] && lesion_resume=(--resume)
  "$python" -m anchor.corrected_sgta.run_vindr_lesion_transplant_v1 \
    --csv /workspace/vinbigdata/train.csv \
    --dicom-root /workspace/vinbigdata/train \
    --output-dir "$lesion" \
    --per-finding 128 \
    --seed 20260806 \
    "${lesion_resume[@]}"
else
  "$python" -m anchor.corrected_sgta.run_vindr_lesion_transplant_v1 \
    --csv /workspace/vinbigdata/train.csv \
    --dicom-root /workspace/vinbigdata/train \
    --output-dir "$lesion" \
    --per-finding 128 \
    --seed 20260806 \
    --resume
fi
if [[ ! -f "$lesion/analysis.json" ]]; then
  "$python" -m anchor.corrected_sgta.analyze_vindr_lesion_relocation_v1 \
    --raw "$lesion/raw.jsonl" \
    --output "$lesion/analysis.json" \
    --draws 10000
fi
stamp lesion_transplant complete

patch_huatuo="$root/patch_scores_huatuo_v1"
stamp patch_huatuo queued
stamp patch_huatuo running
patch_resume=()
[[ -d "$patch_huatuo" ]] && patch_resume=(--resume)
"$python" -m anchor.corrected_sgta.collect_sparse_patch_scores_v1 \
  --model huatuo \
  --raw-visual corrected_runs/evidence_addressability_gate_v2/raw_huatuo_v1 \
  --directions "$root/patch_directions_huatuo_v1/directions.npz" \
  --image-root /workspace/vinbigdata/train \
  --output-dir "$patch_huatuo" \
  --model-dir /home/dbw/models/HuatuoGPT-Vision-7B \
  --progress-every 64 \
  "${patch_resume[@]}"
flock -u 8
"$python" -m anchor.corrected_sgta.analyze_sparse_patch_scan_v1 \
  --model huatuo \
  --development-hidden corrected_runs/vindr_v2/hidden_confirmation_huatuo_recoverability_v1 \
  --confirmation-hidden corrected_runs/evidence_addressability_gate_v2/hidden_fresh_huatuo_v2 \
  --patch-scores "$patch_huatuo" \
  --output "$root/sparse_patch_scan_huatuo_v1.json"
"$python" -m anchor.corrected_sgta.analyze_search_tax_phase_v1 \
  --model huatuo \
  --development-hidden corrected_runs/vindr_v2/hidden_confirmation_huatuo_recoverability_v1 \
  --confirmation-hidden corrected_runs/evidence_addressability_gate_v2/hidden_fresh_huatuo_v2 \
  --patch-scores "$patch_huatuo" \
  --reader-labels-csv /home/dbw/datasets/physionet/vindr-cxr/1.0.0/annotations/image_labels_train.csv \
  --output "$root/search_tax_phase_huatuo_v1.json"

search_reuse="$root/search_reuse_huatuo_v1"
stamp search_reuse_huatuo building
"$python" -m anchor.corrected_sgta.search_reuse_crop_probe_v1 build \
  --patch-scores "$patch_huatuo" \
  --development-hidden corrected_runs/vindr_v2/hidden_confirmation_huatuo_recoverability_v1 \
  --confirmation-hidden corrected_runs/evidence_addressability_gate_v2/hidden_fresh_huatuo_v2 \
  --reader-labels-csv /home/dbw/datasets/physionet/vindr-cxr/1.0.0/annotations/image_labels_train.csv \
  --dicom-root /workspace/vinbigdata/train \
  --output-dir "$search_reuse" \
  --claim-counts 1 7 \
  --region-counts 16 64 361 \
  --window-side 6
stamp search_reuse_huatuo queued
flock 8
stamp search_reuse_huatuo running
score_at_root huatuo "$search_reuse/manifest.json" "$search_reuse/scores" "$search_reuse/images"
flock -u 8
if [[ ! -f "$search_reuse/analysis.json" ]]; then
  "$python" -m anchor.corrected_sgta.search_reuse_crop_probe_v1 analyze \
    --selections "$search_reuse/selections.jsonl" \
    --raw "$search_reuse/scores/raw.jsonl" \
    --output "$search_reuse/analysis.json" \
    --bootstrap-draws 5000 \
    --seed 20260812
fi
stamp search_reuse_huatuo complete
stamp patch_huatuo complete

patch_pass() {
  "$python" - "$1" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["result"]["gate"]["pass"] else 1)
PY
}

if patch_pass "$root/sparse_patch_scan_huatuo_v1.json"; then
  patch_hulu="$root/patch_scores_hulu_v1"
  stamp patch_hulu queued
  flock 8
  stamp patch_hulu running
  patch_resume=()
  [[ -d "$patch_hulu" ]] && patch_resume=(--resume)
  "$python" -m anchor.corrected_sgta.collect_sparse_patch_scores_v1 \
    --model hulu \
    --raw-visual corrected_runs/evidence_addressability_gate_v2/raw_hulu_v1 \
    --directions "$root/patch_directions_hulu_v1/directions.npz" \
    --image-root /workspace/vinbigdata/train \
    --output-dir "$patch_hulu" \
    --model-dir /home/dbw/models/Hulu-Med-4B \
    --max-visual-tokens 1024 \
    --progress-every 64 \
    "${patch_resume[@]}"
  flock -u 8
  "$python" -m anchor.corrected_sgta.analyze_sparse_patch_scan_v1 \
    --model hulu \
    --development-hidden corrected_runs/vindr_v2/hidden_confirmation_hulu_recoverability_v1 \
    --confirmation-hidden corrected_runs/evidence_addressability_gate_v2/hidden_fresh_hulu_v3 \
    --patch-scores "$patch_hulu" \
    --output "$root/sparse_patch_scan_hulu_v1.json"
  "$python" -m anchor.corrected_sgta.analyze_search_tax_phase_v1 \
    --model hulu \
    --development-hidden corrected_runs/vindr_v2/hidden_confirmation_hulu_recoverability_v1 \
    --confirmation-hidden corrected_runs/evidence_addressability_gate_v2/hidden_fresh_hulu_v3 \
    --patch-scores "$patch_hulu" \
    --reader-labels-csv /home/dbw/datasets/physionet/vindr-cxr/1.0.0/annotations/image_labels_train.csv \
    --output "$root/search_tax_phase_hulu_v1.json"
  stamp patch_hulu complete
else
  stamp patch_hulu stopped_huatuo_no_go
fi

stamp pilot_huatuo queued
flock 8
stamp pilot_huatuo running
score huatuo "$pilot/view0.json" "$pilot/scores_huatuo_view0"
score huatuo "$pilot/view1.json" "$pilot/scores_huatuo_view1"
flock -u 8
analyze "$pilot/scores_huatuo_view0" "$pilot/scores_huatuo_view1" \
  "$pilot/analysis_huatuo.json"
stamp pilot_huatuo complete

if [[ "$(decision "$pilot/analysis_huatuo.json")" != GO ]]; then
  stamp promotion stopped_huatuo_no_go
  exit 0
fi

stamp pilot_hulu queued
flock 8
stamp pilot_hulu running
score hulu "$pilot/view0.json" "$pilot/scores_hulu_view0"
score hulu "$pilot/view1.json" "$pilot/scores_hulu_view1"
flock -u 8
analyze "$pilot/scores_hulu_view0" "$pilot/scores_hulu_view1" \
  "$pilot/analysis_hulu.json"
stamp pilot_hulu complete

if [[ "$(decision "$pilot/analysis_hulu.json")" != GO ]]; then
  stamp promotion stopped_hulu_no_go
  exit 0
fi

for model in huatuo hulu; do
  stamp "full_${model}" queued
  flock 8
  stamp "full_${model}" running
  score "$model" "$full/view0.json" "$full/scores_${model}_view0"
  score "$model" "$full/view1.json" "$full/scores_${model}_view1"
  flock -u 8
  analyze "$full/scores_${model}_view0" "$full/scores_${model}_view1" \
    "$full/analysis_${model}.json"
  stamp "full_${model}" complete
done

stamp promotion full_complete

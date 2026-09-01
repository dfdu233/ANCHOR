#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
mkdir -p corrected_runs/detached_jobs/locks
exec 9>corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
flock 9

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/dbw/ANCHOR/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=anchor
export ANCHOR_PYTHON=/opt/miniconda3/envs/huatuo/bin/python
export ANCHOR_MODEL_PATH=/home/dbw/models/LLaVA-Med-v1.5-mistral-7b

python=/opt/miniconda3/envs/huatuo/bin/python
manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
images=/home/dbw/datasets/public/vqa_rad_hf/test_images
t1=corrected_runs/unified_eval/smoke/vista_llava_t1_t2_v1/t1
greedy="$t1/vqa_rad/official_test_oe/open_vqa/greedy/chunk_0000.answers.jsonl"
identity="$t1/vista_off_vs_greedy.json"
root=corrected_runs/unified_eval/smoke/vista_llava_t2_v4

"$python" - "$identity" <<'PY'
import json
import sys

identity = json.load(open(sys.argv[1]))
if not identity.get("passed") or identity.get("generated_token_exact_rate") != 1.0:
    raise SystemExit("frozen VISTA T1 identity is absent or failed")
PY

"$python" -m corrected_sgta.run_llava_med_generation_matrix \
  --question-file "$manifest" --image-folder "$images" --out "$root" \
  --source vqa_rad --dataset official_test_oe --task open_vqa \
  --methods VISTA --chunk-size 32 --limit 32 \
  --max-new-tokens 64 --conv-mode mistral_instruct --seed 42 \
  --disable-keyword-stopping --qualification-run

vista="$root/vqa_rad/official_test_oe/open_vqa/VISTA/chunk_0000.answers.jsonl"
"$python" -m anchor.medeval.evaluate_backend_conformance \
  --canonical "$greedy" --candidate "$vista" \
  --min-normalized-exact 0 --min-token-f1 0 \
  --output "$root/vista_activation_vs_greedy.json"

"$python" - "$root/vista_activation_vs_greedy.json" <<'PY'
import json
import sys

activation = json.load(open(sys.argv[1]))
if not activation.get("passed"):
    raise SystemExit("VISTA T2 output quality failed")
if activation.get("generated_token_exact_rate") == 1.0:
    raise SystemExit("VISTA T2 did not change any generated token sequence")
PY

flock -u 9

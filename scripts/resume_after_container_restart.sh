#!/usr/bin/env bash
set -euo pipefail
cd /home/dbw/ANCHOR

sentinel=/home/dbw/.anchor_persistent_volume_v1.json
marker=corrected_runs/detached_jobs/container-restart-requested.json
state=corrected_runs/detached_jobs/common-rag-ce-ladder-v3.json

# A recreated container loses /opt even though /home/dbw is persistent.  Reuse
# the trace-certified persistent runtime at its canonical path instead of
# silently falling back to the container's system Python.
if [[ ! -x /opt/miniconda3/envs/huatuo/bin/python && \
      -x /home/dbw/.runtime/miniconda3/envs/huatuo/bin/python ]]; then
  ln -sfn /home/dbw/.runtime/miniconda3 /opt/miniconda3
fi
[[ -x /opt/miniconda3/envs/huatuo/bin/python ]] || {
  echo "Missing /opt/miniconda3 Huatuo runtime; restart the existing container or recreate from the same image/runtime." >&2
  exit 4
}
[[ -x /home/dbw/.venvs/hulumed/bin/python ]] || {
  echo "Missing mounted Hulu runtime under /home/dbw/.venvs." >&2
  exit 4
}
if ! command -v tmux >/dev/null 2>&1; then
  apt-get update
  apt-get install -y tmux
fi
[[ -s "$sentinel" ]] || {
  echo "Persistent /home/dbw volume sentinel is missing; refusing to run on an empty/wrong mount." >&2
  exit 4
}
for required in \
  /home/dbw/ANCHOR \
  /home/dbw/models/HuatuoGPT-Vision-7B/config.json \
  /home/dbw/models/Hulu-Med-4B/config.json \
  /home/dbw/datasets/physionet/vindr-cxr/1.0.0/manifests/summary.json; do
  [[ -e "$required" ]] || { echo "Missing mounted prerequisite: $required" >&2; exit 4; }
done
nvidia-smi >/dev/null
export HF_HOME=/home/dbw/.cache/huggingface

# Do not append post-restart generations to prefix-safe JSONL files until the
# recreated runtimes reproduce frozen pre-restart token IDs.  These are the
# two backends with unfinished RAG arms; Huatuo's completed arms remain frozen.
identity_root=corrected_runs/unified_eval/sanity/post_restart_runtime_identity_v1
identity_report="$identity_root/identity.json"
if [[ ! -s "$identity_report" ]] || ! /opt/miniconda3/envs/huatuo/bin/python - "$identity_report" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
raise SystemExit(0 if report.get("passed") is True else 1)
PY
then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export CUDA_VISIBLE_DEVICES=0
  mkdir -p "$identity_root"

  hulu_manifest=corrected_runs/unified_eval/rag/common_protocol_v1/mimic/visual_ce_v2/t3_n200_top3/prompts/no_context.json
  hulu_canonical=corrected_runs/unified_eval/rag/common_protocol_v1/mimic/visual_ce_v2/ladder_v3/T2_n32/hulu/no_context/answers.jsonl
  PYTHONPATH=anchor /home/dbw/.venvs/hulumed/bin/python \
    -m anchor.medeval.run_native_oe_vqa \
    --model hulu --manifest "$hulu_manifest" \
    --image-root /home/dbw/ANCHOR/data/medheval/images \
    --output-dir "$identity_root/hulu" --limit 32 \
    --max-new-tokens 128 --seed 42
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.medeval.evaluate_backend_conformance \
    --canonical "$hulu_canonical" \
    --candidate "$identity_root/hulu/answers.jsonl" \
    --min-normalized-exact 1 --min-token-f1 1 --require-token-exact \
    --output "$identity_root/hulu.conformance.json"

  llava_manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
  llava_canonical=corrected_runs/unified_eval/sanity/llava_canonical_runtime_gate_v2/n32/canonical/answers.jsonl
  PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.medeval.run_native_oe_vqa \
    --model llava --manifest "$llava_manifest" \
    --image-root /home/dbw/datasets/public/vqa_rad_hf/test_images \
    --output-dir "$identity_root/llava" --limit 32 \
    --max-new-tokens 64 --seed 42
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.medeval.evaluate_backend_conformance \
    --canonical "$llava_canonical" \
    --candidate "$identity_root/llava/answers.jsonl" \
    --min-normalized-exact 1 --min-token-f1 1 --require-token-exact \
    --output "$identity_root/llava.conformance.json"

  /opt/miniconda3/envs/huatuo/bin/python - \
    "$identity_root/hulu.conformance.json" \
    "$identity_root/llava.conformance.json" "$identity_report" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
rows = [json.load(open(path)) for path in sys.argv[1:3]]
payload = {
    "protocol": "post-restart-runtime-identity-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "passed": all(row.get("passed") is True for row in rows),
    "backends": {name: row for name, row in zip(("hulu", "llava"), rows)},
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2) + "\n")
raise SystemExit(0 if payload["passed"] else 2)
PY
fi

rm -f "$marker"

# Resume the prefix-safe common RAG ladder if it was nonterminal at restart.
if [[ -s "$state" ]] && /opt/miniconda3/envs/huatuo/bin/python - "$state" <<'PY'
import json, sys
status = json.load(open(sys.argv[1])).get("status")
raise SystemExit(0 if status in {"starting", "running", "paused_for_container_restart"} else 1)
PY
then
  python scripts/start_detached_job.py \
    --name common-rag-ce-ladder-v3 \
    --log corrected_runs/detached_jobs/common-rag-ce-ladder-v3.log \
    --state "$state" \
    bash scripts/run_common_rag_ce_ladder_v3.sh
fi

# Start the recovery watchdog after explicit recovery, avoiding a duplicate race.
python scripts/start_detached_job.py \
  --name research-watchdog-v1 \
  --log corrected_runs/detached_jobs/research-watchdog-v1.log \
  --state corrected_runs/detached_jobs/research-watchdog-v1.json \
  /opt/miniconda3/bin/python scripts/research_watchdog.py --interval 30 || true

vindr=/home/dbw/datasets/physionet/vindr-cxr/1.0.0
external_vindr=/workspace/vinbigdata/train
v2_manifest="$vindr/manifests_v2/reader_vote_manifest_v2.jsonl"
v2_audit="$vindr/manifests_v2/external_mount_audit_v1.json"
if [[ -d "$external_vindr" && -s "$v2_manifest" ]] && \
   [[ "$(find "$external_vindr" -maxdepth 1 -type f -name '*.dicom' | wc -l)" -eq 15000 ]]; then
  echo "Using immutable full VinDr mount at $external_vindr; selective password download is superseded."
  if ! /opt/miniconda3/envs/huatuo/bin/python - "$v2_audit" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
report = json.load(p.open()) if p.is_file() else {}
raise SystemExit(
    0
    if report.get("protocol_version") == "vindr-readonly-external-subset-audit-v1"
    and report.get("passed") is True
    else 1
)
PY
  then
    audit_state=corrected_runs/detached_jobs/vindr-v2-external-mount-audit-v2.json
    if ! [[ -s "$audit_state" ]] || ! /opt/miniconda3/bin/python - "$audit_state" <<'PY'
import json, sys
status = json.load(open(sys.argv[1])).get("status")
raise SystemExit(0 if status in {"starting", "running"} else 1)
PY
    then
      python scripts/start_detached_job.py \
        --name vindr-v2-external-mount-audit-v2 \
        --log corrected_runs/detached_jobs/vindr-v2-external-mount-audit-v2.log \
        --state "$audit_state" \
        bash -lc "while /opt/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('corrected_runs/detached_jobs/vindr-v2-huatuo-dev-hidden-all-findings-v3.json')
raise SystemExit(0 if p.is_file() and json.load(p.open()).get('status') in {'starting', 'running'} else 1)
PY
do sleep 30; done; exec env PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.audit_vindr_external_mount --manifest '$v2_manifest' --source-csv '$vindr/annotations/image_labels_train.csv' --image-root '$external_vindr' --workers 4 --output '$v2_audit'"
    fi
  fi
else
  if ! /opt/miniconda3/envs/huatuo/bin/python - "$vindr/manifests/dicom_download_audit.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
report = json.load(p.open()) if p.is_file() else {}
raise SystemExit(
    0
    if report.get("protocol_version") == "vindr-selective-dicom-audit-v2"
    and report.get("passed") is True
    else 1
)
PY
  then
    tmux has-session -t vindr-selective-download 2>/dev/null || \
      tmux new-session -d -s vindr-selective-download \
        "cd /home/dbw/ANCHOR && bash scripts/download_vindr_subset.sh images '$vindr'"
    echo "VinDr full mount is absent; fallback selective download awaits authentication."
    echo "Run: tmux attach -t vindr-selective-download"
  fi

  tmux has-session -t vindr-post-download 2>/dev/null || \
    tmux new-session -d -s vindr-post-download \
      "cd /home/dbw/ANCHOR && bash scripts/vindr_post_download_v1.sh"
  tmux has-session -t vindr-mechanism-boundary 2>/dev/null || \
    tmux new-session -d -s vindr-mechanism-boundary \
      "cd /home/dbw/ANCHOR && bash scripts/run_vindr_layer_boundary_formal_v1.sh; exec bash"
fi
tmux has-session -t common-rag-finalize-v3 2>/dev/null || \
  tmux new-session -d -s common-rag-finalize-v3 \
    "cd /home/dbw/ANCHOR && bash scripts/finalize_common_rag_ce_ladder_v3.sh; exec bash"

python scripts/research_status.py
tmux list-sessions

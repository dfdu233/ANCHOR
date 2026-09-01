#!/usr/bin/env bash
set -euo pipefail
cd /home/dbw/ANCHOR

state=corrected_runs/detached_jobs/common-rag-ce-ladder-v3.json
while /opt/miniconda3/envs/huatuo/bin/python - "$state" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
raise SystemExit(0 if p.exists() and json.load(p.open()).get("status") in {"starting", "running"} else 1)
PY
do
  sleep 60
done

export HF_HOME=/home/dbw/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
root=corrected_runs/unified_eval/rag/common_protocol_v1
for dataset in iuxray mimic; do
  if [[ "$dataset" == "iuxray" ]]; then
    images=/home/dbw/ANCHOR/data/medheval/images/IU-Xray
  else
    images=/home/dbw/ANCHOR/data/medheval/images
  fi
  prompts="$root/$dataset/visual_ce_v2/t3_n200_top3/prompts"
  PYTHONPATH=. /home/dbw/.venvs/hulumed/bin/python \
    -m anchor.medeval.audit_hulu_context \
    --baseline-manifest "$prompts/no_context.json" \
    --candidate-manifest "$prompts/rag.json" \
    --image-root "$images" --model-root /home/dbw/models/Hulu-Med-4B \
    --max-new-tokens 128 --limit 200 \
    --output "$root/$dataset/visual_ce_v2/ladder_v3/T3_n200/hulu/context_budget_audit.json"
done

PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.medeval.summarize_common_rag_ladder \
  --root corrected_runs/unified_eval/rag/common_protocol_v1 \
  --output corrected_runs/unified_eval/rag/common_protocol_v1/visual_ce_ladder_v3_summary.json \
  --registry corrected_runs/unified_eval/provenance/artifact_registry_v1.jsonl

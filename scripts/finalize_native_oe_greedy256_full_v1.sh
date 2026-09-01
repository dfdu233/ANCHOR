#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
upstream=corrected_runs/detached_jobs/native-oe-greedy256-full-v1.json
while true; do
  upstream_status=$(/opt/miniconda3/bin/python - "$upstream" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
print(json.load(path.open()).get("status", "missing") if path.is_file() else "missing")
PY
)
  [[ "$upstream_status" == "done" ]] && break
  if [[ "$upstream_status" == "failed" ]]; then
    echo "native greedy256 OE generation failed; acceptance cannot continue" >&2
    exit 2
  fi
  sleep 30
done

manifest=corrected_runs/unified_eval/inputs/vqa_rad_official_test_oe.json
registry=corrected_runs/unified_eval/provenance/artifact_registry_v1.jsonl
acceptance=corrected_runs/unified_eval/full/native_oe_greedy256_acceptance_v1.json
PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.accept_native_oe_full \
  --manifest "$manifest" --registry "$registry" \
  --model-output hulu corrected_runs/unified_eval/full/hulu_native_vqa_rad_oe_greedy256_v1 \
  --model-output llava corrected_runs/unified_eval/full/llava_native_vqa_rad_oe_greedy256_v1 \
  --output "$acceptance"

PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python -m anchor.medeval.audit_method_evidence_ladder \
  --t0-audit corrected_runs/unified_eval/provenance/method_ladder_t0_v2.json \
  --registry "$registry" \
  --identity-gate corrected_runs/unified_eval/sanity/post_restart_runtime_identity_v1/identity.json \
  --rag-causal-summary corrected_runs/unified_eval/rag/common_protocol_v1/visual_ce_ladder_v3_causal_controls_v2.json \
  --output corrected_runs/unified_eval/provenance/method_evidence_ladder_v3.json

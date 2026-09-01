#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_rag_dual_track_v1 \
  --method-evidence corrected_runs/unified_eval/provenance/method_evidence_ladder_v8.json \
  --factmm-qualification corrected_runs/unified_eval/provenance/factmm_rag_t0_qualification_v1.json \
  --output corrected_runs/unified_eval/provenance/rag_dual_track_qualification_v1.json

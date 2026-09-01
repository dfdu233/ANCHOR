#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_factmm_rag_archive_v3 \
  --archive /home/dbw/model_cache/factmm_rag/official_retriever_v1/model.zip \
  --output corrected_runs/unified_eval/provenance/factmm_rag_official_archive_audit_v3.json

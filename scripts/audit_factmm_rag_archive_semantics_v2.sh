#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_factmm_rag_archive_semantics_v2 \
  --archive /home/dbw/model_cache/factmm_rag/official_retriever_v1/model.zip \
  --inventory corrected_runs/unified_eval/provenance/factmm_rag_official_archive_audit_v3.json \
  --download-provenance /home/dbw/model_cache/factmm_rag/official_retriever_v1/download_provenance_v3.json \
  --materialize-dir /home/dbw/model_cache/factmm_rag/official_retriever_v1/materialized_v2 \
  --output corrected_runs/unified_eval/provenance/factmm_rag_official_archive_semantic_audit_v2.json

#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
root=/home/dbw/model_cache/factmm_rag/official_retriever_v1
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.audit_factmm_rag_checkpoint_v1 \
  --checkpoint "$root/factmm_rag_retriever_checkpoint.pt" \
  --download-provenance "$root/download_provenance.json" \
  --output "$root/checkpoint_audit.json"

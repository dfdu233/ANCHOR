#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.qualify_factmm_rag_t0_v2 \
  --source-audit corrected_runs/unified_eval/provenance/method_ladder_t0_v3.json \
  --archive-inventory corrected_runs/unified_eval/provenance/factmm_rag_official_archive_audit_v3.json \
  --semantic-audit corrected_runs/unified_eval/provenance/factmm_rag_official_archive_semantic_audit_v2.json \
  --repository third_party/FactMM-RAG \
  --mimic-root /home/dbw/datasets/physionet/mimic-cxr-jpg/2.1.0 \
  --chexpert-root /home/dbw/datasets/chexpert \
  --output corrected_runs/unified_eval/provenance/factmm_rag_t0_qualification_v2.json

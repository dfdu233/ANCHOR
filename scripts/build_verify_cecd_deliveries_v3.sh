#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dbw/ANCHOR
PACK=/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_pack_v2
DELIVERY=/home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_admission_pack_v2_reviewer_deliveries_v3

cd "$ROOT"
mkdir -p "$DELIVERY"
.venv-full/bin/python -m anchor.medeval.package_cecd_deliveries_v3 \
  --pack-dir "$PACK" \
  --output-dir "$DELIVERY"
.venv-full/bin/python -m anchor.medeval.verify_cecd_deliveries_v3 \
  --pack-dir "$PACK" \
  --delivery-dir "$DELIVERY" \
  --output "$DELIVERY/verification.json"

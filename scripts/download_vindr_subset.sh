#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-annotations}"
VINDR_ROOT="${2:-/home/dbw/datasets/physionet/vindr-cxr/1.0.0}"
VINDR_USER="${PHYSIONET_USER:-dfdu233}"
VINDR_BASE="https://physionet.org/files/vindr-cxr/1.0.0"

mkdir -p "${VINDR_ROOT}/annotations" "${VINDR_ROOT}/train"

case "${PHASE}" in
  annotations)
    # Password input is deliberately interactive. Do not redirect this command
    # or place the credential in an environment variable, .netrc, script, or log.
    wget -c \
      --user "${VINDR_USER}" \
      --ask-password \
      -O "${VINDR_ROOT}/annotations/image_labels_train.csv" \
      "${VINDR_BASE}/annotations/image_labels_train.csv"

    PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
      -m corrected_sgta.prepare_vindr_reader_manifest \
      --labels-csv "${VINDR_ROOT}/annotations/image_labels_train.csv" \
      --output-dir "${VINDR_ROOT}/manifests" \
      --ontology configs/missing_third_state_vindr_ontology.json \
      --min-per-bin 100 \
      --samples-per-bin 100

    # Preserve the raw four-bin manifest as the primary reference.  Fit a
    # separate dev-only reader-bias sensitivity model; it never overwrites or
    # upgrades the official observed votes.
    PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
      -m corrected_sgta.fit_reader_adjusted_support \
      --manifest "${VINDR_ROOT}/manifests/reader_vote_manifest.jsonl" \
      --output-dir "${VINDR_ROOT}/manifests/reader_adjusted_support"

    echo "Stopped after annotations and manifest construction."
    echo "Audit ${VINDR_ROOT}/manifests/summary.json before running phase 'images'."
    exit 0
    ;;
  images)
    if [[ ! -s "${VINDR_ROOT}/manifests/summary.json" || \
          ! -s "${VINDR_ROOT}/manifests/image_urls.txt" ]]; then
      echo "Missing audited manifest; run phase 'annotations' first." >&2
      exit 4
    fi
    ;;
  triplets)
    if [[ ! -s "${VINDR_ROOT}/manifests/reader_vote_manifest.jsonl" ]]; then
      echo "Missing reader-vote manifest; run phase 'annotations' first." >&2
      exit 4
    fi
    ;;
  *)
    echo "Usage: $0 {annotations|images|triplets} [VINDR_ROOT]" >&2
    exit 64
    ;;
esac

if [[ "${PHASE}" == "triplets" ]]; then
  PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
    -m anchor.medeval.audit_vindr_download \
    --url-manifest "${VINDR_ROOT}/manifests/image_urls.txt" \
    --image-root "${VINDR_ROOT}/train" \
    --output "${VINDR_ROOT}/manifests/dicom_download_audit.json"

  PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
    -m corrected_sgta.prepare_vindr_selectivity_triplets \
    --reader-manifest "${VINDR_ROOT}/manifests/reader_vote_manifest.jsonl" \
    --image-root "${VINDR_ROOT}" \
    --output-dir "${VINDR_ROOT}/manifests/clinical_selectivity"

  PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
    -m corrected_sgta.prepare_vindr_commitment_tetrads \
    --reader-manifest "${VINDR_ROOT}/manifests/reader_vote_manifest.jsonl" \
    --image-root "${VINDR_ROOT}" \
    --output-dir "${VINDR_ROOT}/manifests/commitment_tetrads"
  exit 0
fi

available_kb="$(df --output=avail -k "${VINDR_ROOT}" | tail -n 1 | tr -d ' ')"
reserve_kb=$((100 * 1024 * 1024))
download_guard_kb=$((1024 * 1024))
if (( available_kb <= reserve_kb + download_guard_kb )); then
  echo "Refusing image download: less than 100 GiB would remain." >&2
  exit 2
fi
quota_kb=$((available_kb - reserve_kb - download_guard_kb))

wget -N -c \
  --user "${VINDR_USER}" \
  --ask-password \
  --quota="${quota_kb}k" \
  --no-directories \
  --directory-prefix "${VINDR_ROOT}/train" \
  --input-file "${VINDR_ROOT}/manifests/image_urls.txt"

available_kb="$(df --output=avail -k "${VINDR_ROOT}" | tail -n 1 | tr -d ' ')"
if (( available_kb <= reserve_kb )); then
  echo "WARNING: free space is now at or below the 100 GiB reserve." >&2
  exit 3
fi

echo "Image phase complete. Verify selected DICOMs, then run phase 'triplets'."

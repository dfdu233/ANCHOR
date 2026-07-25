#!/usr/bin/env bash
set -euo pipefail

DATASETS="mimic_cxr_rule,chexpert_subset_report"
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

ARGS=(--datasets "$DATASETS")
if [[ "$CHECK_ONLY" == "1" ]]; then
  ARGS+=(--check-only)
fi
PYTHONPATH=. python -m anchor.runners.registry "${ARGS[@]}"

if [[ "$CHECK_ONLY" == "0" ]]; then
  echo "Data are expected to be present through Git LFS. Run 'git lfs pull' if files are missing."
fi

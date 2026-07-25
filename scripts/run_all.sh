#!/usr/bin/env bash
set -euo pipefail

DATASETS="mimic_cxr_rule,chexpert_subset_report"
METHODS="greedy,source_margin,source_word_center"
JUDGES="rule_parser,rouge"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --judges) JUDGES="$2"; shift 2 ;;
    --models) shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

PYTHONPATH=. python -m anchor.runners.registry \
  --datasets "$DATASETS" \
  --methods "$METHODS" \
  --judges "$JUDGES"

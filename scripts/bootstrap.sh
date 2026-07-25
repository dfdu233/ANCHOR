#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"

if [[ "$DRY_RUN" == "1" ]]; then
  command -v git >/dev/null
  command -v git-lfs >/dev/null || command -v git-lfs >/dev/null 2>&1 || true
  python - <<'PY'
import sys
print(sys.version)
PY
  exit 0
fi

if command -v conda >/dev/null 2>&1; then
  conda env list | awk '{print $1}' | grep -qx anchor || conda env create -f environment.yml
  echo "Activate with: conda activate anchor"
else
  python -m pip install -U pip
  python -m pip install -e .
fi

git lfs install || true
python -m pip install -e .

#!/usr/bin/env bash
set -euo pipefail

cd /home/dbw/ANCHOR

# This must be an audited, repository-local worker implementing the protocol
# documented in CECD_DUAL_SEMANTICS_CONTROLLED_RUNNER_V1.md.  There is no
# permissive fallback to a surrogate or to the unresolved Treble release.
worker=${CECD_DUAL_WORKER:?set CECD_DUAL_WORKER to the frozen repository-local worker}

args=(
  --authorization "${CECD_DUAL_AUTHORIZATION:-corrected_runs/vindr_v2/cecd_dual_semantics_v1/authorization.json}"
  --preflight "${CECD_DUAL_PREFLIGHT:-configs/cecd_dual_semantics_preflight_v1.json}"
  --worker "$worker"
  --huatuo-python "${CECD_DUAL_HUATUO_PYTHON:-/opt/miniconda3/envs/huatuo/bin/python}"
  --hulu-python "${CECD_DUAL_HULU_PYTHON:-/home/dbw/.venvs/hulumed/bin/python}"
  --gpu-lock corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
)

if [[ ${CECD_DUAL_EXECUTE:-0} == 1 ]]; then
  args+=(--execute)
fi
if [[ ${CECD_DUAL_EXECUTE_CE_ONLY:-0} == 1 ]]; then
  args+=(--execute-ce-only)
fi
if [[ ${CECD_DUAL_RESUME_AFTER_FAILURE:-0} == 1 ]]; then
  args+=(--resume-after-failure)
fi

exec .venv-full/bin/python -u \
  -m anchor.corrected_sgta.run_cecd_dual_semantics_controlled_v1 \
  "${args[@]}"

#!/usr/bin/env bash
set -euo pipefail

# Retired fail-closed: this one-model wrapper did not bind its factorial
# artifacts to the human-admission result.  The v2 wrapper validates and
# propagates one admission into both exact model runs and their joint verifier.
echo "RETIRED: use scripts/run_cecd_two_model_stage1_v2.sh; no command was run" >&2
exit 64

# CECD dual-semantics transition operational closure

**Date:** 2026-08-03  
**Scope:** CPU/outcome-blind control-plane repair only. No GPU job, model arm,
human-return synthesis, outcome analysis, or paper claim was opened.

## Closed dead ends

1. **Missing preflight producer.**
   `build_cecd_dual_semantics_preflight_v1.py` now reconstructs the dev and
   locked-confirmation claim inputs from the frozen VinDr reader manifest only
   after validating the genuine admission, successful canonical detached
   `cecd-three-stage-v3` state, locked confirmation GO, and exact v3 input
   gate. It writes the preflight, input sidecar, four input files, and a
   fingerprinted receipt with write-once equality.
2. **Authorization stopped before a runner handoff.**
   The transition monitor now executes one idempotent control-plane sequence:
   canonical build -> v2 hash binder -> write-once detached formal-CE launch
   handoff. The handoff is exact and runnable but is emitted as
   `ready_not_launched`; the monitor does not spawn a GPU process.
3. **GPU-lock namespace drift.**
   The controlled runner and shell wrapper now require the same lock used by
   all VinDr jobs:

   ```text
   /home/dbw/ANCHOR/corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock
   ```

   Passing the former `gpu0-cecd-dual-semantics-v1.lock`, or any other path,
   fails before authorization parsing, worker/runtime description, output-root
   creation, or GPU access.

## Authority chain

The canonical build receipt binds the bytes and absolute paths of:

```text
cecd-three-stage-v3 detached state
clinical admission analysis
locked confirmation analysis
three-stage-v3 input gate
preflight and input sidecar
dev calibration claim manifest
locked-confirmation evaluation claim manifest
ordered record keys
formal CE claim contract
Huatuo/Hulu model fingerprints
canonical GPU lock
```

The runtime authorization now also binds this build receipt. A manually
written preflight without the receipt cannot authorize the runner. Replaying
the builder/monitor is byte-stable; any upstream or handoff drift produces a
write-once collision.

The builder does not read factorial/model-output rows. It only accepts the
already locked two-model GO through the existing three-stage validator and
reconstructs claim identities from the reader manifest. No reviewer return is
created or modified.

## Formal handoff boundary

The emitted handoff intentionally requests `--execute-ce-only`. It covers the
seven already implemented centered-logit controls:

```text
unmitigated
full_orbit
render_only
prompt_only
random_norm
sign_permuted
main_effect_removal
```

OE, CECD hidden intervention, and the two Treble semantic variants remain
blocked. The handoff cannot be cited as full mitigation closure and has
`paper_claim_authorized=false`.

Canonical future artifacts (only after a real two-model GO) are:

```text
configs/cecd_dual_semantics_preflight_v1.json
configs/cecd_dual_semantics_preflight_v1.inputs.json
configs/cecd_dual_semantics_inputs_v1/*
corrected_runs/vindr_v2/cecd_dual_semantics_v1/preflight_build.json
corrected_runs/vindr_v2/cecd_dual_semantics_v1/authorization.json
corrected_runs/vindr_v2/cecd_dual_semantics_v1/formal_ce_launch_handoff.json
```

None of these formal artifacts exists at this audit point because the genuine
admission and three-stage job have not completed.

## CPU verification

```bash
PYTHONPATH=.:anchor .venv-full/bin/pytest -q \
  tests/test_build_cecd_dual_semantics_preflight_v1.py \
  tests/test_authorize_cecd_dual_semantics_preflight_v1.py \
  tests/test_monitor_cecd_dual_semantics_transition_v1.py \
  tests/test_run_cecd_dual_semantics_controlled_v1.py \
  tests/test_cecd_dual_semantics_scientific_worker_v1.py
```

Result after final source-only checks: `31 passed in 2.97s`.

The tests cover successful write-once construction/replay, authority drift,
noncanonical detached state, two-model NO-GO, automatic build/authorize/handoff,
handoff drift, absent authorization, canonical-lock rejection/contention,
runtime/input/source hash drift, atomic failure/recovery, and the two-model
formal CE shared-cache closure. `py_compile`, `bash -n`, and `git diff --check`
also pass for the touched files.

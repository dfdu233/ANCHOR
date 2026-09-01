# CECD dual-semantics controlled-comparison runner v1

**Date:** 2026-08-03  
**Status:** executor implemented and synthetically verified; real scientific
worker not yet admitted; no GPU or sealed outcome was opened.  
**Scope:** orchestration and provenance only. This runner neither implements an
official Treble reproduction nor computes a scientific verdict.

## Outcome

`run_cecd_dual_semantics_controlled_v1.py` closes the operational gap between
the write-once Stage-1/preflight authorization and a later blinded outcome
analysis. It has two deliberately separate invocations:

1. **prepare:** validate the existing authorization and preflight, query both
   runtimes without loading an experiment arm, bind all code/runtime/input/model
   identities, and write one immutable `run_contract.json`;
2. **execute:** re-create that contract byte-for-byte, acquire the single-GPU
   `flock`, run every frozen `model x method` arm, validate CE and aligned-OE
   outputs, atomically promote complete arms, and emit a manifest that still
   says `results_interpreted=false` and `paper_claim_authorized=false`.

The executor is generic over the method names in the preflight. It requires
exact equality among `preflight.methods`, `authorization.allowed_methods`, the
run-contract arm order and the final manifest. It never imports the current
ten-method Python constant as its execution closure.

This distinction matters now: the existing v1 preflight validator contains ten
methods, whereas the latest collision audit requires at least one compatible
dynamic or multimodal activation baseline in addition to a static steering
baseline. Therefore the current ten methods are a **strict common-protocol
closure but not the final ICLR-oral baseline closure**. The scientific contract
must be extended before any preflight, authorization or method output exists;
the runner will then execute the extended frozen list without a code change.

## Fail-closed boundary

The runner rejects before worker, runtime-descriptor or GPU access when the
authorization is absent. With an authorization present, it independently
checks:

- the authorization fingerprint and all narrow-scope true/false flags;
- byte identity of admission, Stage-1 input gate, Stage-1 analysis and
  preflight, without parsing the sealed Stage-1 outcome;
- exact two-model identity and exact ordered method closure;
- a repository-child method-output root;
- the Huatuo and Hulu Python binaries by content hash;
- worker, authorization-binder, collision-contract and runner source hashes;
- each worker runtime descriptor, including its own source closure;
- calibration manifest, locked evaluation manifest, record-key manifest and
  claim contract against the hashes frozen in the preflight;
- the full preflight model-fingerprint object for each family;
- a deterministic offline environment and one fixed GPU-lock path.

`general_gpu_authorized=false` remains true in every artifact. The only GPU
scope accepted is “this hash-bound controlled comparison” under
`cecd_hidden_state_intervention_authorized_only_inside_locked_comparison=true`.
It cannot be reused as general hidden-state or GPU authority.

## Worker protocol

The worker is intentionally separate because Treble proceedings and released
source semantics differ, Hulu needs an explicitly admitted variable-token
transport, and a future dynamic baseline must be selected before outcomes. A
worker may be used only after code review and inclusion in the frozen runtime
contract.

For `--describe-runtime --model-family FAMILY --preflight PATH`, it prints one
JSON object with schema
`cecd-dual-semantics-runtime-descriptor-v1` and exactly:

```text
schema_version
model_family
model_id
model_fingerprint
python_executable
runtime_versions
source_files[]
input_bindings{
  calibration_manifest,
  evaluation_manifest,
  record_keys,
  claim_contract
}
```

Every source/input item is `{path, sha256, bytes}`. The worker itself must occur
exactly once in `source_files`; every input hash must equal the preflight.

For an arm, it receives:

```text
--authorization PATH --preflight PATH --run-contract PATH
--model-family FAMILY --method METHOD --output-dir TEMP_DIR
```

It must write CE and OE raw outputs plus `completion.json` under `TEMP_DIR`.
The completion schema is `cecd-dual-semantics-arm-shard-v1` and contains exact
model/method/run identity, fixed task order `[ce, oe]`, hash/size/row/cluster
records for both outputs, a nonnegative heterogeneous compute ledger, worker
hash, runtime-descriptor hash and a canonical completion fingerprint. Each task
needs at least 30 clusters. The master promotes the temporary directory only
after all checks pass.

No worker is allowed to describe a prose-derived surrogate as paper-native or
official. The two Treble variants retain their different semantics and compute
ledgers.

## Crash recovery and failure behavior

- A complete arm is a validated atomic shard at
  `arms/<model>/<method>/completion.json`; a restart skips it only after
  re-hashing all of its declared outputs.
- Incomplete attempts remain under `partial/` and are never promoted or counted.
- Any nonzero worker exit, malformed output, missing CE/OE cell, hash drift or
  lock collision stops the entire run immediately.
- Failures are append-only under `failures/attempt_NNNN.json` and explicitly say
  `automatic_retry_authorized=false`.
- A subsequent run refuses while state is failed unless
  `--resume-after-failure` is supplied after log audit. Existing valid shards
  are still reused, and failure history is preserved.
- A changed worker, source, Python executable, runtime descriptor, input,
  model fingerprint, method list or GPU-lock path cannot resume an old run.

## Commands

Preparation is the default and does not acquire the GPU lock:

```bash
cd /home/dbw/ANCHOR
CECD_DUAL_WORKER=anchor/corrected_sgta/<audited_worker>.py \
  bash scripts/run_cecd_dual_semantics_controlled_v1.sh
```

Only after inspecting the immutable contract:

```bash
CECD_DUAL_WORKER=anchor/corrected_sgta/<same_audited_worker>.py \
CECD_DUAL_EXECUTE=1 \
  bash scripts/run_cecd_dual_semantics_controlled_v1.sh
```

After an audited operational failure:

```bash
CECD_DUAL_WORKER=anchor/corrected_sgta/<same_audited_worker>.py \
CECD_DUAL_EXECUTE=1 CECD_DUAL_RESUME_AFTER_FAILURE=1 \
  bash scripts/run_cecd_dual_semantics_controlled_v1.sh
```

Today these commands correctly fail before worker/GPU access because no
write-once authorization exists. They must not be run with a synthetic or
unreviewed worker merely to make the pipeline appear complete.

## Verification

```bash
cd /home/dbw/ANCHOR
bash -n scripts/run_cecd_dual_semantics_controlled_v1.sh
PYTHONPATH=.:anchor .venv-full/bin/pytest -q \
  tests/test_run_cecd_dual_semantics_controlled_v1.py \
  tests/test_authorize_cecd_dual_semantics_preflight_v1.py \
  tests/test_monitor_cecd_dual_semantics_transition_v1.py \
  tests/test_treble_collision_contract.py
```

Synthetic tests cover absent authorization, generic method-list expansion,
two-model closure, runtime/input/source drift, replay without recomputation,
worker fail-stop, audited resume and GPU-lock collision. They do not constitute
a scientific model run.

## Frozen implementation audit — 2026-08-03

The exact focused command above completed with:

```text
.............................                                            [100%]
29 passed in 1.21s
```

The three executable/test artifacts were then content-hashed:

```text
2bb59aad5bf04216a74e791fdb14c7cdd77522926002e41cfa3f5f8df04e7ff3  anchor/corrected_sgta/run_cecd_dual_semantics_controlled_v1.py
22ab54160ca3dae3fcfc04c3203bd41645bc671fce1b594c6c010eef74846e22  tests/test_run_cecd_dual_semantics_controlled_v1.py
726a8a89900fafe9abe486c1e6d3792ac09ce16c8d02e892a80462f1361b4ee3  scripts/run_cecd_dual_semantics_controlled_v1.sh
```

Finally, the formal CLI was invoked against the actual current state with the
runner file supplied as a harmless worker tripwire. The authorization file was
absent, exit status was `1`, and the terminal error was:

```text
ControlledRunError: controlled comparison is not authorized; refusing before worker/runtime/GPU access
```

GPU state was `18 MiB, 0%` immediately before and `18 MiB, 0%` immediately
after. This is direct evidence that the current unadmitted state stops before
the worker/runtime descriptor and before GPU access. No synthetic scientific
worker was added, no Stage-1 result was parsed, and no method output exists.

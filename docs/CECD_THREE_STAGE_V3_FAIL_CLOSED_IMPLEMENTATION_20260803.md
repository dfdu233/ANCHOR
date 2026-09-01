# CECD three-stage v3 fail-closed implementation

**Date:** 2026-08-03

**Status:** implementation and CPU validation complete; scientific execution is
waiting for independent human admission and GPU availability.

## Why v3 was necessary

The original Stage-1 rule required both an observed delta-AUROC of at least
the MCID (`0.03`) and a bootstrap lower confidence bound above zero. At a true
effect exactly equal to the MCID, the point-estimate condition caps
single-finding asymptotic power near `0.5`; requiring three of four findings
and two independent models makes the original conjunction intrinsically
underpowered. The outcome-blind audit therefore revoked the 160-claim pilot as
a scientific decision set.

V3 separates operational screening, model fitting, and scientific
confirmation:

| Stage | Source split | Per finding/vote bin | Claims/model | Role | Frozen selection SHA-256 |
|---|---:|---:|---:|---|---|
| `pilot_screen` | `pilot` | 10 | 160 | engineering canary; no decision | `276bac3ffe3f06e47e6377f3dcc2b5877959a9ad372cd1c5801629719051a24a` |
| `dev_fit` | `dev` | 20 | 320 | fit and serialize all transforms/predictors | `2e9b0b0c427068e017a5ce1fbc098dbe7028bfacfa3604d395aa782807e57420` |
| `confirmation_locked` | `confirmation` | 60 | 960 | apply-only locked test | `39195d0f606da9acfa1b2b2de413176496efa6e4e235c09913c417a95c6bd1e9` |

The three selections contain no shared whole-image IDs. Both models must use
the same selection within each stage.

## Implemented contract

- `run_cecd_factorial_v1.py` now enforces the exact stage/split/sample-size
  tuple and writes stage and source split into both configuration and every
  row. `--max-claims` is restricted to the pilot engineering canary.
- `analyze_clinical_equivalence_composition_defect_v1.py` fits numeric
  standardization, categorical encoding, reader scales, and logistic
  coefficients only in `dev_fit`. The serialized predictor is applied to
  confirmation without a refit path.
- The locked primary statistic is pooled across the four findings with
  image-cluster bootstrap inference. Per-finding direction and heterogeneity
  guards prevent one finding from carrying the conclusion. Both Huatuo and
  Hulu must pass before method-level authorization.
- `verify_cecd_three_stage_v3.py` independently verifies admission binding,
  exact 19-cell orbit closure, selection hashes, stage image-disjointness,
  identical scientific contracts across models, stable model identity across
  stages, invariant weight provenance across stages, next-token conformance,
  analysis input hashes, current analyzer code hash, the frozen seed/fold/
  bootstrap contract, and the immutable dev-fit binding.
- All legacy pilot-as-dev paths remain readable for historical tests but emit
  no method or hidden-state authorization. Hidden-state authorization remains
  false even after a behavioral pass.

Canonical entry points:

- `scripts/run_cecd_three_stage_v3.sh`
- `scripts/monitor_cecd_admission_pipeline.py`
- `scripts/monitor_cecd_dual_semantics_transition_v1.py`
- `anchor/corrected_sgta/authorize_cecd_dual_semantics_preflight_v1.py`

## Runtime state at validation

The active job registry contains only the v4 admission monitor, v2 transition
monitor, and v3 three-stage scientific job name. Process inspection found one
admission supervisor/child pair (`595803`/`595805`) and one transition
supervisor/child pair (`596222`/`596224`), with no live legacy CECD monitor.

The admission heartbeat is
`waiting_for_four_independent_returns`: all eight expected return/attestation
files are absent, synthesized-label and synthesized-attestation flags are
false, and the sealed mapping has not been exposed. Only this heartbeat was
read; no human return or sealed outcome was inspected. The transition monitor
is `waiting_for_two_model_stage1` and correctly sees no v3 stage state.

No CECD GPU scoring was started. GPU 0 was occupied by a pre-existing non-CECD
process during validation, so the `flock`-guarded scientific runner was left
untouched.

## Verification

The focused three-stage regression passed `59/59`. The complete CECD,
clinical-equivalence, and Treble-collision suite passed `122/122`:

```bash
PYTHONPATH=.:anchor .venv-full/bin/pytest -q \
  $(rg --files tests | rg -i 'cecd|clinical_equivalence|treble_collision' \
  | sort | tr '\n' ' ')
```

Static checks also cover Python compilation, shell syntax, and whitespace
integrity. The power audit remains outcome-blind and reports
`FIXED_THREE_STAGE_SOURCE_TRUTHFUL`; its current artifact fingerprint is
`8f2b4de04bdbc9a7c6ff452f200a6296a437b4eee22b1a10d5b86327d12495d4`.

## Scientific interpretation guardrail

V3 makes a future result interpretable; it does not make the CECD hypothesis
true. The pilot cannot select thresholds, the confirmation split cannot fit
parameters, a one-model or one-finding effect cannot authorize the method, and
a behavioral pass can authorize only a separate outcome-blind closest-work
preflight—not hidden-state intervention or an ICLR-level claim.

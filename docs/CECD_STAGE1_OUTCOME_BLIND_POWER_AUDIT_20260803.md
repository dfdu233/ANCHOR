# CECD Stage-1 outcome-blind power and split audit

Date: 2026-08-03  
Scope: prospective behavior-gate design only  
Outcome firewall: no model score, packed factorial payload, sealed Stage-1
analysis, clinician return, or admission verdict was opened  
Executable artifact:
`corrected_runs/vindr_v2/cecd_stage1_power_audit_v1/power_audit.json`

## Verdict

**The current 160-claim run is an engineering canary, not a scientific
Stage-1 decision.**  More importantly, the current per-finding rule cannot be
powered at its own minimum meaningful effect.  It requires both
`observed delta-AUROC >= 0.03` and a 95% CI lower bound above zero.  If the true
effect is exactly the frozen MCID `0.03`, the first condition is satisfied with
asymptotic probability at most `0.5`.  Consequently:

| Gate | Maximum power at true delta=0.03 |
|---|---:|
| One finding | 0.5000 |
| At least 3/4 findings in one model | 0.3125 |
| Both models, independence planning approximation | 0.0977 |

No finite sample size can raise that rule to 80% or 90% power at the MCID.
Failure of the 160 pilot must therefore never be written as a mechanism
NO-GO.

The frozen replacement should be a hierarchical gate:

1. pooled four-finding, whole-image-clustered delta-AUROC is primary;
2. at least 3/4 findings must point in the same direction, but they need not
   each be independently significant;
3. no finding may show a clinically meaningful opposite effect;
4. Huatuo and Hulu must both pass;
5. dev fits the predictor once and locked confirmation applies it without
   refitting.

## Exact three-stage data contract

The existing manifest supports the agreed chain without reading an outcome.
Selection uses the existing deterministic hash order and the four frozen
findings.

| Stage | Manifest split | N / finding / vote bin | Claims/model | Unique images | Selection SHA-256 |
|---|---|---:|---:|---:|---|
| `pilot_screen` | pilot | 10 | 160 | 154 | `276bac3f...051a24a` |
| `dev_fit` | dev | 20 | 320 | 283 | `2e9b0b0c...e57420` |
| `confirmation_locked` | confirmation | 60 | 960 | 837 | `39195d0f...c6bd1e9` |

The exact whole-image intersections are zero for all three stage pairs.  Some
images carry multiple findings within a stage (`6`, `34`, and `116` images),
so every CV split and bootstrap must continue to cluster on `image_id`, never
on image-claim.

### Provenance defect found and repaired before scoring

The current runner packs:

```text
split = dev
source_manifest_split = pilot
```

and the analyzer rejected anything whose label was not `dev`.  This aliased a
pilot OOF screen into a dev artifact and is not a harmless name.  The audit
original audit marked it `MUST_FIX_BEFORE_NEW_FORMAL_OUTPUT`.  The v3 runner
and analyzer now use the exact
labels `pilot_screen`, `dev_fit`, and `confirmation_locked`; legacy
pilot-as-dev artifacts remain canary-only.

## Frozen power model

The prospective calculation uses paired equal-variance binormal AUROC,
DeLong influence-function variance evaluated by deterministic 48-point
Gauss-Hermite quadrature, and a normal approximation to a two-sided 95% CI.
The central assumptions are:

| Quantity | Frozen value |
|---|---:|
| Minimum meaningful delta-AUROC | 0.03 |
| Planning alternative for the full gate | 0.05 |
| Baseline AUROC | 0.70 |
| Polarity-error prevalence | 0.20 |
| Paired baseline/candidate score correlation | 0.95 |
| Image-cluster design effect | 1.10 |

This is an **optimistic oracle-score calculation**: it does not add the
variance of fitting the OOF logistic predictor.  A negative result must not be
declared powered when the observed dev nuisance parameters fall outside the
planning envelope.

### Power of the available stages

The table below is joint two-model power for the recommended pooled primary
plus the 3/4 positive-direction and no-opposite-MCID guards, under the central
assumptions.

| Stage N/bin | True delta | Detection (`CI low > 0`) | Full gate (also point >=0.03) |
|---:|---:|---:|---:|
| 10 | 0.03 | 0.040 | 0.040 |
| 20 | 0.03 | 0.136 | 0.136 |
| 60 | 0.03 | 0.669 | 0.250 |
| 10 | 0.05 | 0.206 | 0.206 |
| 20 | 0.05 | 0.577 | 0.577 |
| 60 | 0.05 | 0.994 | **0.942** |

Thus `20/bin` is suitable for fit/freeze and a directional development
estimate, not a formal null decision.  The locked `60/bin` test has about 94%
power for the full gate at the predeclared planning alternative `0.05`, but it
does not have 80% detection power at the exact MCID.

### N/bin required for 80% and 90%

Under the central assumptions:

| Target | 80% | 90% |
|---|---:|---:|
| Detect true delta=0.03; no point-MCID threshold | 76/bin | 94/bin |
| Full hierarchical gate at true delta=0.05 | 29/bin | 46/bin |
| Current 3/4 individually significant gate at true delta=0.03 | impossible | impossible |

Sensitivity is material.  For the full gate at true delta `0.05`, varying
error prevalence from `0.10` to `0.30` and paired score correlation from
`0.90` to `0.99` moves the 80% requirement from `7` to `97/bin`, and the 90%
requirement from `10` to `152/bin`.  This is why `60/bin` may support a positive
confirmation, but an unqualified “powered null” claim would be false.

If the paper requires 90% **detection** power at the exact MCID, a pre-outcome
expansion to `94/bin` is feasible in the raw confirmation census for all four
selected findings/bins (the smallest relevant available cell exceeds 94), but
it requires a newly frozen manifest and selection hash.  It still does not
remove the mathematical boundary created by requiring the observed estimate
itself to exceed the true MCID.

## Frozen confirmation gate

For each model, all of the following are required on
`confirmation_locked`:

1. pooled four-finding image-cluster delta-AUROC point estimate `>= 0.03` and
   bootstrap 95% CI lower `> 0`;
2. pooled harmful-interaction error-minus-correct difference CI lower `> 0`;
3. interaction RMS `>= 0.25` reader-equivalents and CI lower `> 0`;
4. identity-render and duplicate-prompt RMS each `<= 0.1 x` clinical
   interaction RMS;
5. every finding's reader slope CI lower `> 0`;
6. at least 3/4 per-finding delta-AUROC point estimates `> 0`;
7. at least 3/4 per-finding harmful-alignment point estimates `> 0`;
8. no finding has delta-AUROC point estimate `<= -0.03` or a bootstrap 95% CI
   upper bound `< 0`.

Both Huatuo and Hulu must pass.  No finding-specific CI significance is
required.  Temperature, length, identity, full-orbit, marginal, generic
stability and behavioral PID controls remain unchanged.

## Required implementation transition

No GPU may start from this audit.  Before admitted scoring, the execution path
must be changed as follows:

- `run_cecd_factorial_v1.py`: accept and hash-bind explicit stage label,
  manifest split, and per-bin N; write truthful payload labels; reject output
  directory or selection-hash reuse.
- `analyze_clinical_equivalence_composition_defect_v1.py`: add `dev_fit` mode
  that serializes all feature transforms, reader scales, categorical schema,
  coefficients and thresholds; add apply-only `confirmation_locked` mode with
  no test refit; make pooled clustered delta-AUROC primary and per-finding
  results heterogeneity guards.
- `verify_cecd_two_model_stage1_v2.py`: bind the three distinct selections,
  claim/row counts, image-disjointness, output roots and fit artifact; reject
  pilot-as-dev inputs.
- admission monitor: run separate roots such as
  `cecd_{model}_pilot_screen_v3`, `cecd_{model}_dev_fit_v1`, and
  `cecd_{model}_confirmation_locked_v1`; a pilot scientific failure cannot
  stop the mechanism.
- dual-semantics transition monitor: consume only the two-model locked
  confirmation verdict and its exact dev-fit provenance.

The exact machine-readable implementation list and thresholds are embedded in
the audit artifact.

## Runtime and storage

Scaling the measured two-model 160-claim runtime of 45--75 minutes:

| Stage | Two-model cells | Wall time | Conservative JSON |
|---|---:|---:|---:|
| pilot 160 | 6,080 | 45--75 min | 0.033 GiB |
| dev 320 | 12,160 | 90--150 min | 0.066 GiB |
| confirmation 960 | 36,480 | 270--450 min | 0.197 GiB |
| all three | 54,720 | **6.75--11.25 h** | **0.295 GiB** |

All DICOMs are already local.  No new download is required, and storage is not
the limiting factor.

## Reproduction

```bash
cd /home/dbw/ANCHOR
PYTHONPATH=.:anchor .venv-full/bin/python -m \
  anchor.corrected_sgta.audit_cecd_stage1_power_v1
PYTHONPATH=.:anchor .venv-full/bin/pytest -q \
  tests/test_cecd_stage1_power_audit_v1.py
```

Current post-migration artifact fingerprint:
`8f2b4de04bdbc9a7c6ff452f200a6296a437b4eee22b1a10d5b86327d12495d4`.

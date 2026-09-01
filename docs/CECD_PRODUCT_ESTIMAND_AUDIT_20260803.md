# CECD product estimand outcome-blind audit

**Date:** 2026-08-03  
**Scope:** source, protocol, configuration, and synthetic-test definitions only. No sealed model outcome, human return, admission verdict, or scientific result was opened. No GPU/model execution was started. Frozen selections, data, and numerical thresholds were not changed.

## Executive verdict

The current analyzer correctly computes a balanced-factorial render-by-wording interaction that is algebraically beyond both marginal main effects. It also contains a much stronger behavioral nuisance ladder than the original implementation: marginal effects, generic grid dispersion, full-orbit summaries, and an output-distribution MMI synergy proxy are all present before the signed interaction is added.

It does **not yet identify the headline claim** “clinician-admitted equivalence-product clinical-error residual beyond both marginals, generic two-axis instability, and behavioral synergy.” Four gaps prevent that interpretation:

1. **CRITICAL — the runtime may score render cells that did not pass human admission.** The admission analyzer authorizes when only two nonbaseline renders pass, while the frozen factorial always scores all four nonbaseline renders plus baseline.
2. **CRITICAL — the independent three-stage verifier trusts asserted confirmation booleans rather than recomputing the scientific gate.** Its own positive fixture contains empty model metric blocks and is accepted.
3. **MAJOR for measurement, CRITICAL for the headline claim — the confirmation delta-AUROC is algebraically target-coupled.** Adding the signed interaction completes the exact cell score that defines `polarity_error`; the increment is therefore a useful descriptive attribution screen, but not independent evidence that a product residual predicts a separately measured clinical outcome.
4. **MAJOR — current generic/PID controls are label-invariant magnitude/distribution summaries.** They do not constitute a matched null for a signed reader-oriented product interaction, and the code explicitly acknowledges that the MMI proxy is not hidden-state or causal synergy.

Therefore the current locked result, if eventually positive, may support only:

> A sizeable reader-oriented signed render-by-wording interaction contributes to the model's own one-token polarity decisions beyond an additive reconstruction and several label-free instability summaries.

It cannot by itself support:

> Clinician-admitted product nonseparability uniquely explains independent clinical error beyond generic two-axis instability or multimodal synergy.

## 1. What the current estimand actually identifies

For one image-claim orbit, let the signed Yes-minus-No score matrix be

\[
M\in\mathbb{R}^{R\times P},\qquad
J=H_R M H_P,
\]

where `H` is a centering matrix. In elementwise form the implementation uses

\[
J_{rp}=M_{rp}-\bar M_{r\cdot}-\bar M_{\cdot p}+\bar M_{\cdot\cdot}.
\]

This is implemented exactly in `anchor/corrected_sgta/analyze_clinical_equivalence_composition_defect_v1.py:85-93` and attached to every orbit at `:412-475`. The decomposition is

\[
M_{rp}=\mu+a_r+b_p+J_{rp},
\]

with `row_main`, `col_main`, and `interaction` constructed at `:657-666`. Consequently, under a complete balanced grid, `J` is orthogonal to the intercept and both additive main-effect subspaces. The narrow answer to “is the statistic beyond both marginals?” is therefore **yes, algebraically**.

The primary magnitude statistic is

\[
\operatorname{RMS}_{\mathrm{RE}}(J)
=\sqrt{\mathbb{E}_{i,r,p}[(J_{i,r,p}/\beta_{i})^2]},
\]

where `beta` is the finding/model reader-adjacent score scale. The implementation is at `:1126-1135` for locked confirmation and `:1558-1565` for the development diagnostic. The reader scale is learned on dev and applied without confirmation refitting (`:926-1016`, `:1068-1079`). This part of the locked design is coherent.

### Exact target coupling

For clear cases define independent reader polarity

\[
s_i=+1\quad(v_i=3),\qquad s_i=-1\quad(v_i=0),
\]

and harmful orientation `h_i=-s_i`. The code defines

\[
E_{irp}=\mathbf 1[s_iM_{irp}<0]
=\mathbf 1[h_iM_{irp}>0]
\]

at `:688-706`. The strongest baseline already contains

\[
h_i\mu_i,\quad h_i a_{ir},\quad h_i b_{ip},
\]

through `full_orbit_harmful_re`, `render_main_harmful_re`, and `prompt_main_harmful_re` (`:707-725`, feature tuples at `:783-817`). The candidate then adds `h_iJ_{irp}` (`:763-765`, `:815-818`). Therefore

\[
h_iM_{irp}
=h_i\mu_i+h_ia_{ir}+h_ib_{ip}+h_iJ_{irp}
\]

is exactly available to the candidate, and the target is its zero-threshold indicator. Dev fitting and confirmation application use precisely these two extra interaction features (`:970-978`, `:1077-1087`).

This is not train/test leakage: dev/confirmation image isolation and apply-only coefficients are correctly designed. It is instead **estimand-label algebraic coupling**. The locked AUROC asks whether revealing the missing term of the score decomposition helps predict a label defined by that complete score. A positive increment can be clinically oriented because `s_i` comes from independent readers, but it is not an independent-outcome prediction result.

The current `harmful_alignment` contrast has the same limitation: it compares `h_iJ` between cells whose error status is determined by `h_iM` (`:1019-1042`, `:1084-1089`). It is a contribution/attribution diagnostic, not a causal or external clinical-error estimand.

## 2. Admission does not close the scored product orbit

The frozen runner always includes five science renders—baseline plus four nonbaseline transformations—at `anchor/corrected_sgta/run_cecd_factorial_v1.py:80-86`, and the full science grid is constructed from those names.

However, `anchor/corrected_sgta/analyze_cecd_admission_reviews_v1.py:167-176` sets `passed=True` when:

```text
len(admitted_nonbaseline_renders) >= 2
```

It does not require every nonbaseline render that the runner later scores to be admitted. `anchor/corrected_sgta/cecd_admission_gate.py:41-130` verifies provenance, roles, thresholds, and top-level authorization, but never compares `admitted_nonbaseline_renders` with the runner's exact `SCIENCE_RENDER_NAMES`. Repository-wide source search found no later exact-set check.

Thus an admissible execution trace exists in which two render families fail clinical equivalence, two pass, the top-level gate authorizes, and all four are included in `J`. In that trace `J` is not a clinician-admitted product-orbit residual.

This is a scientific validity failure, not a numerical-threshold question. The minimum repair is to require exact set equality between:

```text
admitted_nonbaseline_renders
and
SCIENCE_RENDER_NAMES minus BASELINE_VIEW
```

and exact equality between admitted prompt names and the runner's primary prompt names. An alternative is to dynamically restrict the science grid to the admitted set, but that changes frozen geometry and is therefore less safe. The exact-set failure should happen both in the admission analyzer and again in the runtime gate.

## 3. Generic two-axis instability: present as controls, not ruled out as an explanation

The strongest baseline is materially better than the older protocol. It includes:

- slice score standard deviations;
- visual/language entropy means;
- visual/language/full-orbit probability dispersion;
- cell-to-orbit KL and orbit predictive entropy;
- full-orbit harmful score;
- render/prompt main effects and RMS sensitivities;
- prompt length, fixed answer length, finding, and acquisition view.

These features are computed at `anchor/corrected_sgta/analyze_clinical_equivalence_composition_defect_v1.py:720-760` and frozen into `CLOSEST_WORK_FEATURES`/`BEHAVIORAL_PID_CONTROL_FEATURES` at `:795-817`. The candidate is genuinely nested on top of this ladder in both development OOF analysis (`:1246-1336`) and the locked dev-fit predictor (`:970-978`).

What is missing is a **matched signed product null**. Most generic features summarize magnitude, entropy, KL, or dispersion and are intentionally insensitive to which clinical polarity a cell shift harms. They can rule out “the interaction is only large because the grid is noisy,” but not “an ordinary product interaction happens to align with the reader label.” Because the candidate is `h_iJ`, this orientation is exactly the information the generic controls omit.

A minimal outcome-blind falsification is an orbit-cluster randomization that preserves:

- the complete centered interaction subspace (`row sums = column sums = 0`);
- per-orbit interaction Frobenius norm;
- additive marginals and grand mean;
- cell count and image clustering;

while destroying the alignment between `J` and reader polarity. Random sign flips of the whole `J_i` per orbit are the simplest valid null; orthogonal rotations inside the `(R-1)(P-1)` interaction subspace are a stronger matched null. The observed reader-oriented product-loss contrast must exceed that null. This tests the unique clinical orientation rather than merely adding another instability scalar to a classifier.

## 4. Behavioral synergy: useful MMI proxy, incomplete exclusion

`behavioral_pid_mmi` at `anchor/corrected_sgta/analyze_clinical_equivalence_composition_defect_v1.py:488-529` computes

\[
S_{\mathrm{MMI}}=I(R,P;Y)-\max\{I(R;Y),I(P;Y)\},
\]

plus per-cell local excess. Both scalar and local terms enter the strongest baseline (`:751-760`, `:807-817`). This correctly rules out a simple claim that CECD wins only because the output distribution has more label-free joint information.

It does not rule out behavioral synergy in general:

- the code itself states it is not representation-level PID or causal synergy (`:525-528`, `:1822-1825`);
- KL/MMI quantities are invariant to a permutation of the Yes/No/Maybe labels, whereas `h_iJ` is reader-polarity oriented;
- the current delta-AUROC can therefore improve by receiving the exact signed component of its target even when MMI has already summarized all label-free joint dependence.

The correct conclusion is “MMI-style output synergy is controlled as a nuisance feature,” not “behavioral synergy is excluded.” A matched reader-label-destroying product null, followed after behavioral GO by the already-planned representation-level/causal controls, is required for the stronger statement.

## 5. The three-stage verifier is not an independent scientific verifier

`anchor/corrected_sgta/verify_cecd_three_stage_v3.py:247-278` checks artifact versions, paths, source/input hashes, and dev binding. At `:279-290` it only checks that the asserted passing-model list, `both_models_pass`, and authorization boolean agree with each other. It does not recompute any of the following from model metrics:

- delta-AUROC point/CI threshold;
- harmful-alignment CI;
- interaction RMS/CI;
- identity-noise ratio;
- reader-slope gate;
- per-finding heterogeneity guard;
- the conjunction forming `model_confirmation_pass`.

The current positive verifier fixture makes the issue executable: `tests/test_verify_cecd_three_stage_v3.py:68-130` supplies **empty model metric dictionaries** and asserted true gate booleans; `:134-145` expects the verifier to authorize. Thus the verifier validates provenance plumbing and internal boolean consistency, not the scientific calculation.

The same artifact-trust issue applies to the dev predictor: `apply_serialized_predictor` consumes the feature list and coefficients embedded in the bundle (`anchor/corrected_sgta/analyze_clinical_equivalence_composition_defect_v1.py:889-912`), while the verifier does not assert the exact baseline/candidate feature schemas or independently regenerate the bundle.

Minimum robust repair: from the bound raw dev and confirmation JSONL inputs, independently rerun `fit_dev_stage` and `apply_confirmation_stage` in CPU mode with the frozen seed/folds/draws, canonicalize deterministic fields, and compare the recomputed artifacts before emitting authorization. A weaker field-by-field recomputation of all gate predicates is acceptable as a temporary guard, but it still trusts potentially altered metric values and predictor coefficients.

## 6. Test and documentation coverage gaps

The synthetic tests correctly establish several negative cases:

- additive grids fail (`tests/test_clinical_equivalence_composition_defect.py:220-225`);
- nonlinear interaction without new error information fails (`:228-238`);
- an interaction already captured by generic stability fails (`:415-442`).

There is currently no analyzer-generated positive fixture that passes the complete behavioral ladder and locked two-model gate. Apparent positive downstream tests insert an asserted true confirmation gate rather than deriving it from scores. In addition, `docs/CLINICAL_EQUIVALENCE_COMPOSITION_DEFECT_PROTOCOL.md:160-167` still says a “reader-oriented true interaction” is required to pass, while the corresponding current test intentionally expects that fixture to fail because generic stability explains it.

Minimum added tests should cover:

1. admission fails if even one runner science render is not admitted;
2. the runtime gate rejects exact-set mismatch even if top-level `passed=True`;
3. the verifier rejects true gate booleans with empty/failing model metrics;
4. exact algebraic reconstruction `hM = hμ + ha + hb + hJ` is documented as a guardrail;
5. a pair of synthetic generators matched on marginals, interaction RMS, output entropy, probability dispersion, and MMI synergy but differing only in reader-oriented interaction alignment;
6. a valid product-specific fixture passes the new matched-null/product-loss criterion, while its orbit-sign-randomized twin fails.

## 7. Minimal estimand revision without changing existing frozen thresholds

The current `0.25 RE` and `0.03 delta-AUROC` rules should remain unchanged and be relabeled as a **behavioral contribution screen**. They should not alone authorize the headline product-residual claim.

Add an outcome-blind, non-authorizing descriptive estimand now, using the already available score matrix and independent reader target:

\[
M^A_{irp}=\mu_i+a_{ir}+b_{ip}=M_{irp}-J_{irp}.
\]

For clear reader labels, report the cluster-bootstrapped excess 0-1 error

\[
\Delta E=
\mathbb E[\mathbf 1(s_iM_{irp}<0)-\mathbf 1(s_iM^A_{irp}<0)],
\]

and preferably a dev-calibrated proper-loss contrast

\[
\Delta L=
\mathbb E[\ell(s_i,M_{irp})-\ell(s_i,M^A_{irp})].
\]

Positive values have a direct interpretation: adding only the product interaction, while holding the grand mean and both marginals fixed, increases reader-grounded clinical loss. Negative values mean the interaction repairs errors. This removes the rhetorical dependence on “predicting” a label defined by the same complete score.

Compare `Delta L` to the orbit-sign/interaction-subspace matched null above. Do not assign a new pass threshold after outcomes are visible; power and the smallest clinically meaningful `Delta L` must be frozen before this estimand becomes authorizing. Until then it is a transparent diagnostic and the current confirmation gate remains only a screen.

For all four reader bins, a later extension can fit a dev-only calibration from signed score to present probability and compare Brier loss against the reader fraction `v/3`. That would be closer to the paper's reader-grounded claim, but it requires a separately frozen calibration/power contract and should not be silently added to the current locked gate.

## 8. Severity-ranked repair order

| Priority | Repair | Why it is minimal | Threshold/data impact |
|---|---|---|---|
| P0 | Require exact admission-set equality with every scored render and prompt, in both analyzer and runtime gate | Restores the meaning of “clinician-admitted product orbit” | No threshold or selection change; fail-closed contract correction |
| P0 | Independently recompute dev fit and confirmation gate in `verify_cecd_three_stage_v3.py` | Prevents asserted/tampered booleans or predictor bundles from authorizing | No scientific threshold change; CPU only |
| P1 | Relabel delta-AUROC as a behavioral contribution screen and add `Delta E`/`Delta L` decomposition | Removes the target-reconstruction overclaim while preserving existing results | Existing thresholds unchanged; new estimand non-authorizing until separately frozen |
| P1 | Add orbit-sign and interaction-subspace matched nulls | Distinguishes clinical orientation from generic two-axis magnitude/MMI synergy | No model/GPU calls; requires preregistered inference rule before authorization |
| P2 | Add a genuine positive synthetic identification fixture and synchronize protocol wording | Proves the complete gate can accept only its intended data-generating process | Tests/docs only |
| P2 | Keep representation-level PID and architecture controls downstream of behavioral GO | The current MMI proxy is explicitly incomplete | Already consistent with existing stage boundary |

## 9. Final claim boundary

The code already has a strong base: complete balanced grids, exact duplicates, image-grouped inference, dev-only fitting, locked application, reader-oriented scaling, and explicit non-claims around Treble and hidden-state PID. The remaining problems are not reasons to discard CECD; they are reasons to avoid mistaking a mathematically valid ANOVA residual for a fully identified clinical mechanism.

No pipeline source was modified in this audit because the current analyzer/admission/verifier hashes are bound into multiple dormant handoffs. A partial patch would invalidate those locks without closing the full chain. The safe implementation unit is one coordinated, outcome-blind version bump covering admission exact-set closure, independent verifier recomputation, tests, protocol text, and regenerated source-lock/handoff hashes before any genuine return or model output becomes eligible.

# Clinical-Equivalence Composition Defect (CECD) analysis protocol

> **2026-08-03 execution addendum — three-stage v3 supersedes the original
> behavioral execution contract below.** The original 160-claim/model sample
> is now an operational `pilot_screen` only and can neither stop the mechanism
> nor authorize follow-on work. All transforms, scales, encoders, and
> coefficients are fit once on an image-disjoint `dev_fit` split
> (20/finding/vote-bin; 320 claims/model), serialized, and applied without
> refitting to `confirmation_locked` (60/finding/vote-bin; 960 claims/model).
> The locked primary test is the pooled four-finding, image-clustered
> delta-AUROC, guarded by preregistered per-finding direction and heterogeneity
> checks and a two-model conjunction. Pilot-as-dev artifacts remain readable
> for historical compatibility but are explicitly unauthorized. The canonical
> executable and verifier are `scripts/run_cecd_three_stage_v3.sh` and
> `anchor/corrected_sgta/verify_cecd_three_stage_v3.py`.
>
> **2026-08-03 P0 integrity addendum.** Authorization now requires the human-
> admitted nonbaseline render names and candidate prompt names to exactly equal
> the frozen runner science grid. A subset such as two admitted render families
> can no longer authorize scoring of the remaining cells. Admission analysis is
> `cecd-human-admission-analysis-v2-exact-science-grid`; dev fit and locked
> confirmation use their `v2-recomputable` schemas; the only authorizing input
> gate is `cecd-three-stage-input-gate-v4-independent-recomputation`. The v4
> verifier rebuilds dev fit and confirmation from the bound raw JSONL files and
> independently reconstructs every scientific gate predicate. Legacy v1/v3
> JSON remains readable but cannot authorize model, method, listing, or GPU work.
>
> Exact-set closure proves only that every scored **axis realization** received
> the frozen render-pair or wording-pair review. Those reviews were performed
> separately: they do not establish that humans have zero render-by-wording
> interaction over a full product task. Therefore a model-specific composition
> claim still requires an outcome-blind joint human product negative control, or
> must use the narrower wording “nonseparability under independently admitted
> axes.” This scientific boundary does not weaken the exact-set P0 requirement.

**Freeze:** 2026-08-02. **Role:** dev-only behavioral screen. A pass only
authorizes a separate method-level closest-work comparison; it is not paper
evidence and cannot authorize hidden-state experiments.

## Estimand and naming guardrail

For each model, image-claim, render `r`, and proposition/speech-act-preserving
prompt `p`, the analyzer uses the FP32 signed claim score `m = Yes - No` and
computes

```text
I[r,p] = m[r,p] - mean_p m[r,p] - mean_r m[r,p] + mean_{r,p} m[r,p].
```

This is a two-way centered interaction, equivalently a discrete mixed
derivative on the admitted product orbit. It is **not** an algebraic
commutator, an `RP-PR` contrast, or an order effect. A generic cross-modal
interaction is also not a novelty claim: Treble Counterfactual VLMs
([arXiv:2503.06169](https://arxiv.org/abs/2503.06169)) already estimates and
intervenes on vision, text, and cross-modal effects. Importantly, official
Treble is not a per-claim direct-effect scalar: it learns global PCA steering
directions from a demonstration/calibration set and evaluates intervened model
outputs. The only surviving CECD claim is the conjunction of a clinically
admitted product orbit, independent reader-vote units, and incremental
prediction of medical error.

## Runner-independent JSON contract

The analyzer accepts either one normalized JSON object:

```json
{
  "schema_version": "clinical-equivalence-factorial-v1",
  "split": "dev",
  "frozen_before_outputs": true,
  "score_definition": "fp32_yes_minus_no_logit",
  "primary_renders": ["canonical", "native_linear", "center_minus", "center_plus", "width_1.25"],
  "primary_prompts": ["is_there", "does_show", "can_be_seen"],
  "baseline_render": "canonical",
  "baseline_prompt": "is_there",
  "identity_render": "canonical_identity",
  "duplicate_prompt": "is_there_duplicate",
  "records": []
}
```

Each record requires `model`, `image_id`, `finding`, `reader_votes` in `0..3`,
`render_id`, `prompt_id`, finite `signed_score`, and finite
`commitment_score`, normalized `acquisition_view`, and finite three-way
`tristate_entropy`. An optional
`crossmodal_direct_effect_scalar_surrogate` may be included only as a generic
sensitivity diagnostic. It is never called Treble, never satisfies a
closest-work baseline, and never affects authorization. The minimal cell
contract per image-claim is:

- the complete primary render × primary prompt science grid (15 cells for the
  current 5 × 3 runner);
- identity render × each of the three primary prompts (3 cells);
- baseline render × duplicate baseline prompt (1 cell).

An optional `fold_id` must be constant for every `(model, image_id)`. The
analyzer creates its own image-grouped folds and rejects any observed external
group leakage. Invalid or incomplete orbits are rejected, not imputed.

It also directly accepts the crash-safe runner's `factorial_rows.jsonl`. In
that form it maps `positive_votes` to the 0..3 target, infers the baseline from
the duplicate prompt's `reference_cell_id`, and uses `cell_role` to recover the
science and identity cells. Huatuo and Hulu files may be supplied together:

```bash
PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python -m \
  corrected_sgta.analyze_clinical_equivalence_composition_defect_v1 \
  --input /path/to/huatuo/factorial_rows.jsonl \
  --input /path/to/hulu/factorial_rows.jsonl \
  --output /path/to/cecd_dev_screen.json
```

## Reader grounding and separate outcomes

The polarity scale is the cross-fitted, model- and finding-specific median of
the three adjacent clean reader-bin mean differences. Its uncertainty is
reported using an image-cluster bootstrap. Non-positive scales are excluded
fail-closed rather than made positive by taking an absolute value.

- Vote `0/3` and `3/3` cells define clear negative and clear positive polarity
  error respectively. Interaction direction is called harmful only after
  orienting it using this independent reader polarity.
- Vote `1/3` and `2/3` cells never enter the polarity-error AUROC. They report
  an ordinal reader-support gap and a separate commitment interaction. The
  `Maybe` verbalizer is a model coordinate, never clinical truth.

## Predictive controls and gates

All predictions are out-of-fold with image-grouped CV. The engineering
baseline contains clean reader-oriented margin, signed render main effect,
signed prompt main effect, both marginal RMS sensitivities, finding, and the
full-orbit score, plus preregistered acquisition-view and three-way output
entropy controls. Categorical view/finding encoders and all numeric transforms
are fit inside each image-grouped fold; unseen held-out view categories are
ignored safely. CECD adds only harmful-oriented and absolute two-way interaction
residuals. It reports AUROC and Brier increments with image-cluster bootstrap
intervals.

Stage 1 passes per model only if all hold:

1. interaction RMS point estimate is at least `0.25` adjacent-reader
   equivalents and its CI is above zero;
2. AUROC increment over clean + both marginals + full orbit is at least `0.03`
   with CI above zero;
3. harmful-oriented interaction is larger in reader-defined polarity errors,
   with CI above zero;
4. identity-image and duplicate-prompt RMS are each below one tenth of the
   admitted clinical interaction;
5. at least three of four findings have positive AUROC increment and harmful
   alignment.

Two models passing Stage 1 authorizes only preparation of a **separate,
outcome-blind closest-work comparison preflight**. The behavioral analyzer itself always emits
`authorized_for_hidden_state_stage: false`. It cannot consume a scalar feature
as exact Treble evidence.

The exact external adapter contract remains `cecd-treble-method-collision-v1`
and remains blocked because `paper_and_code_semantics_resolved` is not a
truthful description of the public release. The non-fabricated fallback is a
two-step common-protocol envelope: the outcome-blind
`cecd-treble-dual-semantics-preflight-v1`, followed by
`cecd-treble-dual-semantics-envelope-v1`. It separately freezes and evaluates
the proceedings-faithful and released-source-faithful definitions, plus
full-orbit and all marginal/random/main-effect controls. An independent runtime
binder must first reconstruct the passed Huatuo+Hulu Stage-1 files and verify
that no method output exists. It may authorize only that locked comparison;
the post-run validator then recomputes whether CECD beats both source variants
and full-orbit averaging without claim-count, coverage, length, omission,
refusal or Brier exchange. Neither variant is called exact or paper-native.

A wide interval is fail/inconclusive. The `0.25 RE`, `0.03 AUROC`, identity,
two-model, and three-of-four thresholds are not relaxed because the 160-case
pilot is underpowered. Full-orbit averaging and both source-faithful Treble
variants remain direct controls later; failure to beat any member of that
envelope terminates CECD-specific advantage.

## Executable adversarial contract

`tests/test_clinical_equivalence_composition_defect.py` requires:

- pure additive data to fail;
- nonzero nonlinear interaction carrying no incremental error information to
  fail;
- an image split leak to be rejected;
- a reader-oriented true interaction to pass the two-model gate;
- payloads with no scalar surrogate to remain fully compatible with Stage 1;
- legacy `treble_nde_score` fields to be ignored rather than masquerade as an
  exact Treble result;
- every behavioral result to keep hidden-state authorization false pending the
  external method-level validator.

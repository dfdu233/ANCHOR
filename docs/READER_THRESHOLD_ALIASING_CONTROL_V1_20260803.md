# Reader-threshold aliasing control v1

**Date:** 2026-08-03  
**Role:** outcome-blind alternative-explanation control for Candidate B in
`POST_CEBC_MECHANISM_PIVOT_SEARCH_20260803.md`. It is not a new mainline, a
mitigation method, or authority to change the CECD gate.

## Frozen question

Conditional on vote count, finding, model, and a clean-condition model score,
does a VLM's binary positive commitment depend on the exact positive-reader
pattern of the fixed VinDr `R8/R9/R10` panel?

The formal inputs are future `dev_fit` and `confirmation_locked` clean CE
records. No existing Huatuo/Hulu outcome, sealed result, hidden state, image, or
GPU was read while implementing this control.

## Statistical contract

The baseline is deliberately saturated enough to close the obvious
confounder:

```text
vote_count
+ model × finding cell intercepts
+ clean_margin × model × finding cell slopes
```

The augmented model freezes that baseline's logits and fits only exact
`R8/R9/R10` pattern residuals, saturated by model × finding cell. Thus a gain
cannot be created by omitted model/finding-specific sensitivity or by
re-estimating baseline coefficients.

- Development predictions use deterministic whole-image group cross-fitting.
- The full development fit and reader ordering are serialized and hashed.
- Confirmation applies that fit once, with no refit, threshold selection, or
  tuning.
- The primary identity gate requires pooled **and both per-model** gains:
  ΔAUROC ≥ 0.05, relative NLL improvement ≥ 5%, and image-cluster bootstrap
  lower bounds above zero.
- For the `positive_commitment` endpoint, reader ordering is computed from
  baseline-adjusted commitment residuals and must exactly recur in at least six
  of eight findings separately for Huatuo and Hulu.
- For `clinical_error`, reader ordering is not defined. That endpoint can only
  be a predictive alternative-control and can never establish a virtual-reader
  operating point.

## The important structural negative result

Exact reader identity has no variation on unanimous `000` or `111` cases. The
frozen-baseline pattern increment is therefore exactly zero on every 0/3 and
3/3 record. The analyzer asserts this at confirmation time and records:

```text
clear_case_identity_increment_defined = false
maximum_absolute_prediction_difference = 0
```

There is no empirical clear-case bootstrap masquerading as a possible gate.
The current control can at most support **reader-disagreement semantics**. A
clear-case hallucination mechanism would require a separately preregistered
independent predictor or intervention not implemented here. Even a positive
control cannot alter CECD's primary gate or authorize mitigation.

## Fail-closed implementation

- Analyzer: `anchor/corrected_sgta/analyze_reader_threshold_aliasing_v1.py`
- Outcome-blind preflight:
  `anchor/corrected_sgta/validate_reader_threshold_aliasing_preflight_v1.py`
- Frozen preflight config:
  `configs/reader_threshold_aliasing_preflight_v1.json`
- Synthetic tests: `tests/test_reader_threshold_aliasing_control_v1.py`

The current preflight is truthfully blocked until immutable hashes exist for:

1. the future `dev_fit` input;
2. the disjoint `confirmation_locked` input;
3. the matched-count and matched-length direct-listing transfer input.

The preflight checks bindings and empty output roots without opening outcome
content. The analyzer checks exact panel membership, vote-pattern/count
agreement, clean-only CE status, endpoint closure, duplicate units, complete
model/finding/vote cells, source/fingerprint drift, and whole-image split
overlap.

## Synthetic falsification coverage

Tests cover fixed-panel reordering, vote mismatch rejection, whole-image folds,
write-once dev/confirmation semantics, fit fingerprint drift, claim-promotion
rejection, late-output rejection, clinical-error ordering exclusion, and the
structural zero identity increment on clear cases. A dedicated confounding
test varies clean-margin slopes across every model × finding cell while making
reader pattern irrelevant; the saturated baseline leaves the aliasing gain
below the preregistered margins.

Power remains a boundary: sparse pattern cells or sparse endpoint events make a
failure indeterminate, never a reason to lower thresholds after confirmation.

# CECD product-attributable clinical-risk contract v4 (outcome-blind draft)

**Status:** outcome-blind replacement candidate; no human return, sealed model
outcome, or locked confirmation output was used.  This document does not
authorize model scoring.  The v3 AUROC gate remains reproducible as a historical
diagnostic but cannot establish the paper claim below.

## 1. Why v3 cannot be the primary identification gate

For one image-claim orbit, write the signed cell score as

\[
m_{rp}=\bar m+a_r+b_p+I_{rp},
\qquad
I_{rp}=m_{rp}-\bar m_{r\cdot}-\bar m_{\cdot p}+\bar m.
\]

The v3 baseline already contains the orbit mean and both main effects.  Its
candidate adds the signed and absolute interaction.  Consequently the
candidate contains an exact reconstruction of the same cell score whose sign
defines `polarity_error`.  Image-disjoint fitting prevents ordinary leakage but
does not remove this algebraic target reconstruction.  A delta-AUROC may still
be a useful descriptive diagnostic; it is not independent evidence that the
product component predicts clinical error.

The replacement gate therefore evaluates the *paired change in reader-grounded
clinical loss caused by the product component*, rather than learning the cell
error from a decomposition of that same cell score.

## 2. Fixed objects

- Unit: one complete, clinician-admitted `(model, image, finding)` product
  orbit.  One invalid cell excludes the whole orbit.
- Factors: admitted render and wording operations only.
- Axis-wise admission is necessary but does not by itself prove that human
  behavior is additive on the full product.  Before a model-specific
  composition claim, a separately randomized, outcome-blind clinician product
  control must bound human render × wording interaction on a stratified subset.
  If that control cannot be obtained, the claim is limited to model score
  nonseparability under two individually admitted axes.  The minimal fractional
  design is specified in `docs/CECD_HUMAN_PRODUCT_CONTROL_DRAFT_20260803.md`.
- Reader target: `q=votes/3`; all four vote bins remain distinct.  Clear-case
  error is a secondary endpoint, not a substitute for reader disagreement.
- Split: pilot is operational only; probability calibration and nuisance
  matching use image-disjoint dev; confirmation is apply-only.
- Clustering: whole patient when available, otherwise whole image.  All cells
  and findings from the cluster are resampled together.

## 3. Product-attributable risk

Fit one monotone score-to-reader-support map `g_dev` per model and finding using
only the canonical dev cell.  Freeze its parameters before confirmation.
Calibration must be cross-fitted for dev diagnostics and apply-only on
confirmation.  Define

\[
p_{rp}=g_{dev}(m_{rp}),\qquad
p^{(-I)}_{rp}=g_{dev}(m_{rp}-I_{rp}).
\]

The additive counterfactual retains the orbit mean and both marginal effects;
only the centered product component is removed.  Primary paired losses are

\[
\Delta BS=(p_{rp}-q)^2-(p^{(-I)}_{rp}-q)^2,
\]

Positive values mean that the product component worsens agreement with the
reader distribution.  Brier is the sole confirmatory proper loss; the analogous
soft-label binomial NLL is a tail-sensitive sensitivity analysis with a frozen
probability floor.  Both the mean difference and the fraction of harmful cells
are reported.

Compute one loss contrast per complete orbit, then take an equal-weight macro
mean over the 16 frozen finding × reader-vote strata.  Cells are repeated
conditions, not independent clinical samples.

For vote `0/3` and `3/3`, additionally report, without redefining truth:

- product-introduced error: actual margin is wrong and additive margin is not;
- product-repaired error: actual margin is correct and additive margin is not;
- net introduced-error risk: introduced minus repaired;
- polarity flip rate and reader-oriented margin loss.

No learned predictor of `polarity_error` is a primary gate.

## 4. Product specificity controls

The exact product-risk contrast already conditions on both marginals.  It is
geometry-adjusted by one outcome-blind reference and challenged by three
additional sensitivities frozen on dev:

1. **Isospectral orientation reference:** express the centered interaction in
   centered render/prompt bases and apply Haar left/right rotations.  This
   preserves zero row/column sums and the complete singular-value spectrum.
2. **Cell-coordinate stress null:** permute the interaction cells within an
   orbit, two-way-center the permuted matrix again, then rescale it to the
   original Frobenius norm.  This preserves the interaction constraints and
   magnitude while breaking the original render × wording localization.
3. **Matched-orbit conditional sensitivity:** exchange complete interaction
   matrices within `(model, finding, reader-vote, clean-margin,
   interaction-RMS)` strata.  The matching bins and fallback rule are
   dev-frozen.  This is the only null that retains the empirical meaning of
   every render and wording coordinate.
4. **Sign-orientation stress null:** multiply a complete centered interaction
   matrix by a Rademacher sign at the image-claim level.  Never flip individual
   cells.

These nulls preserve interaction magnitude or generic grid instability but
break instance/cell-specific clinical orientation.  Define the descriptive
product-aligned excess loss

\[
\operatorname{PAEL}_{Haar}=\Delta BS_{observed}-
\mathbb E_{U,V}\Delta BS_{U I V^\top}.
\]

Haar defines the primary deterministic geometry-adjusted reference, but it is
not an exact randomization law: factor levels have fixed clinical meaning and
group exchangeability is not assumed.  Sampling inference for Brier PAEL comes
from a whole-image cluster bootstrap with common weights across both models.
The other three transformations are sensitivities; matched-orbit exchange may
report a conditional-randomization p-value only if every frozen whole-image
stratum satisfies prespecified overlap and exchangeability diagnostics.
Otherwise all reference percentiles remain descriptive.  All seeds are frozen
before confirmation.  Use at least 4,096 antithetic Haar draws per orbit and a
frozen doubling rule until numerical Monte Carlo SE is below 10% of the
image-cluster sampling SE.

Behavioral MMI/PID-style synergy, entropy, probability dispersion, prompt
length, acquisition view, finding, and both marginal RMS values remain reported
as stratification/sensitivity variables.  They cannot replace the geometry
reference and sensitivity family, and CECD cannot claim a new generic synergy
measure.

Two collision controls are mandatory before PAEL can authorize a layerwise
mechanism test:

- a MetaRA/composite-metamorphic-relation baseline that measures joint failure
  while retaining the two explicit single-axis cells; and
- a dev-frozen semantic-boundary-proximity predictor in the spirit of Semantic
  Robustness Certification, without reader votes.

If those generic controls absorb held-out PAEL, the result is ordinary
compositional metamorphic fragility near a decision boundary. PAEL remains a
descriptive clinical readout but cannot authorize a fusion-orientation claim.
PAEL itself, factorial centering, Haar rotation, and orbit-averaged proper loss
are not method or metric novelty.

## 5. Calibration and aliasing safeguards

- Report calibration slope/intercept, Brier, NLL, and reliability by reader-vote
  bin on dev and confirmation without refitting.
- Repeat the risk contrast using the empirical vote fraction directly and with
  leave-one-reader-out targets wherever reader identity is available.
- A majority-threshold-only effect that disappears for reader-distribution
  Brier/NLL is reader-threshold aliasing and fails the mechanism claim.
- Non-positive or non-monotone dev score-to-reader relation is a directional
  admission failure for that model/finding; it is never repaired with an
  absolute value.

## 6. Decision semantics after independent power red-team

Let `B0` be the same 16-stratum macro additive-reference Brier and report the
ratio of macro means `R=PAEL_Haar/B0`.  The outcome-blind operational MCID is
`R >= 0.05`; it is a project threshold, not a clinician-elicited clinical MCID.
A model cannot pass unless all are true:

1. Brier `R >= 0.05` and its whole-image cluster-bootstrap 95% lower bound is
   above zero;
2. net introduced-minus-repaired error is positive on clear cases;
3. direction holds in at least three of four findings and no finding is at or
   below the same-scale `-0.05` meaningful-opposite boundary;
4. both Huatuo and Hulu pass independently under shared image-bootstrap draws;
5. identity-render and duplicate-wording controls remain below the frozen noise
   ceiling.

NLL, matched-orbit exchange, cell permutation, sign reversal, reader-threshold
and leave-one-reader-out analyses are mandatory reported sensitivities, not
additional significance gates.  A qualitative reversal is a construct warning
that must be explained; it is never hidden by the primary pass.

Failure terminates CECD as a positive paper framing.  Passing authorizes only
the already-preregistered collision and mechanism-discrimination controls; it
does not by itself establish a decoder, mitigation, new-problem, or new-metric
contribution. A later mechanism result additionally requires a dev-selected
fusion-to-decoder orientation jump and an upstream spectrum/norm/marginal-
preserving intervention that lowers PAEL by at least 20% relative to matched
random/ispectral controls, with no more than 1 percentage point clear-case loss
in both models.

The current 60-per-bin confirmation can establish a sufficiently large
positive result.  Unless dev-frozen influence variance demonstrates adequate
joint two-model power, failure means “CECD not established,” not “effect proven
absent at the 5% boundary.”

## 7. Required adversarial tests before implementation is authoritative

- Exact reconstruction of `polarity_error` from v3 decomposition is detected
  and v3 is marked diagnostic-only.
- Pure additive grids have zero attributable product risk.
- Large generic interactions with random orientation have near-zero Haar PAEL;
  a reader-loss-localized orientation is distinguishable at matched spectrum.
- A product component that repairs as many errors as it introduces fails net
  clinical harm even if interaction RMS is large.
- Reader-threshold effects that do not worsen soft reader-distribution scores
  fail.
- Dev/confirmation overlap, refitting, cellwise deletion, null-seed changes, or
  incomplete matching fail closed.

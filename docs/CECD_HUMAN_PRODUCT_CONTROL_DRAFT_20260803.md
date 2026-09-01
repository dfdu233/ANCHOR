# CECD clinician product-control (outcome-blind authorizing draft)

**Purpose:** distinguish a model-specific render x wording composition defect
from a product interaction that is also present in human interpretation. This
control is frozen without clinician returns or model outcomes. It does not
change any existing model-scoring, model-effect, or mechanism gate; it changes
only whether a passing model result may be described as **model-specific**.

Axis-wise admission remains necessary but is not sufficient. Separate evidence
that `render` and `wording` preserve clinical meaning does not imply that human
interpretation is additive after they are composed. Without this joint control,
the maximum claim is “model-score nonseparability under two individually
clinician-admitted axes.”

## 1. Preferred recall-free design

Select **240 unique image-claims from 240 unique images**, stratified over the
four frozen findings and four reader-vote bins. An image may contribute only
one claim to this control. The selection, exclusions, factor assignments, and
reader allocation are frozen before any return is opened.

Assign each claim to one nonbaseline render and one nonbaseline wording. The
joint assignment is balanced over all `4 x 2 = 8` variant pairs, not merely over
the render and wording marginals, with an orthogonal near-balance over finding
and reader-vote bin. No post-return reassignment is permitted. Each claim then
has a genuine four-cell product:

```text
baseline render x baseline wording   (00)
variant render  x baseline wording   (10)
baseline render x variant wording    (01)
variant render  x variant wording    (11)
```

Use a **fixed panel of four independent clinicians**. For every claim, assign
the four cells one-to-one to the four clinicians with a frozen Latin-balanced
schedule. Across claims, every clinician sees each cell, render family,
wording family, finding, and vote bin equally often. Each clinician sees a
given claim exactly once; clinicians never see four sessions of the same
recognizable image. Image IDs, reader votes, factor names, baseline identity,
other clinicians' answers, and all model outputs remain hidden.

This produces 960 primary decisions in total, 240 per clinician. The four
cells remain within the same claim, but reader allocation removes the dominant
within-reader recall and consistency-anchoring pathway. Clinician identity is
part of the frozen design and is adjusted as a fixed effect; it is not treated
as an independently sampled population.

`n=160` unique images is the minimum only if a fully outcome-blind variance
basis, frozen before returns, demonstrates at least 80% power for every intended
human upper-bound gate. Otherwise use 240. Cells and clinicians do not increase
the clinical cluster count beyond the number of unique images.

## 2. Frozen response schema

For every displayed cell, record in this order:

- probability that the finding is present on a frozen numerical scale
  (`0`--`100`, converted to `[0,1]` without post-return remapping);
- `supported / refuted / undetermined`;
- visibility `clear / limited / not_assessable`;
- confidence on a fixed ordinal scale;
- optional free-text reason, never used as automatic truth.

The direct probability is required for the proper-loss estimand. The
three-state decision and confidence are secondary clinical outcomes; they are
not mapped post hoc into probabilities. `not_assessable` is retained as a
separate joint-harm endpoint and never silently deleted from the primary
analysis. A frozen conservative sensitivity must cover its probability loss.

Let `q_i` be the original three-reader support fraction. It is a reader-
distribution target, not disease truth. Therefore all resulting loss claims
are phrased as reader-distribution agreement, with clear-vote-bin and
leave-one-reader-out analyses reported as sensitivities.

## 3. Human product estimands

For clinician probability `h_ijrp` on claim `i`, clinician `j`, and cell
`(r,p)`, define the reader-distribution Brier loss

\[
Y_{ijrp}=(h_{ijrp}-q_i)^2.
\]

Estimate the clinician-adjusted product effect using the frozen Latin
allocation:

\[
Y_{ijrp}=\alpha_i+\gamma_j+\beta_R r+\beta_W p+\theta_H rp+\epsilon_{ijrp},
\]

where `alpha_i` is a claim effect and `gamma_j` is a fixed clinician effect.
The equivalent clinician-adjusted within-claim contrast is

\[
D_i^H=L_{i,11}-L_{i,10}-L_{i,01}+L_{i,00}.
\]

Whole-image bootstrap resamples all four cells and both model records with
common weights. Clinicians are never resampled. Report clinician-specific
interaction sensitivities; a qualitative clinician sign reversal is a
construct warning, not evidence of reader-population heterogeneity.

The human gate is conjunctive:

1. **Signed product harm:** the one-sided 95% upper bound for `theta_H` must be
   below a prospectively justified human acceptability margin `delta_H`.
2. **Non-cancelling product instability:** the upper bound for the mean
   absolute adjusted interaction `E|D_i^H|`, and the upper bound for the
   product-exclusive adverse-transition rate on clear `0/3` and `3/3` claims,
   must each remain below separately frozen ceilings. Harmful and repairing
   claims therefore cannot cancel into a pass.
3. **Assessability:** joint excess `not_assessable` and `limited` rates must
   remain below frozen ceilings. Factor-dependent missingness makes the human
   control fail or become inconclusive.
4. **Model-over-panel excess:** for each model separately, the lower 95% bound
   of `Delta_m=theta_m-theta_H` on the exact same sampled 2 x 2 cells must
   exceed a frozen minimum excess. Shared image-bootstrap weights preserve the
   paired comparison.

The model's full-grid isospectral-orientation PAEL and this human fractional
contrast are not the same numerical estimand. The fixed full-grid model gate is
unchanged. For the model-versus-panel statement only, recompute a plain model
2 x 2 Brier contrast on the exact human-control cells; do not relabel it PAEL.

## 4. Power and margins

`delta_H`, the absolute-interaction ceiling, the adverse-transition ceiling,
the assessability ceiling, and the model-over-panel minimum excess must be
clinically/statistically justified and frozen before returns are opened. The
model project's `R >= 0.05` threshold is not automatically a human
noninferiority margin.

For planning only, if a standardized human contrast has true mean zero,
cluster SD `sigma`, a one-sided 5% upper-bound test, and `delta_H=0.05`, then

\[
\operatorname{power}\approx
\Phi\left(\sqrt{n}\,\delta_H/\sigma-z_{0.95}\right).
\]

| Unique image clusters | SD=.15 | SD=.25 | SD=.30 | SD=.50 |
|---:|---:|---:|---:|---:|
| 60 | 0.826 | 0.462 | 0.362 | 0.192 |
| 160 | 0.995 | 0.812 | 0.678 | 0.352 |
| 240 | ~1.000 | 0.927 | 0.826 | 0.462 |

Thus 160 is admissible only with a credible SD near or below 0.25 and adequate
power for the additional non-cancelling gates. The preferred 240 supports SD
near 0.30 for the signed upper-bound gate. If plausible variance is larger,
increase unique images prospectively or report “human product harm not bounded”;
never convert low power into evidence that clinicians are invariant.

## 5. Two-clinician fallback

If only two clinicians and approximately 480 total decisions are feasible,
use 240 unique image-claims in a stratified randomized **between-image 2 x 2**.
Assign each claim to one of the four cells; both clinicians independently judge
that single cell once. This preserves 240 image clusters and removes recall,
but different images occupy different cells.

Analyze cell means under the frozen stratified randomization with fixed
clinician effects and whole-image inference. Do not compute or imply an
instance-level human mixed derivative. A passing fallback supports only:

> the model interaction exceeds the fixed-panel average interaction under
> randomized cell assignment over the sampled claims.

It does not authorize “humans compose the two operations for the same claim”
or an instance-level product-orbit defect.

## 6. Decision and claim boundary

- Clinically meaningful positive human product harm blocks “model-specific
  clinical composition defect.”
- Failure to place the human effect below its upper-bound margins is
  inconclusive, never a pass and never proof that humans share the defect.
- Substantial fixed-panel disagreement, clinician sign reversal,
  factor-dependent assessability, allocation drift, or incomplete image
  clustering is inconclusive.
- If axis-wise admission passes but the preferred control is absent, fails, or
  is underpowered, the maximum claim is “model-score nonseparability under two
  individually clinician-admitted axes.”
- A passing preferred control authorizes only a model excess relative to this
  fixed four-clinician panel over the frozen VinDr sample and assigned factor
  distribution. It does not generalize to a clinician population, every one of
  the eight factor pairs, the full human 5 x 3 surface, a decoder mechanism,
  mitigation, unrestricted OE, reports, or medical VLM hallucination broadly.

The full 5 x 3 human grid is not the minimal control: at 60 claims it requires
900 decisions per clinician while retaining at most 60 clusters and exposing
each clinician to the same claim 15 times. It may be used only as a distributed-
reader mapping study after the fractional gate, never as a repeated-reader
shortcut to stronger inference.

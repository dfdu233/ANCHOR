# CECD v4 PAEL outcome-blind power / MCID audit

**Date:** 2026-08-03  
**Scope:** proposed `CECD_PRODUCT_ATTRIBUTABLE_RISK_V4_DRAFT`, especially the
proposed product-attributable excess loss (PAEL).  This audit used only the
frozen design (`dev=20`, `confirmation=60` per finding and vote bin; four
findings; two models; 15 science cells/orbit; whole-image clustering) and
generic prospective variance ranges.  It did not open human returns, model
outcomes, GPU state, or sealed confirmation artifacts and made no source-code
change.

## Executive verdict

The narrower statistic is useful, but only under a narrower name and inference
claim:

> **KEEP as the single primary statistic:** image-cluster-macro-averaged
> **isospectral-orientation excess Brier loss**.  It measures how much more
> reader-distribution Brier loss the observed centered interaction produces
> than a frozen Haar left/right rotation reference with the same row/column
> centering and singular values.

It is an operational, geometry-adjusted estimand.  It is **not** a causal
product-attributable effect and its Haar distribution is **not an exact
conditional randomization distribution**.  Render and prompt levels have fixed
clinical semantics, so neither axis is exchangeable under arbitrary orthogonal
mixing.  The Haar expectation can define a deterministic stress reference;
sampling inference must come from a whole-image cluster bootstrap.

Use reader-distribution **Brier as the only confirmatory proper loss**.  NLL is
a valuable tail-sensitive corroboration but should not need a positive CI to
authorize the result.  Cell permutation, matched-orbit exchange, and sign
orientation should also be predeclared sensitivities rather than three further
confirmatory tests.  Requiring positive Brier and NLL CIs plus formal passage
of all three nulls is a mathematically coherent intersection-union test (IUT),
so Bonferroni is not required, but it is severely overconstrained and several
of its alleged randomization p-values would not be exact.

The existing `60/bin` confirmation has 960 orbits/model but only 837 unique
images.  It can support a sufficiently large positive PAEL.  It cannot support
a general powered-negative claim at the exact MCID, especially when both models
must pass.  The 15 cells and thousands of Haar draws are repeated measurements,
not additional clinical sample units.

## 1. Primary estimand

For orbit (i), let the admitted 5-render by 3-prompt signed-score matrix be
(M_i=A_i+J_i), where (A_i) contains the grand mean and both marginal effects
and (J_i=H_5M_iH_3).  The dev-fitted monotone calibration for the orbit's model
and finding is fixed as (g_{mf}), and the independent reader target is
(q_i\in\{0,1/3,2/3,1\}).

For a proper loss ℓ, first average over the 15 named science cells:

\[
D_i^{\ell}(J_i)=\frac1{15}\sum_{r,p}
\left[\ell\{g_{mf}(A_{i,rp}+J_{i,rp}),q_i\}
-\ell\{g_{mf}(A_{i,rp}),q_i\}\right].
\]

Draw (U\) Haar-uniformly on the four-dimensional centered render subspace and
(V\) Haar-uniformly on the two-dimensional centered prompt subspace.  Embedded
back into the original axes, (UJ_iV^\top) has zero row/column sums and the
same singular values as (J_i).  Define

\[
\operatorname{PAEL}_i^{\ell}
=D_i^{\ell}(J_i)-
\mathbb E_{U,V}\left[D_i^{\ell}(UJ_iV^\top)\right].
\]

The model-level target should be the equal-stratum macro estimand

\[
\theta_m^{BS}=\frac1{16}\sum_{f=1}^{4}\sum_{v=0}^{3}
\mathbb E\left[\operatorname{PAEL}_{i}^{BS}\mid f_i=f,v_i=v\right].
\]

This prevents a finding or vote bin with more usable clusters from silently
changing the target.  The denominator for any relative presentation must use
the same macro weighting:

\[
B_{0m}=\frac1{16}\sum_{f,v}\mathbb E
\left[\frac1{15}\sum_{r,p}BS\{g_{mf}(A_{i,rp}),q_i\}\mid f,v\right],
\qquad R_m=\theta_m^{BS}/B_{0m}.
\]

Positive PAEL means the observed orientation is more harmful than its
isospectral rotation reference.  It does **not** mean every interaction is
harmful, that the product was randomized, or that removing (J) is a realizable
clinical intervention.

### Why Brier is primary

- It is strictly proper for the fractional reader target and bounded, so a few
  nearly 0/1 calibrated probabilities cannot dominate the result.
- Its scale supports an interpretable relative MCID against additive-reference
  Brier.
- It directly rules out a majority-threshold-only artifact; a second proper
  score is not logically necessary to establish that point.
- Soft-label binomial NLL is unbounded and highly sensitive to the fixed
  probability floor, calibration tail behavior, and small vote-bin samples.

NLL should be fully reported using a predeclared probability floor and the same
macro/cluster procedure.  A negative NLL point estimate is a warning requiring
explanation; a second independently positive 95% CI should not be an
authorization condition.

## 2. Haar rotation is a stress reference, not exact randomization

The Haar construction is attractive because it preserves exactly the centered
interaction constraints, Frobenius norm, rank, and singular values.  It
therefore asks a precise geometric question: is the named clinical orientation
more harmful than generic orientations with the same isospectral interaction
geometry?

It does not satisfy the exchangeability needed for an exact randomization test:

1. the five render levels and three prompt levels are named interventions with
   fixed, clinically different semantics;
2. an orthogonal mixture of render or prompt contrasts is generally not an
   observable or clinician-admitted factor level;
3. rotation preserves singular values but not cellwise score distribution,
   named-coordinate effects, or plausibility under the image/model data
   generator;
4. after the nonlinear map (g_{mf}), an isospectral score interaction need not
   produce an exchangeable probability/loss interaction.

Thus a Monte Carlo percentile under Haar is a **reference percentile**, not an
exact design-based p-value.  This does not invalidate PAEL as a prespecified
algorithmic estimand: integrate over the frozen reference distribution, then
bootstrap clinical clusters.  It only limits its interpretation.

### Matched-orbit exchange

Exchanging complete (J) matrices among orbits within dev-frozen
`(model, finding, vote bin, acquisition view, clean/additive score, interaction
scale)` blocks is semantically better because every matrix retains the original
render/prompt cell coordinates.  Nevertheless, it is exact conditional
randomization inference only under the strong null that complete matrices are
exchangeable across image-claim orbits conditional on those blocks.

That assumption is not guaranteed: (J) is deterministically generated from
the image and model, and unmeasured image content can affect both (A) and
(J).  Multi-finding images make orbitwise permutation still less exact because
the independent unit is the whole image.  Exact exchange would require swapping
whole image bundles with identical finding/vote patterns and covariates, which
will generally leave sparse or singleton blocks.  Estimated propensity or
nearest-neighbor exchange is an approximate model-based falsification, not an
exact CRT.

Consequently:

- use Haar integration to define the primary geometry-adjusted estimand;
- use whole-matrix matched-orbit exchange as the strongest semantics-preserving
  sensitivity;
- retain cell-coordinate permutation and complete-orbit sign reversal as
  additional stress nulls;
- do not label any of the three as exact randomization inference unless a
  prospective cluster-level assignment/exchangeability mechanism is supplied.

## 3. Sampling inference and cluster structure

The analysis unit is not a cell.  Compute one PAEL per complete orbit, then
estimate the 16-stratum macro mean.  Resample or multiplier-weight the whole
`image_id` cluster, carrying all cells, findings, and both models together.
Common bootstrap weights across models preserve their paired dependence.

Because a cluster can contribute to more than one finding/vote stratum, a
whole-image cluster multiplier bootstrap is cleaner than independently
resampling each stratum.  Each draw should recompute every stratum numerator and
denominator and then the macro mean/ratio.  A two-sided 95% interval (equivalent
to a conservative 2.5% one-sided directional test) is acceptable and consistent
with the existing protocol.  Report the number of unique clusters and the
maximum cluster contribution in every analysis.

The frozen confirmation counts imply:

- 960 image-claim orbits/model (`4 findings x 4 bins x 60`);
- 837 unique images in the current selection;
- 14,400 science cells/model, but these are repeated conditions;
- effective clinical N no greater than 837 and plausibly around 700--837 after
  multi-membership/weight imbalance, to be measured from the locked design;
- Haar draws only reduce numerical integration error and never increase N.

Use antithetic Haar draws and a manifest-keyed seed.  A reasonable pre-outcome
numerical rule is at least 4,096 total rotations/orbit and an independent-stream
audit requiring aggregate Monte Carlo SE below 10% of the image-cluster sampling
SE.  If not met, increase only the numerical draws under a frozen doubling rule;
the scientific data and decision thresholds remain unchanged.

Calibration is fit on only 80 canonical dev orbits per model/finding
(`20 x 4 bins`).  Confirmation inference may condition on the frozen dev map,
which is standard apply-only evaluation.  It must not refit or choose the
calibration family from confirmation.  A nonpositive slope remains a
directional-admission failure.  The limited dev calibration size is another
reason NLL should not be a conjunctive primary endpoint.

## 4. MCID and prospective power

### Recommended MCID

The cleanest pre-outcome threshold is the project's already motivated reader
calibration scale:

\[
\textbf{operational MCID: }R_m=\theta_m^{BS}/B_{0m}\ge 0.05.
\]

That is, the clinically named interaction orientation must add Brier loss equal
to at least 5% of the additive-reference Brier beyond the isospectral reference.
Use ratio-of-macro-means, not a mean of unstable per-orbit ratios.  Also report
the absolute θ; a planning sensitivity of absolute Brier `0.005` is reasonable
when (B_0\approx0.10).

This `5%` is defensible as a **project-level operational MCID** because it is
consistent with the previously declared 5% reader-distribution Brier
improvement target.  It is not yet an independently elicited clinical MCID and
must not be called one unless clinicians/statisticians justify that scale before
opening confirmation.  The threshold must not be estimated from dev effect
sizes.  Dev may estimate nuisance variance for sample-size adequacy after the
estimand and 5% threshold are frozen.

### Generic variance simulation

No outcome was fitted.  The table uses a conservative effective cluster count
of 700, a normal approximation to the cluster-macro estimator, a two-sided 95%
CI, and the gate `point >= 0.05 and CI lower > 0`.  `SD` is the unknown
cluster-level SD of the normalized PAEL influence value.  The broad range
0.30--1.20 intentionally spans stable to highly heterogeneous losses.

| Normalized cluster SD | SE at N_eff=700 | Power/model if true R=.05 | true R=.075 | true R=.10 |
|---:|---:|---:|---:|---:|
| 0.30 | 0.0113 | 0.500 | 0.986 | ~1.000 |
| 0.50 | 0.0189 | 0.500 | 0.907 | 0.996 |
| 0.80 | 0.0302 | 0.380 | 0.699 | 0.911 |
| 1.20 | 0.0454 | 0.196 | 0.380 | 0.597 |

If model tests were independent, requiring both models would square these
numbers: at true `R=.075`, joint power ranges from `0.973` to `0.144`; at true
`R=.10`, from approximately `1.000` to `0.356`.  Positive cross-model
correlation on the shared images raises conjunctive power, but it must not be
assumed without measurement.  These calculations are illustrative envelopes,
not a post-hoc power claim.

Two mathematical consequences are non-negotiable:

1. if the true effect equals the point-estimate MCID, the point threshold alone
   caps per-model power at 0.5; no sample-size increase fixes that boundary;
2. `60/bin` may verify a large positive effect, but a failure is not a powered
   absence result unless the dev-frozen nuisance variance shows adequate power
   under a prospectively frozen planning alternative above the MCID.

For planning, use an alternative of at least `R=.075` (1.5 x MCID), report power
over the full SD envelope, and require at least 80% **joint two-model** power for
any claim that a negative result is informative.  If this is not met, retain the
existing N and interpret NO-GO as failure to establish CECD, not proof of no
effect.  Do not lower the MCID or relax the both-model rule after dev.

## 5. Multiplicity and conjunction

If a claim is defined as “both models, both proper scores, and all three null
families pass,” then requiring all component tests to reject is an IUT.  Under
the union null, testing each necessary component at alpha controls the overall
type-I error without Bonferroni.  Statistical coherence is therefore not the
problem.

The problems are construct validity and power:

- Brier and NLL test nearly the same reader-distribution ordering, but NLL adds
  calibration-tail variance rather than a distinct necessary mechanism fact;
- the three null transformations are not all valid exact randomization laws;
- even if five within-model components each had 80% power and were independent,
  all five would pass only with probability `0.8^5=0.328`; requiring both models
  would yield `0.8^10=0.107`;
- correlations make the exact number different but do not justify designing a
  ten-component authorizing gate with unknown joint power.

Freeze this hierarchy instead:

1. **Single confirmatory endpoint per model:** Brier PAEL ratio, point `>=5%`
   and image-cluster 95% CI lower `>0`.
2. **Replication IUT:** Huatuo and Hulu both pass.  No Bonferroni across models
   is needed because both are necessary.
3. **Heterogeneity guards:** at least 3/4 finding point estimates positive; no
   finding point estimate at or below its same-scale `-5%` meaningful-opposite
   boundary.  Do not require four finding-specific significant CIs.
4. **Construct guards:** clear-case net introduced-minus-repaired error is
   positive, and identity/duplicate controls stay below the frozen noise
   ceiling.  These are directional/engineering guards, not extra efficacy
   significance tests.
5. **Sensitivity family:** NLL, matched-orbit exchange, cell permutation, sign
   reversal, reader-threshold/leave-one-reader-out analyses.  Report estimates,
   CIs/reference percentiles, and any qualitative reversal transparently; do
   not multiply them into the primary pass probability.

If the authors insist on all-null passage as part of the definition, combine
the three one-sided reference p-values as `max(p_cell,p_match,p_sign)` and call
it an IUT sensitivity gate; never describe it as multiplicity-adjusted exact
randomization unless the exchangeability assumptions are actually established.

## 6. What can be frozen now

The following are outcome-blind and ready to hash-bind:

- Brier PAEL formula, 15-cell within-orbit averaging, and 16-stratum equal
  macro weighting;
- (5\times3) centered interaction and Haar left/right isospectral reference;
- the limited interpretation “isospectral-orientation excess,” not causal
  attribution or exact CRT;
- image-level cluster multiplier/bootstrap with common weights across findings
  and models;
- Brier as sole confirmatory loss; NLL and the three alternative nulls as
  sensitivities;
- ratio-of-macro-means presentation and the prospective 5% operational MCID;
- point-MCID plus CI-above-zero decision geometry, both-model IUT, 3/4
  directional guard, and no meaningful opposite finding;
- numerical Haar seed/draw/convergence rule and probability clipping for NLL;
- the rule that 15 cells and Haar replicates do not increase clinical N;
- the rule that a confirmation failure is not a powered null unless the frozen
  variance envelope establishes the declared joint power.

## 7. What cannot honestly be frozen or claimed yet

- A **clinical** MCID cannot be inferred from dev/model outcomes; 5% remains an
  operational threshold until independently justified.
- Exact power cannot be stated before observing dev-only nuisance variance,
  cluster influence dispersion, calibration slope, and cross-model dependence.
- Haar, sign, or cell permutations cannot become exact randomization tests by
  increasing Monte Carlo draws.
- Matched-orbit exchange cannot be called an exact CRT without a credible
  whole-image conditional assignment/exchangeability model and adequate block
  support.
- A negative result at `60/bin` cannot generally establish absence at the MCID.
- PAEL cannot authorize a causal removal/intervention claim; later selective
  activation or path interventions would still be required.

## Final recommendation for the v4 draft

Replace the current six-part primary conjunction with one narrow gate:

> For each model, the frozen 16-stratum macro **Brier PAEL relative to additive
> reference Brier is at least 5%, with a whole-image cluster-bootstrap 95% lower
> bound above zero; both models pass, at least three findings point positive,
> and none crosses the -5% meaningful-opposite boundary.**

Keep NLL, sign, cell-coordinate, and matched-orbit analyses as adversarial
sensitivity evidence.  This is stricter where it matters—independent reader
loss, clinical clustering, two-model replication—and removes pseudo-rigor from
an unpowered conjunction of redundant endpoints and non-exact randomization
tests.

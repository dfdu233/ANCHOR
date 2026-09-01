# CECD human product-control: outcome-blind red-team

**Date:** 2026-08-03  
**Audited object:** `docs/CECD_HUMAN_PRODUCT_CONTROL_DRAFT_20260803.md`  
**Scope:** design logic, estimand, recall/session bias, fixed-panel inference,
workload, and prospective noninferiority power only. No clinician return, model
outcome, GPU artifact, or sealed confirmation output was opened. The source
draft was not modified.

## Executive verdict

**A separate human joint-product control is logically necessary** for the
phrase **“model-specific clinical composition defect.”** Separate admission of
the render axis and wording axis does not imply that clinicians are additive,
or even stable, after the two operations are composed. A render can change the
visibility of evidence relevant to one wording without changing the answer to
a broader wording; conversely, a wording can change which feature a reader
inspects only under one render. This is exactly an interaction, and neither
marginal review observes it.

The current fractional 2 x 2 idea is the right minimum *factorial object*, but
the proposed two-clinician, four-session implementation is **not yet an
authorizing control**:

1. the same recognizable image-claim is shown four times to each clinician;
   recall and consistency anchoring can artificially drive the human
   interaction toward zero;
2. 60 claims, and potentially fewer than 60 unique images, are usually too few
   to establish a 5-percentage-point equivalence/noninferiority ceiling;
3. a signed mean `D^H` permits harmful and repairing interactions to cancel;
4. an ordinal three-state response does not define a proper loss against
   `q=votes/3` until its numerical mapping is frozen;
5. one randomly assigned render-wording pair per claim identifies an average
   over that assignment distribution, not safety of all eight nonbaseline
   render-wording pairs or the complete 5 x 3 grid;
6. a fixed panel supports inference about that panel over images, not about a
   population of clinicians.

**Recommended authorizing design:** a recall-free, within-claim fractional 2 x
2 evaluated by a fixed four-clinician panel, with one cell of each claim shown
to each clinician under a balanced Latin allocation. Use at least 160 unique
image-claims, preferably 240 unless an outcome-blind variance justification
supports 160. Every claim retains all four cells, but no clinician sees the
same claim twice. This preserves the within-claim product contrast while
removing the dominant recall pathway.

If only two clinicians and roughly 480 total decisions are feasible, use a
240-claim randomized **between-image** 2 x 2 and narrow the conclusion to an
average randomized human interaction. It cannot authorize the stronger
instance-level “product-orbit defect” wording.

## 1. Why axis admission is insufficient

Let `R` be a render operation, `W` a wording operation, and `H(x,c)` a
clinician's decision process. Axis-wise admission checks only

\[
H(Rx,c) \approx H(x,c), \qquad H(x,Wc) \approx H(x,c).
\]

It does not check

\[
H(Rx,Wc) \approx H(x,c)
\]

or the product contrast

\[
H(Rx,Wc)-H(Rx,c)-H(x,Wc)+H(x,c).
\]

The conclusion would follow only under an additional separability assumption
about human perception and question interpretation. That assumption is the
object under dispute and therefore cannot be inserted as a premise. Pairwise
admission is also finite-sample behavioral evidence, not mathematical
equivalence or transitivity.

Accordingly, the valid claim hierarchy is:

- **axis admission only:** “model-score nonseparability under two individually
  clinician-admitted axes”;
- **axis admission plus bounded fixed-panel product effect:** “a model excess
  product effect relative to this fixed clinician panel”;
- neither result proves a decoder mechanism, a general human/model divide, or
  a general solution to medical VLM hallucination.

## 2. Audit of the proposed 60-claim within-reader 2 x 2

### What it identifies well

- It contains a real four-cell product rather than inferring an interaction
  from two unrelated marginal experiments.
- It compares the model and humans on the same assigned factor pair.
- Hash-frozen assignment prevents post-outcome selection.
- Four cells per claim remove image-level baseline heterogeneity from the
  within-claim contrast.
- At 240 decisions per clinician it is much cheaper than 900 decisions for a
  full 5 x 3 grid.

### Fatal or major limitations

#### A. Recall can manufacture the desired negative control

Each clinician sees the same image and finding four times. A washout does not
make a chest radiograph unrecognizable. A remembered prior answer encourages
within-person consistency, which specifically suppresses the interaction the
control is intended to measure. Session learning, fatigue, monitor differences,
and knowledge that cases recur can act in the opposite direction. Merely
randomizing rows inside four blocks does not separate cell condition from
session/order.

If this repeated-reader version is retained as a sensitivity, it requires a
four-sequence Williams/Latin counterbalance, a prespecified washout, distractor
cases, no duplicate image within a session, a per-exposure recognition probe,
and explicit cell-by-session/order/recognition analyses. A material recall or
session effect makes it inconclusive. These controls reduce but do not remove
the causal concern.

#### B. Sixty claims are not sixty guaranteed clusters

The current pack builder guarantees unique `(image, finding)` rows, not unique
images. The inferential N is the number of unique images after clustering, not
240 cells, 120 clinician-claim records, or the number of model evaluations.
Factorial repetitions improve a contrast within a claim but do not create new
clinical sampling units.

#### C. The factor-pair estimand is narrower than the 5 x 3 science grid

There are four nonbaseline renders and two nonbaseline wordings, hence eight
possible assigned variant pairs. Sixty claims yield only seven or eight claims
per pair even under exact global near-balance, and far fewer within the 16
finding-by-reader-vote strata. “Balanced over each family” is insufficient;
the **joint pair** and its association with finding, vote bin, and cell order
must be frozen and audited.

The valid estimand averages over the prespecified pair-assignment distribution.
It cannot rule out a harmful localized pair and cannot be described as a human
control of every cell in the full grid.

#### D. The proposed primary contrast can cancel harm

For clinician `j` and claim `i`,

\[
D^H_{ij}=L_{11}-L_{10}-L_{01}+L_{00}
\]

is a legitimate signed difference-in-differences. But a mean near zero can be
created by equal numbers of harmful and repairing interactions. That does not
establish product stability. The signed mean is useful, but it cannot be the
only human acceptability gate.

#### E. The response-to-loss map is underdefined

`supported/refuted/undetermined` plus ordinal confidence does not uniquely
define a proper loss against a fractional reader target. Mapping
`undetermined` to 0.5 or mapping ordinal confidence to probabilities after
seeing returns would change the estimand. Elicit a frozen probability of
finding presence directly (for example, 0--100 in fixed increments), retain
the three-state decision as a secondary clinical label, and record
`not_assessable` separately. Factor-dependent nonassessability must be a failure
or separate endpoint, never silently excluded.

The target `q` is a three-reader empirical support fraction. Therefore the
loss is **agreement with that reader distribution**, not disease truth or
diagnostic accuracy. Leave-one-reader-out and clear-bin analyses remain
sensitivities.

## 3. Minimal credible authorizing design

### 3.1 Sampling and allocation

1. Select **at least 160, preferably 240, unique images**, stratified over the
   four frozen findings and four reader-vote bins. If an image supports more
   than one selected claim, retain only one for this control.
2. Assign one of the eight nonbaseline render-wording pairs to each claim using
   a frozen randomization. Balance the **joint pair**, not only the two
   marginals. At 240 claims every pair receives 30 claims globally; use an
   orthogonal near-balance over finding and vote bin.
3. Construct the four cells `00, 10, 01, 11` for every claim.
4. Use four fixed clinicians. For each claim, randomly assign its four cells
   one-to-one to the four clinicians. Rotate the assignment with a balanced
   Latin schedule so every clinician sees each cell, render family, wording
   family, finding, and vote bin equally often.
5. Each clinician sees each claim exactly once. Blind image ID, vote bin,
   factor names, baseline identity, other readers, and all model outputs.
6. Include a small shared baseline calibration set (for example 20 additional
   unique claims judged once by all four clinicians) to report fixed-panel
   calibration and disagreement. It is not used to tune the product margin.

This design is a balanced incomplete-reader design: all four cells exist
within each claim, but no within-reader repeat exists. Clinician main effects
are orthogonalized by the cell rotation. The estimand remains fixed-panel and
image-population, not reader-population.

### 3.2 Workload

| Design | Primary decisions | Decisions/clinician | Unique claim clusters | Recall risk | What it supports |
|---|---:|---:|---:|---|---|
| Current 60-claim fractional 2 x 2, 2 clinicians | 480 | 240 | at most 60 | High: four views/clinician/claim | Within-claim contrast, but not a credible small human-effect bound |
| Recommended 160-claim fractional 2 x 2, 4 clinicians | 640 | 160 | 160 | None within clinician | Within-claim fixed-panel interaction; viable if contrast SD is at most about 0.25 for a 0.05 margin |
| Preferred-power 240-claim fractional 2 x 2, 4 clinicians | 960 | 240 | 240 | None within clinician | Stronger within-claim fixed-panel interaction and joint-pair coverage |
| Two-clinician between-image 2 x 2, 240 claims | 480 | 240 | 240 | None | Average randomized interaction only; no per-claim product contrast |
| Full 5 x 3, 60 claims, 2 clinicians | 1,800 | 900 | at most 60 | Extreme: 15 views/clinician/claim | Full-grid mapping but poor population power and severe fatigue/recall |

The 20-claim shared baseline calibration set adds 80 decisions to either
four-clinician design: 720 total (180/clinician) for `n=160`, or 1,040 total
(260/clinician) for `n=240`.

### 3.3 Outcomes and estimands

For every displayed cell, record before any free text:

- probability that the finding is present, on a frozen numerical scale;
- `supported / refuted / undetermined`;
- `assessable / limited / not_assessable`;
- fixed ordinal confidence;
- response time and recognition only as process diagnostics.

Let `Y_ijrp=(h_ijrp-q_i)^2` be reader-distribution Brier loss from clinician
probability `h`, with clinician `j` assigned to cell `(r,p)`. Estimate the
clinician-adjusted product contrast using claim and clinician effects:

\[
Y_{ijrp}=\alpha_i+\gamma_j+\beta_R r+\beta_W p+\theta_H rp+\epsilon_{ijrp}.
\]

The Latin allocation makes clinician identity orthogonal to the four cells.
Inference resamples whole images and preserves all four cells, pair assignment,
and clinician assignment. Report clinician-specific interactions as fixed-panel
heterogeneity; a qualitative sign reversal is a construct warning.

Three quantities are required:

1. **Human signed product harm:** `theta_H`, with a one-sided upper confidence
   bound below a prospectively justified human acceptability margin `delta_H`.
2. **Non-cancelling human instability:** excess absolute interaction or a
   product-exclusive adverse-transition rate, referenced to the frozen shared
   baseline/sham noise estimate. This is a conjunctive guard against harmful
   and repairing claims cancelling.
3. **Model-over-panel excess:** on the exact same sampled cells and probability
   loss, `Delta_m=theta_m-theta_H`, with a positive cluster-bootstrap lower
   bound and a prospectively frozen minimum excess. Test Huatuo and Hulu
   separately with shared image weights.

`not_assessable` has its own joint-excess rate. A joint cell that becomes
unassessable cannot be removed from the Brier analysis and counted as evidence
of human stability; it triggers the missingness guard and a conservative
sensitivity.

The model's full-grid Haar PAEL and this human 2 x 2 contrast are not numerically
identical estimands. For an exact model-human statement, recompute the model's
plain 2 x 2 contrast on the **same assigned cells**. PAEL remains the separate
full-grid primary model statistic.

## 4. Prospective noninferiority/equivalence power

The human margin cannot be borrowed automatically from the model's project
threshold. The existing `R >= 0.05` is explicitly an operational model MCID,
not an elicited bound on acceptable clinician instability. `delta_H`, its
scale, and the non-cancelling harm-rate ceiling must be justified and frozen
before returns are opened.

For planning only, suppose the standardized human product contrast has true
mean zero, cluster SD `sigma`, a one-sided 5% upper-bound test, and margin
`delta_H=0.05`. Approximate power is

\[
\Phi\left(\sqrt{n}\,\delta_H/\sigma-z_{0.95}\right).
\]

| Unique clusters | SD=.15 | SD=.25 | SD=.30 | SD=.50 |
|---:|---:|---:|---:|---:|
| 60 | 0.826 | 0.462 | 0.362 | 0.192 |
| 160 | 0.995 | 0.812 | 0.678 | 0.352 |
| 240 | ~1.000 | 0.927 | 0.826 | 0.462 |
| 320 | ~1.000 | 0.973 | 0.909 | 0.557 |

The required cluster counts for 80% power at a true mean of zero are
approximately 56, 155, 223, and 619 for SD values 0.15, 0.25, 0.30, and 0.50.
These are optimistic normal approximations: finite-panel dependence,
stratum-macro weighting, factor-pair heterogeneity, and a conjunctive
non-cancelling guard can reduce power.

Therefore:

- `n=60` can authorize a 0.05 ceiling only if an independent, outcome-blind
  variance basis makes an SD near 0.15 credible;
- `n=160` is a defensible minimum at SD about 0.25;
- `n=240` is preferred if SD up to about 0.30 is plausible;
- if the plausible SD is 0.50, none of these designs is powered for a 0.05
  absence-style statement; report “not bounded” rather than “humans are
  invariant.”

The final bootstrap must use unique images, not cells or clinician decisions.
Failure to establish the upper bound is inconclusive; it is not evidence that
humans have a product effect. A clinically meaningful positive human effect,
however, directly blocks the model-specific wording.

## 5. Alternatives

### Full 5 x 3 within-claim grid

**Strength:** matches every named model cell and exposes localized pair harm.

**Why not minimal:** 15 presentations of the same claim per clinician create
more recall than scientific information; 900 decisions/clinician still give
only 60 image clusters. It maps the grid but does not solve population power.
It is credible only if cells are distributed across a much larger reader panel
so no clinician repeats a claim, which turns it into a substantially larger
human study.

**Use:** optional mapping after a positive fractional result, not the initial
authorizing gate.

### Fractional 2 x 2 with assigned factors

**Strength:** best match to the product question per unit workload; retains a
genuine interaction and can be evaluated on exactly the same model cells.

**Risk:** identifies only the prespecified average over assigned variant pairs;
60 claims are sparse across eight pairs. Repeated evaluation by the same reader
creates the strongest validity threat.

**Use:** recommended, but distribute the four cells across four clinicians and
increase unique claims. This is the only minimal option that preserves an
instance-level product contrast without recall.

### Between-image randomized factorial

**Strength:** with two clinicians and the same 480-decision budget, 240 unique
claims can be evaluated once each, eliminating recall and increasing cluster
N fourfold. Stratified random assignment gives an unbiased average interaction.

**Risk:** different images occupy different cells. The result is a
population-average randomized interaction, not a within-claim mixed derivative;
claim heterogeneity increases variance and the human control no longer mirrors
the model's orbit-level object.

**Use:** resource-constrained fallback. It supports “the model interaction
exceeds the fixed-panel average interaction under randomized cell assignment,”
not “humans compose these operations for the same claim.”

## 6. Exact claim boundary

If the recommended within-claim design passes, the strongest justified wording
is:

> Across the prespecified VinDr four-finding/four-reader-vote sample and the
> frozen distribution of one admitted render variant and one admitted wording
> variant, the render-by-wording product increases reader-distribution loss in
> each evaluated model more than in this fixed clinician panel, while the
> panel's mean and non-cancelling product instability remain below
> prospectively frozen acceptability bounds.

It does **not** justify:

- “humans are invariant” or generalization to the clinician population;
- safety of every one of the eight variant pairs unless pairwise bounds are
  separately powered;
- equivalence of the full 5 x 3 human response surface;
- a decoder, representation, or causal neural mechanism;
- general medical-VLM hallucination mitigation, unrestricted OE, or reports.

If human product control is absent or underpowered, use only:

> Model-score nonseparability under two individually clinician-admitted axes.

If the between-image fallback is used, replace “product for the same claim” by
“average randomized render-by-wording interaction over the sampled claims.”

## 7. Authorizing decision rule

The draft should remain non-authorizing until all of the following are frozen:

1. unique-image sampling and joint factor-pair allocation;
2. recall-free reader allocation, or an explicit downgrade to the between-image
   estimand;
3. direct probability elicitation and treatment of `not_assessable`;
4. `delta_H`, the model-over-panel minimum excess, and the non-cancelling
   instability ceiling;
5. a variance/power basis showing at least 80% power for the intended human
   upper-bound statement;
6. fixed-panel wording and prohibition on reader-population generalization;
7. same-cell model reanalysis for the model-human comparison.

**Final disposition:** `MAJOR REVISION`. Keep the requirement for a human
joint-product control. Do not treat the current 60-claim, repeated-reader 2 x 2
as sufficient for the model-specific claim.

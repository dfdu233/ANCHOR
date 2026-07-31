# Conditional Style-Field Mechanism Audit

## Question

Global source-center and per-style additive corrections explain little of the
late-fusion displacement produced by acquisition-style counterfactuals. The
next identifiable question is whether the residual has a simple conditional
structure:

\[
\Delta_{i,s}=h(T_sx_i)-h(x_i).
\]

If a reusable single-view DG operator exists, it must both represent this
field and predict it without observing target labels or additional target
styles.

## Exact decomposition

For a balanced case-by-style grid,

\[
g=\mathbb E_{i,s}\Delta_{i,s},\quad
c_i=\mathbb E_s\Delta_{i,s}-g,\quad
a_s=\mathbb E_i\Delta_{i,s}-g,
\]

\[
\epsilon_{i,s}=\Delta_{i,s}-g-c_i-a_s.
\]

These four components are orthogonal in the empirical Frobenius inner
product. Their squared-norm ratios therefore sum to one exactly. At the final
prompt token, the case component is 68.20%--69.31% across all 11 lineages,
whereas the reusable centered style component is 1.42%--1.53%.

This refines the preceding result. The field is not merely “non-global”: most
of its observed energy is a response shared across transformations of the
same image.

## Transductive structure versus inference

For each held case-style cell, a crossed predictor estimates the case term
from the other styles of the same image and estimates the style term only
from other patients. It explains 63.78%--65.21% of the observed field.

That result is a transductive multi-view reconstruction, not an inference
method or formal ceiling: it observes five transformed displacements sharing
the held case's clean-state origin. To test a
single-view route, a linear-kernel ridge map was fit from the clean image
state to the case response, with patient-grouped outer folds and nested
source-only ridge selection. Its incremental error reduction over
patient-LOO style means is only 0.38%--1.21% at the final prompt state. Random
target permutations are descriptive only because they use 20 repeats and do
not reselect ridge penalties. The supported conclusion is little incremental
prediction by this tested metric, not absence of all actionable predictors.

## Parameter-induced prior direction

The parameter/prompt-defined null-image direction is

\[
v_i^\varnothing=h(x_\varnothing)-h(x_i).
\]

The same-case orthogonal projection

\[
P_i^\varnothing \Delta_{i,s}
=
\frac{\langle\Delta_{i,s},v_i^\varnothing\rangle}
{\|v_i^\varnothing\|^2}v_i^\varnothing
\]

accounts for 12.72%--13.49% of final-prompt style displacement. This exceeds
the upper 95% interval obtained by permuting case directions in every
lineage; 72.5%--76.25% of cells point toward, rather than away from, the null
state.

However, this is not a null-specific control. Let

\[
v_i^\mu=\mu_{-p(i)}-h(x_i)
\]

point from the clean state to a leave-one-patient clean-state centroid. It
explains 22.95%--23.94% of the field, more than the null direction in every
lineage, and 91.25%--93.75% of cells point toward it. Orthogonalized unique
energies are 15.39%--16.21% for the centroid and 5.03%--5.70% for null. The
null and centroid projections overlap and cannot be added.

The supported mechanism is therefore case-conditioned contraction toward a
generic representation attractor, not a null-specific causal prior. It is a
control for the separately tested
style-conditioned prior-switching hypothesis:

- **generic contraction:** multiple styles move a case toward a common
  representation region, without yet identifying a clinical prior;
- **prior switching:** different styles select different disease-prior
  directions.

The decisive output-level \(2\times2\) content/style experiment should test
whether style-specific disease directions remain after residualizing generic
contraction.

## Literature boundary

This probe is related to but distinct from:

- [MatchDG](https://proceedings.mlr.press/v139/mahajan21b.html), which uses
  same-object cross-domain matching as a DG objective;
- [Test-Time Style Shifting](https://proceedings.mlr.press/v202/park23d.html),
  which moves target style statistics toward a source style;
- [STYLIP](https://openaccess.thecvf.com/content/WACV2024/html/Bose_STYLIP_Multi-Scale_Style-Conditioned_Prompt_Learning_for_CLIP-Based_Domain_Generalization_WACV_2024_paper.html),
  which conditions CLIP prompts on instance style;
- [Selecting Data Augmentation for Simulating Interventions](https://proceedings.mlr.press/v139/ilse21a.html),
  which explains why a synthetic augmentation is a causal intervention only
  under an equivariance condition.

The present evidence does not establish novelty or clinical benefit. It
identifies which representation-level factorization survives the controlled
probe and which algorithm class does not.

## Scope and stop rule

The evidence is limited to six fixed synthetic Fourier transforms, 40 exposed
MIMIC development images, one prompt, stored float16 coordinates, and
correlated Qwen2.5-VL-7B lineages. It does not justify:

- natural scanner/hospital generalization;
- clinical accuracy or hallucination reduction;
- a single-view conditional correction;
- interpreting crossed cell completion as deployment performance.

Because clean-state predictability remains below 2%, no new correction module
is launched from this branch. The next trustworthy experiment is the
orthogonal content/style prior test already running on the other server, with
the clean-centroid contraction direction included as a nuisance control.

## Layerwise follow-up

The full five-layer trace was repeated for all 11 lineages. At prompt states,
median clean-centroid projected energy changes from 18.97% at layer 0 to
19.38%, 20.30%, 26.44%, and 23.61% at layers 7, 14, 21, and 27. It exceeds
the corresponding image-token projection at every sampled layer and lineage.

That normalized projection statistic is an alignment measure, not proof of
distance contraction. The median across lineages of the mean log
squared-distance ratio

\[
\log
\frac{\|h(T_sx)-\mu_{-p}\|^2}
{\|h(x)-\mu_{-p}\|^2}
\]

is +0.414, +0.182, +0.150, -0.095, and -0.106 at layers 0, 7,
14, 21, and 27. Every lineage moves farther from the centroid through layer
14 and closer at layers 21 and 27. The median fraction of individual
case-style cells that becomes closer rises from 17.92% at layer 0 to 62.50%
at layer 27.

The null endpoint remains weaker at every layer. It is not distinguishable
from a case-permuted direction at layer 0, then exceeds that control from
layer 7 onward. The defensible dynamic statement is therefore:

> A centroid-aligned component is present before contraction is realized;
> actual case-anchored contraction emerges only in late language layers. A
> null-aligned component also emerges but is not specific enough to identify
> a clinical prior.

This is descriptive, nonmonotonic layerwise evidence. It does not establish
that fusion causally amplifies contraction, that prompt and pooled image-token
states are directly comparable in absolute scale, or that downstream utility
improves.

Evidence and figure:
`corrected_runs/layerwise_attractor_v1/`.

## Complete-sentence evidence follow-up

The same direct-distance test was applied to a six-dimensional clinical
evidence vector. Each coordinate is a complete-sentence teacher-forcing
contrast between positive and negative statements for cardiomegaly, device,
edema, effusion, opacity, or pneumothorax. On 64 exposed MIMIC development
images (58 patients), the mean log after/before **squared Euclidean
distance** to a patient-LOO clean evidence centroid is:

- HuatuoGPT-Vision-7B: -0.161, patient-cluster 95% CI
  [-0.316, -0.022], with 65.89% of case-style cells closer;
- exact Qwen2.5-VL-7B base: -0.113, 95% CI [-0.255, +0.042],
  with 60.68% closer.

Centroid directions explain 43.23% and 39.76% of style displacement,
respectively, exceeding null-endpoint directions (25.85% and 30.43%). The
paired Huatuo-minus-base difference is -0.048, 95% CI [-0.241, +0.147], so
there is no evidence that medical instruction tuning amplified the effect.

The ratio is computed from squared Euclidean distances. Huatuo's point
estimate corresponds to \(\exp(-0.161/2)=0.923\), an approximately 7.7%
geometric distance reduction. The patient-balanced sensitivity remains below
zero, but only four of six individual style intervals do; this is a pooled
fixed-transform property, not a universal statement about style.

This provides a clinical-output-coordinate nuisance control, not a generated
accuracy result: style-specific disease-prior switching must be shown beyond
generic evidence contraction. The centroid uses other clean target-development
cases and is therefore transductive; it is not a source-only deployment
operator. Evidence:
`corrected_runs/clinical_evidence_attractor_v1/`.

## Residual clinical style signature

The generic centroid component does not exhaust the complete-sentence
clinical evidence displacement. For each case \(i\) and fixed style \(s\), we
first remove the projection onto the patient-LOO clean-centroid direction:

\[
\rho_{i,s}
=
\Delta_{i,s}
-
\frac{\langle\Delta_{i,s},r_i\rangle}{\|r_i\|^2}r_i,
\qquad
r_i=\mu_{-p(i)}-e_i.
\]

Before testing style identity, we additionally subtract each case's mean
residual over the six styles. This prevents a case-common residual or grand
effect from being counted as a reusable style signature. In this
case-centered field, the exact style effect accounts for only 1.02% of
residual energy in the Qwen base model and 1.12% in Huatuo; case effects still
account for 76.65% and 69.46%.

Despite its small magnitude, the signature is reproducible across patients.
A leave-one-patient style prototype identifies the six fixed styles at 27.08%
for Qwen and 21.61% for Huatuo, versus chance \(16.67\%\). Patient-blocked
style permutations give \(p=.001\) and \(p=.034\), respectively. Prototype
error reduction over the zero predictor is only 2.33% and 0.97%, but exceeds
its corresponding permutation null in both models (\(p=.001\)).

The matched Huatuo/base style directions have mean cosine 0.579. Only eight
of all \(6!=720\) cross-checkpoint style assignments match or exceed this
value (exact \(p=.0111\)). Disease-wise whitening preserves the result:
matched mean cosine 0.582 and exact \(p=.0069\). Raw-coordinate Huatuo
signature norms are 24.5%--52.1% of the base norms, but this ordering does not
hold under all reasonable scalings. It therefore cannot identify a causal
attenuation effect of medical tuning.

The supported conclusion is narrow:

> After generic centroid contraction and each case's common residual are
> removed, the six fixed spectral transformations retain a weak,
> cross-patient disease-evidence signature shared by exact base and medical
> checkpoints.

This supplies aggregate alignment across the two paired checkpoints, not
proof that every style direction transfers or that an architecture-wide
mechanism has been independently reproduced. It explains about 1% of
residual energy and is not itself a useful decoder. The tests are exploratory
and unadjusted; Huatuo style-identification \(p=.034\) does not survive a
two-endpoint Bonferroni correction.
The independent content-removed \(2\times2\) experiment now has a frozen
directional prediction: its style-only evidence drift should align with the
exported raw or whitened \(6\times6\) signature vectors. Failure to reproduce
that direction falsifies the prior-switching interpretation; reproduction
without generated utility still does not justify a mitigation method.

Evidence and exported vectors:
`corrected_runs/residual_style_signature_v1/`.

## Disease-specificity of the residual signature

A remaining alternative is that each style simply raises or lowers all six
disease scores together. Let

\[
u=6^{-1/2}(1,\ldots,1),\qquad
\rho^\parallel=\langle\rho,u\rangle u,\qquad
\rho^\perp=\rho-\rho^\parallel.
\]

Here \(\rho^\parallel\) is a literal uniform affirmation/negation axis and
\(\rho^\perp\) contains relative disease-evidence changes. The uniform axis
accounts for 24.64% of the Qwen-base style-signature energy and 20.81% of the
Huatuo energy. Patient-cluster 95% intervals are [10.14%, 44.73%] and [5.82%,
41.09%], respectively; more than 99% of resamples remain below one half.
Because the contrast subspace has five dimensions while the uniform space has
one, this is not per-dimension dominance: uniform energy is 1.63 times the
average contrast-coordinate energy in base and 1.31 times in Huatuo. The test
only rejects a purely equal-coordinate explanation.

Removing the uniform axis does not eliminate cross-patient detection.
Contrast-only held-patient style identification is 25.78% in base and 22.14%
in Huatuo (chance 16.67%; patient-blocked permutation \(p=.001\) and
\(p=.017\)). Contrast-only prototype \(R^2_0\) is 3.32% and 1.30%, with
\(p=.001\) in both models.

The exact paired checkpoints have contrast-only matched-style cosine 0.727,
with patient-cluster 95% interval [0.350, 0.792]. Identity is uniquely maximal
among all \(6!\) style assignments and all \(6!\) disease-coordinate
assignments in the fixed cohort, but this ranking does not survive patient
resampling: the style-identity and disease-identity margins have intervals
[-0.231, 0.095] and [-0.189, 0.076]. Thus the supported population-level
statement is positive aggregate contrast alignment, not a unique
style-to-disease mapping.

The observed contrast spectrum has entropy effective rank 2.43--2.58, but
patient-blocked style permutations do not establish unusually low rank. The
result therefore rules out a *purely* uniform answer-bias explanation for this
cohort without proving a low-rank clinical prior. Shared architecture,
teacher-forced sentence templates, fixed synthetic transforms, and
target-derived preprocessing remain alternative explanations.

The sequential radial-then-uniform projections do not commute and reintroduce
a radial component. A primary joint projection onto
\(\operatorname{span}\{r_i,\mathbf1\}^{\perp}\) preserves the finding:
held-patient style identification is 28.13% in base and 24.74% in Huatuo
(\(p=.001\) each), with \(R^2_0=5.04\%\) and 1.32%. Matched-style cosine is
0.699, patient-cluster 95% CI [0.293, 0.790]. A separately column-normalized
disease-profile cosine is 0.717. Their fixed-cohort assignment ranks are
strong, but patient-bootstrap identity margins still cross zero. Joint
nuisance removal therefore supports positive aggregate nonuniform structure,
not a population-unique style-to-disease map.

Evidence:
`corrected_runs/style_prior_specificity_v1/`.

## Equivalent-language falsification

The next audit held patients, images, six styles, six diseases, and
teacher-forcing budget fixed while replacing both the question and the
complete positive/negative answer frames. On 16 HuatuoGPT-Vision development
patients, the original style signature retained positive mean alignment with
the `demonstrates` frame (cosine `0.570`, patient-bootstrap 95% CI
`[0.014, 0.746]`) and with the equal-length `evidence present/absent` frame
(`0.456`, `[0.051, 0.619]`). Thus the residual is not entirely explained by
the original two-token negation-length difference.

The stronger identification test failed. Cross-template held-patient style
identification was `17.7%` and `21.9%` against `16.7%` chance
(`p=0.379/0.058`), and both identity-versus-best-mismatch margins were
negative. The pre-registered template-invariance gate therefore failed and
the unmodified Qwen checkpoint was not run. This narrows the result to a weak
template-robust susceptibility direction, not a uniquely indexed
style-conditioned clinical prior.

Evidence:
`corrected_runs/style_prior_template_probe_v1/`.

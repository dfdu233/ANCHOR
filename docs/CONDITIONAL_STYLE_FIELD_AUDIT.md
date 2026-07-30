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

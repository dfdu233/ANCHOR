# Conditional Acquisition-Style Field Audit

This audit asks what remains after global per-style offsets fail. For case
\(i\), source style \(s\), and a fixed representation,

\[
\Delta_{i,s}=h(T_sx_i)-h(x_i).
\]

The balanced two-way decomposition is

\[
\Delta_{i,s}=g+c_i+a_s+\epsilon_{i,s},
\]

where the grand, centered case, centered style, and interaction components
are mutually orthogonal under the observed squared Euclidean geometry.

## Controlled result

The experiment uses the same 40 exposed frontal MIMIC development images
(38 patients), six fixed PubMedVision Fourier styles, fixed report prompt,
and 11 correlated Qwen2.5-VL-7B lineages as the preceding style-field audit.

At the final prompt token:

- centered case effects explain **68.20%--69.31%** of displacement energy;
- centered style effects explain only **1.42%--1.53%**;
- case-style interaction accounts for **22.97%--23.91%**;
- a transductive held-cell additive predictor, which may use the same case under the other
  five styles, explains **63.78%--65.21%**;
- a patient-held-out linear-kernel predictor using only the unmodified image
  state adds just **0.38%--1.21%** over patient-LOO style means.

Thus the large transductive structural signal is not currently recoverable by the tested
single-view low-complexity operator. The multi-view result is deliberately
not called deployable: it observes other transformed views of the test case.

## Null-prior component

Let

\[
v_i^{\varnothing}=h(x_{\varnothing})-h(x_i)
\]

be the same-case direction from the real-image state to the null-image state.
Projecting each \(\Delta_{i,s}\) onto this one-dimensional direction explains
**12.72%--13.49%** of final prompt displacement. Across lineages,
72.5%--76.25% of case-style cells have positive signed alignment. Every
same-case projection exceeds the corresponding 200-permutation upper 95%
control (finite permutation \(p=1/201\)).

This projection is not null-specific. A direction from each clean state toward
the leave-one-patient clean-state centroid explains **22.95%--23.94%**, and
91.25%--93.75% of cells point toward that centroid. After orthogonalizing the
two endpoint directions, the centroid-unique component explains
15.39%--16.21%, versus only 5.03%--5.70% for the null-unique component.

The supported statement is therefore generic, case-anchored contraction. The
synthetic style field is aligned with the null state, but even more strongly
with an ordinary clean-state centroid. It does not establish a null-specific
clinical prior, changed correctness, or behavior under natural scanner shift.

## Decision

Do not derive a single-view correction from the clean-state KRR signal: the
incremental energy is below 2% in every lineage, and no downstream
actionability threshold was tested. The useful handoff to the independent
style-conditioned-prior experiment is instead a sharper mechanistic
alternative:

1. style-specific prior switching, and
2. image-conditioned contraction toward a generic representation attractor

must be distinguished at output level. The current analysis does not yet
identify that attractor with a clinical prior. A valid counterfactual experiment
should test both directionality hypotheses rather than treating any answer
flip as evidence for a reusable source center.

All statements are finite-sample properties of stored float16 activations.
The lineages share a base model and data and are not independent population
replications.

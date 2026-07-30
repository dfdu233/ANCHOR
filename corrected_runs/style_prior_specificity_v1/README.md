# Disease-Specificity Audit of the Residual Style Signature

This audit tests whether the weak residual style signature is merely a
uniform tendency to raise or lower every disease score. Let

\[
u=\frac{1}{\sqrt 6}(1,\ldots,1),\qquad
\rho^{\rm uniform}=\langle\rho,u\rangle u,\qquad
\rho^{\rm contrast}=\rho-\rho^{\rm uniform}.
\]

The contrast field is exactly orthogonal to the uniform disease axis. It
therefore contains only relative changes among the six disease evidence
coordinates.

## Results

| Model | Uniform share of style signature | Contrast-only style ID | Permutation p | Contrast effective rank |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B base | 24.64% | 25.78% | .001 | 2.43 |
| HuatuoGPT-Vision-7B | 20.81% | 22.14% | .017 | 2.58 |

Patient-cluster bootstrap intervals for the uniform share are [10.14%,
44.73%] for base and [5.82%, 41.09%] for Huatuo; 99.3% and 99.7% of resamples
remain below 50%.

This comparison is dimension-imbalanced: the uniform space is one-dimensional
and the contrast space is five-dimensional. Relative to average energy per
contrast dimension, the uniform dimension is 1.63 times larger in base and
1.31 times larger in Huatuo. The result rules out a purely uniform shift; it
does not show that contrast dominates per dimension.

Chance style identification is 16.67%. Contrast-only prototype \(R^2_0\) is
3.32% for base and 1.30% for Huatuo; both exceed their patient-blocked
permutation nulls (\(p=.001\)).

Across the exact paired checkpoints, matched contrast-only style directions
have mean cosine 0.727. The identity style assignment is uniquely maximal
among all \(6!=720\) assignments in this fixed cohort (exact conditional
\(p=1/720\)). Keeping style labels matched and permuting the six disease
coordinates also gives \(p=1/720\). Disease-wise standardization preserves
these fixed-cohort results, with matched mean cosine 0.716.

Patient-cluster bootstrap supports positive aggregate alignment: the raw
matched-cosine 95% interval is [0.350, 0.792], and the standardized interval
is [0.273, 0.785]. It does **not** support a population-level unique label
assignment. Raw style-identity and disease-identity assignment-margin
intervals are [-0.231, 0.095] and [-0.189, 0.076]; both standardized intervals
also cross zero.

The observed signature spectrum is concentrated rather than one-dimensional:
entropy effective rank is 2.43 for base and 2.58 for Huatuo; the first three
directions capture 96.6% and 98.4% of contrast energy. Patient-blocked style
permutations do not establish unusually low effective rank in either model,
so this is a descriptive spectrum rather than an inferred low-rank law.

## Interpretation

The residual signature cannot be reduced to a uniform affirmation/negation
axis in this cohort: most reusable style-signature energy lies in relative
disease coordinates, and the paired checkpoints have positive aggregate
alignment after that axis is removed. The current sample does not establish
that the exact style-to-disease assignment is unique in the patient
population.

It is not yet evidence of a style-conditioned *clinical prior*. Both
checkpoints share most architecture and language parameters, the coordinates
come from six fixed teacher-forced sentence templates, the centroid is
target-transductive, and all tests concern six fixed synthetic styles on the
same exposed cohort. Exact permutation probabilities are conditional on
these six labels and are exploratory/unadjusted.

The sequential radial-then-uniform projection is retained only as a
sensitivity because the two projections do not commute. The primary joint
projection onto
\(\operatorname{span}\{r_i,\mathbf 1\}^{\perp}\) gives:

| Model | Joint-complement \(R^2_0\) | Style ID | Permutation p |
|---|---:|---:|---:|
| Qwen2.5-VL-7B base | 5.04% | 28.13% | .001 |
| HuatuoGPT-Vision-7B | 1.32% | 24.74% | .001 |

Its matched-style cosine is 0.699 (patient-cluster 95% CI [0.293, 0.790]).
The properly column-normalized matched-disease-profile cosine is 0.717.
Fixed-cohort assignment tests are \(2/720\) and \(1/720\), but neither
identity-versus-best-mismatch margin has a positive patient-bootstrap lower
bound. The standardized joint analysis remains positive in aggregate
(style cosine 0.660; disease-profile cosine 0.674).

The independent content-by-style \(2\times2\) experiment remains decisive.
Its content-removed branch must reproduce the frozen contrast vectors on new
patients before this signal can be interpreted as a style-carried prior.

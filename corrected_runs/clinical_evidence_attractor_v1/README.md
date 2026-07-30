# Complete-Sentence Clinical-Evidence Attractor Audit

This read-only follow-up asks whether the late-layer centroid-contraction
property reaches a clinically interpretable output coordinate. For each image
and view, it constructs a six-dimensional vector from complete positive and
negative disease sentences:

\[
e_k(x)=
\overline{\log p_\theta(y_k^+\mid x,q_k)}
-
\overline{\log p_\theta(y_k^-\mid x,q_k)}.
\]

The probe contains 64 exposed MIMIC development images (58 patients), six
fixed PubMedVision Fourier styles, and six findings. It compares HuatuoGPT
Vision to its exact Qwen2.5-VL-7B base on identical inputs.
The patient-LOO centroid is estimated from the other clean MIMIC development
cases. This is a target-transductive mechanism diagnostic, not a source-only
inference algorithm.

## Result

The mean log ratio of after/before **squared Euclidean distance** to a
case-weighted patient-LOO clean evidence centroid is:

| Model | Mean log ratio | Patient-bootstrap 95% CI | Cells closer |
|---|---:|---:|---:|
| Qwen2.5-VL-7B base | -0.113 | [-0.255, +0.042] | 60.68% |
| HuatuoGPT-Vision-7B | -0.161 | [-0.316, -0.022] | 65.89% |

Centroid directions explain 39.76% and 43.23% of style displacement,
respectively, exceeding the corresponding null-endpoint values of 30.43% and
25.85%.

The paired Huatuo-minus-base log-ratio difference is -0.048, 95% CI
[-0.241, +0.147]. It does not support a claim that medical instruction tuning
amplifies contraction.

For Huatuo, \(\exp(-0.161/2)=0.923\), corresponding to an approximately 7.7%
geometric Euclidean-distance reduction for the pooled estimand. A
patient-balanced sensitivity analysis remains below zero. Four of the six
individual style intervals are below zero; two cross zero, so the pooled
result is not universal across transformations.

## Interpretation

On this controlled probe, complete-sentence clinical evidence in Huatuo
contracts toward a typical clean evidence state under synthetic acquisition
style. The exact base shows the same trend with uncertainty. This supplies an
output-space nuisance control for a style-conditioned prior-switching test:
style-specific disease drift must exceed this operational generic evidence
contraction baseline.

It does **not** establish generated-answer accuracy, factuality improvement,
natural scanner robustness, a deployable source center, or a causal
medical-tuning effect. The unstandardized Euclidean geometry depends on the
six sentence templates; the normalized centroid/null projection contrast is
descriptive and has no interval.

# Residual Clinical Style-Signature Audit

This audit separates generic evidence contraction from a style-specific
clinical direction. For complete-sentence six-disease evidence,

\[
\Delta_{i,s}
=
\beta_{i,s}(\mu_{-p(i)}-e_i)
+
\rho_{i,s},
\qquad
\rho_{i,s}\perp(\mu_{-p(i)}-e_i).
\]

If acquisition style selects a reproducible clinical prior, the residual
\(\rho_{i,s}\) should retain a same-style direction across held-out patients.
Before estimating style prototypes, the residual is additionally centered
within each case across styles. This removes the case's common residual and
isolates cross-patient style correspondence.

## Result

After radial contraction is removed:

| Model | Residual style energy | Held-patient prototype \(R^2_0\) | Style ID | Permutation p |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B base | 1.02% | 2.33% | 27.08% | .001 (ID) |
| HuatuoGPT-Vision-7B | 1.12% | 0.97% | 21.61% | .034 (ID) |

The prototype \(R^2_0\) exceeds its patient-blocked style-permutation null for
both models (\(p=.001\)). Nevertheless, most residual energy remains
case-conditioned: 76.65% for base and 69.46% for Huatuo.

Matched Huatuo/base style directions have mean cosine 0.579. Among all
\(6!=720\) style assignments, only eight reach this value
(exact \(p=.0111\)). Huatuo residual signature norms are only 24.5%–52.1% of
the exact base norms for all six styles.

After standardizing each disease coordinate by its clean-state standard
deviation, cross-model matched-style assignment remains non-random
(\(p=.0069\)); Huatuo prototype \(R^2_0=0.78\%\) and style-identification
accuracy 23.70% both remain above permutation (\(p=.001\) and \(p=.003\)).

## Interpretation

A weak but reproducible disease-evidence signature remains after generic
centroid contraction is removed. The six supplied style labels show
aggregate alignment across the exact base/medical checkpoints, although one
matched style direction has negative cosine. Huatuo's raw-coordinate
signature magnitude is smaller for all six styles, but this ordering changes
under residual-RMS scaling and is not a scale-invariant tuning effect.

The effect explains only about 1% of residual energy and does not yet justify
a decoding method. A style-matched counterfactual method should proceed only
if the independent content-removed \(2\times2\) experiment reproduces these
disease directions and improves generated outputs.

`summary.json` exports the raw and disease-whitened \(6\times6\) style
signature vectors for both exact checkpoints. They define a frozen,
directional prediction for the independent \(2\times2\) experiment; they are
not fitted on that experiment. Rows follow `styles = [style_0, ..., style_5]`
and columns follow the exported `diseases` order.

This is a target-transductive, teacher-forced audit on 64 exposed MIMIC
development images and six synthetic styles. The reported tests are
exploratory and unadjusted; Huatuo style-ID \(p=.034\) does not survive even a
two-endpoint Bonferroni correction. The audit makes no independent
replication, generated-accuracy, natural-scanner, clinical-validity, or
causal-training claim.

A fresh same-family result-to-claim review returned `PARTIAL`: high
confidence in implementation and numerics, but only medium confidence in the
mechanistic interpretation.

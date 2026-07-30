# Layerwise Style-Orbit Audit

## Question

The preceding controlled experiment found no output-level style contraction
after ordinary image--text alignment training. This audit asks a more precise
question:

> Does correct image--text pairing alter where acquisition-style variation is
> represented, even when the final clinical-evidence readout is unchanged?

This is deliberately orthogonal to style-conditioned prior switching. It
studies the *training-induced location* of style sensitivity, not whether a
particular style selects a disease prior.

## Controlled design

The exact Qwen2.5-VL-7B base model is compared with two 250-step
visual-merger-only lineages trained on the same 2,048 strict-CXR PubMedVision
records:

- **Matched:** correct image--text pairs.
- **Image-permuted:** identical empirical image and text marginals, but zero
  fixed image--text pairs and zero same-PMC/figure-group pairs.

Optimizer, initialization, seed, sample order, parameter count, and compute are
matched. The finite derangement is a randomized negative control; it is not an
empirical product distribution and does not estimate mutual information.

The probe uses the same 40 frontal MIMIC development images from 38 patients,
six fixed PubMedVision-derived Fourier styles, a real image, a null image, and
one fixed report prompt. No target label is used. Representations are pooled at
five vision blocks, the merger, and five language layers. At language layers we
separately trace the image-token mean and the last prompt token.

For layer \(\ell\), define per-patient susceptibility

\[
\kappa_\ell(x)=
\frac{
 \sqrt{\mathbb E_s\|h_\ell(T_sx)-h_\ell(x)\|_2^2/d_\ell}
}{
 \|h_\ell(x)-h_\ell(x_\varnothing)\|_2/\sqrt{d_\ell}
}.
\]

The numerator is same-content style drift; the denominator is real-versus-null
visual leverage. Patient-cluster intervals operate on paired lineage effects.

## Result

| Location | Matched \(\kappa\) | Permuted \(\kappa\) | Paired relative effect |
|---|---:|---:|---:|
| LLM prompt layer 7 | 0.1981 | 0.1992 | -0.93% [-1.69, -0.02] |
| LLM prompt layer 14 | 0.1547 | 0.1599 | -3.06% [-3.78, -2.33] |
| LLM prompt layer 21 | 0.1997 | 0.2066 | -2.35% [-3.15, -1.81] |
| LLM prompt layer 27 | 0.1929 | 0.2094 | -5.48% [-7.43, -4.00] |
| Complete-sentence evidence | 0.3175 | 0.3211 | -0.29% [-1.32, +2.51] |

At final prompt layer 27, the paired ratio effect is accompanied by:

- style drift: **-1.91%**, 95% CI [-3.44, -0.08];
- real-versus-null leverage: **+3.98%**, 95% CI [+3.36, +4.59];
- normalized susceptibility: **-5.48%**, 95% CI [-7.43, -4.00].

The two lineages are identical throughout the frozen vision tower. They remain
nearly indistinguishable at the trained merger and image-token means. The
separation appears only after text--image contextualization at the final prompt
token and grows with language depth.

## Interpretation

The strongest defensible observation is an exploratory **late-fusion alignment
gap** in this one seed-42 comparison: the matched lineage contextualizes style
differently relative to the fixed deranged lineage, even though it does not
produce detectable contraction in the complete-sentence clinical readout.
Thus:

1. visual-feature invariance is not required for a late semantic state to
   become relatively less style-sensitive;
2. a hidden-state invariance diagnostic is not sufficient evidence of
   hallucination mitigation;
3. ordinary alignment creates some late-fusion structure, but the output head
   does not automatically convert it into robust clinical generation.

This refines the previous null result rather than overturning it. It suggests
that any explicit orbit objective should target a causally validated
text--image fusion state and must be evaluated at the full generated-sequence
interface.

## Claim ceiling

This is an exploratory, single-seed, single-derangement mechanism result on previously exposed
MIMIC development images and designed Fourier styles. It does **not** establish:

- improved clinical accuracy, factuality, or report quality;
- general domain generalization;
- invariance under real hospital acquisition changes;
- a training-seed-stable effect;
- that Euclidean hidden-state distance is coordinate-invariant.

The source arrays, model checkpoints, and raw evidence traces remain outside
Git. The compact summary stores all paired statistics and fingerprints.

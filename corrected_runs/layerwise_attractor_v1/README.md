# Layerwise Acquisition-Style Contraction Audit

This follow-up traces the case-anchored style field through five sampled
Qwen2.5-VL-7B language layers. It uses the same 40 exposed frontal MIMIC
development images, six fixed PubMedVision Fourier styles, fixed report
prompt, and 11 correlated model lineages.

For each case, two same-origin endpoint directions are compared:

\[
v_i^\varnothing=h(x_\varnothing)-h(x_i),\qquad
v_i^\mu=\mu_{-p(i)}-h(x_i),
\]

where \(\mu_{-p(i)}\) is the clean-state centroid excluding every image from
the held patient's group.

## Result

At prompt states, median clean-centroid projected energy across lineages is:

| LLM layer | 0 | 7 | 14 | 21 | 27 |
|---:|---:|---:|---:|---:|---:|
| Clean centroid | 18.97% | 19.38% | 20.30% | 26.44% | 23.61% |
| Null endpoint | 7.12% | 7.61% | 11.27% | 9.27% | 13.04% |

The normalized clean-centroid alignment fraction exceeds the null endpoint at
every sampled layer in every lineage. The same normalized fraction at prompt
states also exceeds the corresponding pooled image-token value.
Between 92.08% and 98.33% of case-style prompt cells point toward the clean
centroid, depending on layer.

Positive projection is not equivalent to contraction. Directly comparing
distance before and after styling gives:

| LLM layer | 0 | 7 | 14 | 21 | 27 |
|---:|---:|---:|---:|---:|---:|
| Median mean log distance ratio | +0.414 | +0.182 | +0.150 | -0.095 | -0.106 |
| Median fraction actually closer | 17.92% | 30.42% | 32.92% | 58.33% | 62.50% |

Thus the style displacement points partly toward the centroid in every layer
but remains dominated by orthogonal motion through layer 14. Actual centroid
contraction appears only at layers 21 and 27, in all 11 lineages.

Null-endpoint alignment is below its case-permuted control at layer 0 and
exceeds the control from layer 7 onward. This does not make it null-specific:
the generic centroid direction remains stronger throughout.

## Interpretation

The robust property is a **nonmonotonic late-layer contraction onset**, not a
clinical-prior switch. Synthetic style perturbations already contain a
centroid-aligned component in early states, yet those states become farther
from the clean centroid overall. Only late prompt states become closer. The
analysis therefore does not establish that language fusion monotonically
amplifies contraction; it locates where direct contraction first appears.
This generic effect is an explicit nuisance control for any output-level
“style selects disease prior” test.

No downstream answer, factuality, or natural-scanner claim follows from this
activation analysis. Raw arrays remain outside Git; `summary.json` stores
their hashes and fingerprints.

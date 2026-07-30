# Acquisition-Style Field Factorization

This analysis asks whether a synthetic acquisition-style transformation acts
as a reusable additive global direction in a medical VLM.

For case \(i\), style \(s\), and a fixed representation layer, define

\[
\Delta_{i,s}=h(T_sx_i)-h(x_i).
\]

Under squared displacement, the least-squares optimal image-independent
correction for style \(s\) is exactly

\[
a_s^\star=\frac{1}{N}\sum_i\Delta_{i,s}.
\]

Therefore

\[
R_{\mathrm{style}}=
\frac{\|P_{\mathrm{style}}\Delta\|_F^2}{\|\Delta\|_F^2}
\]

is the exact finite-sample fraction removable by any additive per-style
offset. Its complement is not representable by such a global correction.

## Controlled evidence

The calculation uses the same 40 frontal MIMIC development images, six fixed
PubMedVision Fourier styles, and 11 Qwen2.5-VL-7B lineages: base, the
exploratory seed-42 matched/permuted pair, and four new matched/permuted pairs.

At the final prompt token:

- one global offset explains 5.97%--6.90%;
- optimal per-style offsets explain only **7.45%--8.35%**;
- at least **91.65%--92.55%** remains after the optimal style-only correction;
- per-image offsets explain 74.61%--75.60%.

At the final image tokens, optimal per-style offsets explain
13.27%--14.09%, leaving 85.91%--86.73% unresolved.

The narrow ranges across all 11 lineages are more stable than the failed
matched-versus-permuted contraction effect. The defensible conclusion is:

> On these paired synthetic views, acquisition-style displacement is
> predominantly image-conditioned rather than a reusable global style vector.

This provides a mathematical reason that methods assuming an additive global
source-center displacement, global style offset, or additive average can fail
even when style visibly changes the input. It does not prove that nonlinear
or image-conditioned corrections fail, nor that natural scanner shifts have
the same factorization.

Raw activation arrays remain outside Git. `summary.json` records their hashes
and the exact finite-sample projection results. Exactness is relative to the
stored float16 representations; the 11 lineages are correlated checkpoints,
not independent population replications.

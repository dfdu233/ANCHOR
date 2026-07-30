# Why a Global Source Center Is Insufficient

Let \(\Delta\in\mathbb R^{N\times S\times d}\) denote the representation
displacement caused by applying style \(s\) to image \(i\). Consider the
subspace

\[
\mathcal A_{\mathrm{style}}
=
\{A:A_{i,s}=a_s\},
\]

which contains every additive image-independent per-style correction. Under
the Frobenius inner product, its orthogonal projection is

\[
(P_{\mathcal A}\Delta)_{i,s}
=
\bar\Delta_{\cdot,s}
=
\frac1N\sum_i\Delta_{i,s}.
\]

Hence the best possible residual for this complete method class is

\[
\min_{\{a_s\}}\sum_{i,s}\|\Delta_{i,s}-a_s\|_2^2
=
\|\Delta-P_{\mathcal A}\Delta\|_F^2.
\]

This is an identity on the stored tensors, not a fitted downstream predictor
or an asymptotic claim.

## Result

Across 11 Qwen2.5-VL-7B lineages evaluated on identical paired inputs, the
optimal style-only projection explains just 7.45%--8.35% of the final
prompt-token displacement. Thus at least 91.65% remains outside the global
style-only subspace. The result is similarly unfavorable at final image
tokens: 85.91%--86.73% remains.

By contrast, the projection onto one offset per image explains
74.61%--75.60% at the prompt token. The displacement field is therefore
strongly image-conditioned.

## Consequence

This does not say that style is harmless. It says the operator

\[
T_s:x\mapsto h(T_sx)-h(x)
\]

cannot be approximated well by one vector depending only on \(s\) under the
observed representation and squared metric. A source-center method belongs to
this class only when it induces an image-independent additive displacement at
the measured representation.

These measurements motivate testing image-conditioned or nonlinear fields
rather than assuming a reusable additive offset. They are compatible with,
but do not test, the separate hypothesis that style switches a latent
clinical prior.

## Scope

The statement is exact only for the stored float16 activations of the observed
40 exposed frontal MIMIC images, six designed Fourier views, selected layers,
one prompt, and 11 correlated checkpoint lineages. The energy ratio is not a
fraction of cases or a downstream performance measure; case-only and
style-only projections overlap and must not be added. Natural scanner/site
variation and clinical utility require independent validation.

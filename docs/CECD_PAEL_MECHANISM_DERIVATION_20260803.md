# From interaction energy to clinical orientation: PAEL mechanism derivation

**Status:** outcome-blind mechanism derivation; no model or human outcome was
used.  This is a falsifiable prediction for post-behavioral-GO experiments, not
a causal conclusion.

## 1. Two quantities that must not be conflated

For one admitted product orbit, decompose the signed score matrix as

\[
M=A+J,
\]

where `A` contains the grand mean and both factor marginals and `J` is the
two-way-centered render × wording interaction.  Generic product instability is
the geometry of `J`, for example its Frobenius norm or singular values.  It has
no clinical direction by itself.

Let `L_q` be reader-distribution Brier loss after the dev-frozen monotone score
calibration.  The observed product-loss change is

\[
D_q(J)=L_q(A+J)-L_q(A).
\]

Around `A`, a local expansion gives

\[
D_q(J)
\approx
\langle \nabla L_q(A),J\rangle
+\frac12\operatorname{vec}(J)^\top H_q(A)\operatorname{vec}(J)
+O(\lVert J\rVert^3).
\]

The first term is **clinical orientation**: whether the interaction points
toward reader-grounded loss.  The second contains energy/curvature effects that
can make any sufficiently large interaction harmful even when its orientation
is generic.

The isospectral Haar reference preserves the singular values of `J` while
averaging its orientation in the centered render and wording subspaces.  Thus

\[
\operatorname{PAEL}_{Haar}
=D_q(J)-\mathbb E_{U,V}D_q(UJV^\top)
\]

is an operational estimate of named clinical orientation beyond same-spectrum
interaction geometry.  Under exactly isotropic local curvature the reference
removes the quadratic energy term in expectation; with anisotropic curvature
it remains a geometry-adjusted excess, not a pure gradient inner product.
Therefore PAEL must not be called a causal effect or an exact randomization
test.

## 2. Mechanism hypothesis

> Medical VLMs need not create more render × wording interaction energy on
> hallucinated claims.  Instead, a model-specific fusion-to-decoder transition
> can rotate existing interaction energy into alignment with the
> reader-grounded clinical-loss gradient.

This is **orientation formation**, not generic visual-information erasure and
not a universal early-layer claim.  It predicts a dissociation between
interaction energy and harmful orientation.

## 3. Locked predictions after behavioral GO

1. **Energy-matched behavior:** within model/finding/vote and additive-margin
   strata, PAEL separates harmful from non-harmful orbits after conditioning on
   `||J||`, singular values, entropy, both marginals and behavioral MMI; energy
   alone does not.
2. **Layerwise dissociation:** layerwise interaction energy may be present
   before the layer at which reader-oriented PAEL becomes positive.  The
   transition layer is selected on dev independently per architecture and
   applied once on confirmation.
3. **Joint-cell selectivity:** replacing a joint-cell hidden state by its
   additive reconstruction `h10+h01-h00` at the selected layer reduces output
   PAEL and joint-cell clinical loss while preserving the two marginal cells,
   clean-cell polarity, activation norm and claim identity.
4. **Orientation control:** an equal-norm random interaction-subspace patch and
   an energy-matched isospectral orientation do not show the same selective
   correction.
5. **Architecture boundary:** the responsible layer/head/path may differ across
   Huatuo and Hulu; only the input-level orientation phenomenon is required to
   replicate.

The additive hidden-state reconstruction is a causal probe, not method novelty.
No steering, decoder or universal layer claim follows from a positive result.

## 4. Falsification

Stop the orientation-mechanism framing if any hold:

- PAEL is absorbed by interaction norm/spectrum, entropy, marginal sensitivity
  or behavioral synergy;
- PAEL appears only after changing temperature or score calibration;
- layerwise PAEL and energy rise together with no reproducible dissociation;
- additive patching changes marginal cells, activation norm, claim polarity or
  clean cases as much as the joint cell;
- random or isospectral controls correct equally well;
- a human product-control shows comparable render × wording clinical
  interaction.

In that case the credible conclusion is generic product instability under two
individually admitted axes, not a distinct reader-grounded composition
mechanism.


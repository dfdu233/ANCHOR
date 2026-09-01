# C62 — Conservative Residual Refinement (CRR): formula-level audit

**Decision: strict NO-GO as an ICLR-level primitive; do not spend the <=16 GPU budget.**

CRR proposes to replace one coarse visual token `v` by fine crop children

\[
f'_j=f_j-\mu_f+v,\qquad
\mu_f=\sum_j w_jf_j,\qquad \sum_jw_j=1.
\]

The construction is clean, but its advertised novelty and its strongest identity claim do
not survive a computation-level audit.

## 1. What the formula actually is

Let the restriction/coarsening operator be

\[
R(g)=\sum_jw_jg_j.
\]

Then CRR is the unique solution of the standard least-change conservative projection

\[
\min_g\sum_jw_j\|g_j-f_j\|_2^2
\quad\text{s.t.}\quad R(g)=v.
\]

The Lagrange multiplier is the same translation for every child, giving
`g_j=f_j+v-mu_f`.  Thus `R(CRR(v,f))=v` is correct, but it is the textbook affine
projection onto a prescribed-mean hyperplane.

The same algebra is already present in three mature lines:

1. **Wavelet/Laplacian multiresolution:** retain a coarse approximation and represent the
   fine level by zero-mean detail coefficients.  The lifting scheme is the general
   predict/update construction for this decomposition.
2. **Conservative adaptive mesh refinement:** restriction is a weighted child average and
   prolongation is corrected so that restricting the refined cells recovers the parent.
3. **Feature-statistic transplantation / AdaIN:** center source features and add the target
   mean.  CRR is the first-moment-only case of
   `sigma_target * (f-mu_f)/sigma_f + mu_target`.

There is also a recent formula-level cross-domain collision. **Field-Space Attention**
(Witte et al., 2025) explicitly subtracts each parent region's fine-scale mean, stores the
fine tokens as locally zero-mean residuals, and transfers the removed mean to the coarse
scale while proving scale conservation.  CRR changes where the coarse mean is stored, but
not the underlying multiresolution operation.

References:

- Sweldens, *The Lifting Scheme: A Construction of Second Generation Wavelets*, SIAM J.
  Math. Anal. 1998: https://epubs.siam.org/doi/10.1137/S0036141095289051
- Witte et al., *Field-Space Attention for Structure-Preserving Earth System
  Transformers*, 2025: https://arxiv.org/abs/2512.20350
- Conservative AMR restriction/prolongation overview:
  https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node65.html

## 2. The claimed identity is not an identity of the VLM

CRR conserves a **linear token average**.  The next operation in a VLM is nonlinear
softmax attention, which is sensitive to token multiplicity.

Take the claimed zero-detail case, so every refined child equals the parent:
`f'_1=...=f'_J=v`.  Let a query give this token logit `s=q^T k_v`, and let `Z` be the
exponentiated-logit sum of all other tokens.  With the original single token, its attention
contribution is

\[
o_1=\frac{e^s}{Z+e^s}v.
\]

After replacing it by `J` identical children, the total contribution is

\[
o_J=\frac{Je^s}{Z+Je^s}v,
\]

which differs from `o_1` whenever `J>1` and `Z>0`.  Different position embeddings and
child-child self-attention make the mismatch even larger.  Therefore

> `coarsen(refine(v))=v` in feature space does **not** imply `VLM(refine(v))=VLM(v)`;
> even the zero-detail branch is not token-exact identity.

One can add a quadrature weight `log w_j` to each child's attention logit so identical
children have total mass one.  But this becomes weighted-measure attention, requires an
attention-bias hook, and only restores identity for identical keys/values.  For genuine
details, conservation of mean keys and values still does not commute with softmax:

\[
\operatorname{softmax}\!\left(q^TK'_j\right)V'_j
\ne
\operatorname{softmax}\!\left(q^TK_v\right)V_v.
\]

At best the zero-mean condition cancels a first-order term under extra local assumptions;
it is not an exact conservation law for the actual computation.

## 3. The VLM application neighborhood is already dense

- **TokenPacker** uses a coarse-to-fine visual projector and injects local high-resolution
  region keys/values into corresponding coarse point queries:
  https://arxiv.org/abs/2407.02392
- **LLaVA-UHD** divides native-resolution images into local slices, compresses their visual
  tokens, and supplies an explicit spatial schema:
  https://arxiv.org/abs/2403.11703
- **FocusLLaVA** performs vision/text-guided coarse-to-fine visual token acquisition:
  https://arxiv.org/abs/2411.14228
- **DeepStack** injects visual-token features into multiple LLM layers through residual
  connections (NeurIPS 2024):
  https://proceedings.neurips.cc/paper_files/paper/2024/file/29cd7f8331d13ede6dc6d6ef3dfacb70-Paper-Conference.pdf
- **AdaptVision** explicitly implements tool-selected coarse-to-fine crop acquisition:
  https://arxiv.org/abs/2512.03794

CRR is more conservative than these learned fusion modules, but after the exact
multiresolution/AdaIN collision it is an application-specific combination, not a new basic
computation unit.

## 4. The style-cancellation statement is too narrow

If every crop child suffers exactly the same additive feature shift `a`, then

\[
(f_j+a)-\sum_iw_i(f_i+a)=f_j-\mu_f.
\]

This property is correct.  It does not cover the real crop transformation, which changes
scale, receptive field, positional encoding, normalization statistics, and contextual
features nonlinearly.  Earlier repository experiments already rejected a stable common
style/domain direction as the main medical-VLM mechanism, so this algebra cannot revive
that claim without new evidence.

## 5. Why a <=16 smoke test is not warranted

A tiny test could compare CRR against naive crop tokens, but a positive answer would be
uninterpretable unless it also matched:

- visual-token count and attention mass;
- crop resolution and ROI oracle quality;
- naive high-resolution children;
- first-moment AdaIN;
- Laplacian/wavelet residual tokens;
- TokenPacker-style coarse-to-fine injection.

That is already a method-family study, not a lethal test of a novel primitive.  Since the
formula is standard and the stated model-level identity is false, the expected information
gain from occupying the single GPU is lower than continuing the baseline queue or testing a
genuinely different computation.

## Final verdict

CRR is a sensible **engineering baseline** for high-resolution feature injection.  It may
reduce the distribution shift of naive crop tokens, but it is not Songlin-style replacement
of a newly identified faulty computation:

- the update is standard conservative mean projection / multiresolution lifting;
- recent work uses the same zero-mean fine residual construction;
- coarse-to-fine visual-token injection is crowded;
- its strongest `zero-detail identity` fails under softmax attention.

Therefore it should not be promoted to the main idea and should not consume the proposed
GPU smoke-test budget.

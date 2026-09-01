# C68 — Sparse-lesion primitive replacement audit

Date: 2026-08-13  
Scope: training-free medical-VLM hallucination mitigation; no GPU was used.  This
audit deliberately excludes crop/mask, attention bias, VCD-style null images,
PoE, verifier/reranking, high-bit coding, and laterality.  Its question is narrower:

> Does the confirmed small-lesion failure arise from one *wrong computational
> primitive* that can be replaced cleanly in a frozen VLM?

## 1. What is established, and what is not

The natural phenomenon is real.  Within the same VinDr finding, smaller annotated
lesions have lower correct-answer margin.  On the fresh confirmation split, the
Spearman correlations between lesion area and correct margin are `0.323` for
Huatuo and `0.415` for Hulu.  The corresponding development correlations were
`0.232` and `0.475`; all four tests had `p <= 0.0006`.

This does **not** establish that a better scalar pooling rule can recover the
lesion.  The existing Huatuo patch-field confirmation (`n=266`) already found:

| Readout | macro AUROC |
|---|---:|
| final answer margin | 0.7099 |
| multiscale scan | 0.7175 |
| patch maximum | 0.6450 |
| patch top 5% | 0.6783 |
| higher criticism | 0.6909 |
| strong base: final + mean + max + top 5% | 0.7376 |
| strong base + multiscale scan | 0.7416 |

The last increment was `+0.0040`, with 95% CI
`[-0.0194,+0.0273]`.  The current audit therefore treats `+0.02` macro AUROC over
a strong base as the minimum effect-size gate.  These labels had already been
opened for earlier endpoints, so all numbers below are secondary fatal audits,
not blind confirmations.

## 2. Primitive A: low-rank anatomy plus sparse lesion

### Object and exact property

For a claim-conditioned `24 x 24` patch-score field `M`, Principal Component
Pursuit posits

\[
M=L+S,\qquad
(\hat L,\hat S)=\arg\min_{L+S=M}\|L\|_*+\lambda\|S\|_1,
\quad \lambda=1/\sqrt{24}.
\]

The nuclear norm encourages a low-rank global component `L`; the entrywise
one-norm encourages a sparse residual `S`.  The attractive story is that normal
anatomy is globally regular while a focal lesion is sparse.

The exact-recovery theorem in
[Robust PCA](https://arxiv.org/abs/0912.3599) is real but conditional: `L` must be
incoherent and the support of `S` sufficiently sparse and suitably distributed.
It is not a theorem that a lesion in a contextual ViT score map is the recovered
sparse term.  A lesion aligned with the row/column space of `L` can be transferred
between the two components without a clinical label; conversely, ribs, devices,
and sharp anatomy are sparse.  Thus the decomposition is not clinically
identifiable from sparsity and rank alone.

### Zero-GPU fatal result

We ran standard inexact-ALM PCP (at most 150 iterations) on the cached Huatuo
fields.  The direct residual summaries were anti-predictive or weak:

| Individual feature | macro AUROC |
|---|---:|
| positive sparse residual maximum | 0.4820 |
| positive sparse residual mean | 0.5208 |
| residual norm | 0.4737 |
| residual support fraction | 0.4860 |
| negative residual maximum | 0.5679 |

Using the identical development/confirmation fit for this audit, the strong base
was `0.7325`.  Adding positive-residual maximum gave `0.7155` (`-0.0170`), adding
positive-residual mean gave `0.7281` (`-0.0044`), and adding all residual summaries
gave `0.7206` (`-0.0119`).  Because every increment is negative, bootstrap
precision cannot rescue the frozen `+0.02` effect-size gate.

### 2024--2026 collision neighbourhood

Low-rank/sparse manipulation is already an active architecture/compression
primitive: [SoLA (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/33923)
uses activation sparsity and low-rank decomposition; [Frequency-Aware Token
Reduction (NeurIPS 2025)](https://papers.neurips.cc/paper_files/paper/2025/hash/30e15e5941ae0cdab7ef58cc8d59a4ca-Abstract-Conference.html)
preserves high-frequency tokens while compressing low-frequency content; and
[SparseVLM (ICML 2025)](https://proceedings.mlr.press/v267/zhang25s.html) is a
training-free visual-token sparsification/recycling method.  None is exactly this
PCP score-map test, but a medical setting does not create a new mathematical
primitive, and the premise fails locally.

**Verdict: NO-GO.**

## 3. Primitive B: robust location instead of ordinary pooling

### Object and a no-free-lunch property

This family replaces an ordinary spatial mean with a median, trimmed mean, Huber
M-estimator, or median-of-means (MoM).  It computes *robust location*, not an
attention map or a low-rank decomposition.

Let `n` patch scores contain a lesion in `k<n/2` entries.  Then:

* the sample median can remain exactly unchanged even if all `k` lesion scores
  are sent to `+infinity`;
* an upper `alpha`-trimmed mean deletes the lesion exactly whenever
  `k <= alpha n` and those scores occupy the trimmed tail;
* with Huber score function `psi_c`, the lesion's total first-order contribution
  is bounded by `k c / n`;
* MoM ignores a component that changes fewer than half of its groups, while random
  grouping discards the lesion's spatial continuity.

That is the central incompatibility: a high-breakdown estimator is designed to
treat a sparse, extreme component as contamination.  Without a second variable
that distinguishes a true sparse lesion from a false sparse peak, robustness to
the latter and sensitivity to the former are the same statistical direction.
Adding such a selector returns to a scan, mask, or verifier.

### Zero-GPU fatal result

The cached Huatuo confirmation produced:

| Statistic | individual macro AUROC | strong-base delta |
|---|---:|---:|
| median | 0.7036 | +0.0024 |
| 10% trimmed mean | 0.7131 | +0.0004 |
| Huber location | 0.7111 | +0.0036 |
| MoM, 16 groups | 0.6818 | +0.0063 |
| interquartile range | 0.4792 | -0.0016 |

The best increment is `+0.0063`, less than one third of the frozen gate.  There is
therefore no reason to allocate a GPU test.

### 2024--2026 collision neighbourhood

[Integral Transformer (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.118/)
already changes the aggregation of sampled attention-logit signals to denoise
attention without discarding all low-mass information.  More broadly,
[MultiMax (ICML 2024)](https://proceedings.mlr.press/v235/zhou24g.html) explicitly
studies the sparsity-versus-multimodality trade-off of softmax.  Robust spatial
location would still need a lesion-identifying mechanism absent from the frozen
score field; its mathematical robustness alone points in the wrong direction.

**Verdict: NO-GO.**

## 4. Primitive C: change the attention algebra

Three genuinely different algebras were considered:

\[
A_{sig}=\frac{1}{\sqrt n}\sum_i\sigma(s_i+b)v_i,
\qquad
A_{cov}=\frac{1}{\sqrt n}\sum_i(s_i-\bar s)(v_i-\bar v),
\qquad
A_{trop}=\max_i(s_i+v_i).
\]

They do have precise and attractive properties:

* **independent sigmoid evidence:**
  `partial sigma(s_i)/partial s_j = 0` for `i != j`, unlike softmax's negative
  cross-coupling `-a_i a_j`;
* **signed centered covariance:** invariant to a common score shift
  `s_i -> s_i+c` and common value shift `v_i -> v_i+d`;
* **tropical/max-plus aggregation:** idempotent under duplicated evidence,
  `max(x,x)=x`, so duplication cannot manufacture additional mass.

The fatal issue is *frozen-model compatibility*.  For identical scores `s_i=s`
and values `v_i=v`, vanilla softmax returns exactly `v`, whereas sigmoid returns
an `n`- and `s`-dependent multiple of `v`, centered covariance returns zero, and
max-plus returns an object in a different algebra.  No setting-independent scalar
can make any replacement equal to vanilla attention for every sequence length,
score, and value.  Consequently an inference-time swap changes the operating
point of every pretrained residual block.  A downstream gain would not identify
recovered clinical evidence; it would be uncontrolled architecture surgery.

This is also a direct collision zone:

* [Theory, Analysis, and Best Practices for Sigmoid Self-Attention
  (ICLR 2025)](https://openreview.net/forum?id=Zhdhg6n2OG) develops and trains
  sigmoid attention, including the sequence-length-dependent bias needed for
  stability.
* [From Attention to Activation (ICLR 2025)](https://openreview.net/forum?id=IjduZQK8gM)
  introduces softmax-1 to change softmax's sink behaviour.
* [MultiMax (ICML 2024)](https://proceedings.mlr.press/v235/zhou24g.html) replaces
  softmax with a piecewise mapping that preserves sparse multimodality.
* [Integral Transformer (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.118/)
  and signed/differential variants occupy negative or denoised aggregation.
* [Tropical Attention (2025)](https://arxiv.org/abs/2505.17190) directly replaces
  softmax with a max-plus operator.

The available positive work trains or calibrates the new algebra.  That is not a
faithful training-free edit of the present models.

**Verdict: NO-GO at the formula/collision gate; no GPU test.**

## 5. Primitive D: preserve subspace volume rather than salience

A fourth independent object is subset volume:

\[
S^*=\arg\max_{|S|=m}\log\det(K_S+\epsilon I).
\]

The determinant vanishes when the retained vectors are linearly dependent, so a
DPP or leverage-score sampler preserves diverse directions.  This would help only
if a small lesion creates a novel token direction.  A clinically important token
inside the anatomy span has no volume novelty, while an artifact can be maximally
novel; diversity is not clinical relevance.

The route is also directly occupied by [CDPruner
(2025)](https://arxiv.org/abs/2506.10967), a training-free, model-agnostic MLLM
token pruner that explicitly maximizes instruction-conditioned DPP diversity.
It additionally violates this audit's exclusion of token masks/selection.

**Verdict: direct collision and out of scope; no experiment.**

## 6. Why group testing is not advanced as another candidate

Group testing was audited in parallel and is not duplicated here.  Its core
formula-level boundary is short: linearly pooling patches with matrix `A`, then
using a linear readout `u`, yields `u^T A X = (A^T u)^T X`, exactly a patch
reweighting.  Nonlinear group-testing recovery requires an OR-like calibrated
patch detector, which a frozen VLM does not provide; assuming it would assume the
hard part and turn the method into a detector/verifier.

## 7. Consolidated decision

| Primitive | Truly different object? | Exact useful property | Fatal problem | Decision |
|---|---:|---|---|---|
| low-rank + sparse PCP | yes | exact recovery under incoherence/support assumptions | anatomy/lesion decomposition is not identifiable; negative cached increment | NO-GO |
| robust location | yes | high breakdown / bounded influence | by construction suppresses rare lesion evidence together with rare noise | NO-GO |
| sigmoid/covariance/tropical attention | yes | independence / shift invariance / idempotence | no identity-preserving swap in a pretrained softmax network; direct collisions | NO-GO |
| DPP/subspace volume | yes | preserves linear diversity | diversity is not clinical truth; CDPruner collision; token-selection exclusion | NO-GO |

The bounded conclusion is important:

> Sparse-lesion dilution is a reliable measurement fact, but it is not evidence
> that the frozen VLM merely uses the wrong pooling primitive.  On the cached
> interface, candidate primitives either suppress the very sparse signal we want,
> require assumptions violated by contextual visual tokens, collide directly with
> recent work, or cannot replace pretrained softmax while preserving its operating
> point.

The next legitimate search should therefore change the *semantic contract* of
visual evidence—what information may overwrite what—not enumerate another scalar
patch statistic.  Any next candidate must first state an identity-preserving
operator, its clinical noninterference property, and a counterexample showing it
is not merely masking, reweighting, or verification.

## 8. Provenance

Inputs:

* `corrected_runs/daylong_idea_search_v1/patch_scores_huatuo_v1/patch_scores.npz`
* `corrected_runs/vindr_v2/hidden_confirmation_huatuo_recoverability_v1/metadata.jsonl`
* `corrected_runs/evidence_addressability_gate_v2/hidden_fresh_huatuo_v2/metadata.jsonl`

Related frozen evidence:

* `docs/daylong_idea_search/l1_sparse_lesion_boundary.md`
* `corrected_runs/daylong_idea_search_v1/sparse_patch_scan_huatuo_v1.json`
* `docs/daylong_idea_search/C67_SPARSE_OPERATOR_TRIPLE_AUDIT.md`


# Alignment-Induced Style Contraction

## Frozen mechanism question

Does correct medical image--text alignment, rather than medical language
supervision or the image/text marginals alone, causally contract a generative
VLM's sensitivity to acquisition-style perturbations?

The observed checkpoint comparison motivates but does not answer this
question: HuatuoGPT-Vision-7B has lower normalized evidence drift than its
exact Qwen2.5-VL-7B base under six fixed PubMedVision-derived CXR transforms,
but the two checkpoints differ in their entire post-training histories.

## Marginal-matched control and population target

Let \(P\) be the population distribution over image, question, and complete
answer \((X,Q,Y)\). The matched branch trains on \(P\). The ideal control
re-draws the image independently of \((Q,Y)\), giving

\[
P_\perp(X,Q,Y)=P(X)P(Q,Y),
\]

The implemented finite-sample control uses one group-deranged image
permutation. It preserves exactly the same finite image and text multisets,
training steps, model initialization, optimizer, and batch order. It is one
Monte Carlo draw from a marginal-matched randomization distribution; a fixed
permutation is not itself an algebraically independent empirical joint
distribution.

Assuming total, unnormalized sequence NLL, unrestricted conditional
predictors, and finite conditional entropies, the ideal population Bayes
risks are

\[
\mathcal R_P^\star=H(Y\mid X,Q),
\qquad
\mathcal R_{P_\perp}^\star=H(Y\mid Q).
\]

Therefore,

\[
\boxed{
\mathcal R_{P_\perp}^\star-\mathcal R_P^\star
=I(Y;X\mid Q)
}
\]

For the ideal population control, the proof follows immediately from
\(I(Y;X\mid Q)=H(Y\mid Q)-H(Y\mid X,Q)\). This identity does not apply
directly to mean-token NLL, a restricted merger-only model, finite-step
optimization, or the \(\kappa\) statistic below.

A fixed group derangement preserves empirical marginals but is generally not
an empirical product distribution. Repeated derangements quantify
assignment-design variability; a population-product or mutual-information
interpretation additionally requires exchangeability, eligibility,
convergence, and Bayes-consistency assumptions. The control removes the
answer-loss incentive to use the assigned image under the randomization
model, but it does not make invariant representation learning impossible:
pretraining and initialization can already contain such representations.

This identity does not itself guarantee style invariance. It identifies the
necessary training signal whose removal makes visual-semantic quotient
learning impossible.

## Non-identifiability of invariance from alignment alone

For a concrete witness, suppose \(X=(C,S)\) has nondegenerate style
variation and

\[
Y\perp S\mid C,Q.
\]

Then both \(Z_1=C\) and \(Z_2=(C,S)\) can attain
\(H(Y\mid C,Q)\), while \(Z_1\) is constant over same-content style
orbits and \(Z_2\) is not. Sequence log-loss therefore identifies the Bayes
conditional predictor, not an invariant internal representation.
Consequently, ordinary matched image--text alignment has no population
guarantee of contracting style:

\[
\mathcal R(Z_1)=\mathcal R(Z_2)
\quad\centernot\Longrightarrow\quad
Z_2(C,S)=Z_2(C,S').
\]

An explicit same-content orbit constraint can enforce invariance.
Information minimality is only a possible selection principle; minimizing
\(I(Z;X\mid Q)\) does not by itself guarantee invariance because style may
already be encoded by \(Q\). The controlled experiment below therefore asks
whether the architecture and optimization have a useful *implicit* quotient
bias; it does not assume that alignment alone mathematically entails
invariance.

## Mechanism prediction

For complete-sentence clinical evidence \(e_\theta\), define

\[
\kappa_\theta(x)=
\frac{
\sqrt{\mathbb E_{s\in\mathcal S}
\|e_\theta(T_sx)-e_\theta(x)\|^2}
}{
\|e_\theta(x)-e_\theta(\varnothing)\|
}.
\]

The unique prediction is

\[
\operatorname{median}\kappa_{\rm matched}
<
\operatorname{median}\kappa_{\rm permuted}.
\]

Because both branches see the same image and text marginals, this contrast
cannot be attributed to medical vocabulary exposure, answer prevalence,
source-style frequency, or compute. For this single derangement, a failure of
the inequality rejects only the claim that merger-only alignment under the
implemented training budget is sufficient to form the observed contraction.

## Experimental substrate

- Base model: exact `Qwen2.5-VL-7B-Instruct` parent of the evaluated Huatuo
  checkpoint.
- Source: 2,048 strict-CXR PubMedVision instruction records.
- Trainable component: Qwen visual merger only.
- Training: 250 optimizer steps, seed 42, micro-batch 4, accumulation 2,
  learning rate \(5\times10^{-6}\).
- Matched and permuted branches use identical image/text marginals. The
  permutation has no fixed image pair and no same-PMC-group pair.
- First probe: 16 fixed frontal/non-frontal MIMIC development images, six
  acquisition styles, six clinical concepts, and complete positive/negative
  sentence likelihoods.
- Continuation gate: the matched-minus-permuted paired patient-cluster
  bootstrap interval for median \(\kappa\) must have upper bound below zero.

If the gate passes, generation utility is tested on complete-sentence
MIMIC CE and report outputs. If it fails, the merger-only alignment mechanism
is rejected; the single permitted localization follow-up is the
intermediate semantic layer identified by recent causal studies of visual
instruction tuning.

## Result

The continuation gate failed.

The two branches were trained from the identical Qwen2.5-VL-7B checkpoint on
2,048 strict-CXR records for 250 optimizer steps. Their visual-merger
checkpoints differ (\(\ell_2=0.0885\)), so the comparison is not an accidental
checkpoint identity. On 128 strict-CXR source-held-out records, mean
complete-answer NLL was 1.21636 for the base, 1.21378 for matched training,
and 1.21312 for image-permuted training. The matched-minus-permuted
difference was \(+0.00066\), with source-group cluster-bootstrap 95% CI
\([-0.00020,0.00154]\). Both updates slightly improved NLL over the base,
but correct pairing had no detectable advantage.

The primary style probe used 40 selected frontal MIMIC development images
from 38 patients, six fixed PubMedVision-derived styles, six clinical
concepts, and complete positive/negative sentence likelihoods:

| Quantity | Matched | Image-permuted | Matched minus permuted |
|---|---:|---:|---:|
| Normalized susceptibility \(\kappa\) | 0.3175 | 0.3211 | \(-0.0036\), CI \([-0.0301,0.0113]\) |
| Reusable style variance fraction | 2.129% | 2.221% | \(-0.092\) pp, CI \([-0.394,0.150]\) pp |

Relative to the frozen base, \(\kappa\) changed by \(-0.0115\) for matched
training and \(-0.0079\) for image-permuted training; both intervals included
zero. The corresponding reusable style fractions were 2.129%, 2.221%, and
2.056% for matched, permuted, and base checkpoints.

Therefore, under this architecture, component, budget, and fixed style
operator, correct image--text pairing did **not** cause detectable
acquisition-style contraction. The common small NLL improvement is more
consistent with generic merger adaptation to the image/text marginals than
with learning a pairing-specific visual quotient. This rejects
*merger-only alignment is sufficient*, not the existence of conditional
style priors or the possibility that intermediate-layer abstraction plus an
explicit same-content orbit constraint could remove them.

## Closest-work boundary

| Work | Shared object | Remaining difference |
|---|---|---|
| Visual Instruction Bottleneck Tuning (NeurIPS 2025) | Minimal sufficient VLM representations under shift | Adds a stochastic bottleneck; does not isolate the causal effect of correct image--text pairing on acquisition-style contraction |
| Visual Language Hypothesis (2025 preprint) | Nuisance fibers and semantic quotient | Provides a structural theory; does not perform this matched-marginal generative medical VLM intervention |
| Visual Instruction Tuning Aligns Modalities through Abstraction (2026 preprint) | Training-induced cross-modal geometry | Localizes semantic alignment to intermediate layers; does not measure or explain acquisition-style contraction |
| CLIPCEIL (NeurIPS 2024) | Image--text alignment for DG | Supervised CLIP classification with adapters, not complete-sequence medical generation or a marginal-preserving causal control |
| Moment Alignment (UAI 2025) | DG transfer through moment/derivative alignment | General DG theory; does not identify whether image--text mutual information creates nuisance invariance |

No mechanism-equivalent work was retrieved under the documented searches as
of 2026-07-31. The remaining contribution is a causal measurement and,
only if the probe succeeds, a training principle derived from the measured
boundary—not another inference-time center or reranker.

## Claim ceiling

The first probe can establish only whether correct source image--text pairing
causes lower sensitivity to the fixed style operator in one trainable
component and one exposed development protocol. It cannot establish external
hospital generalization, hallucination mitigation, natural scanner
robustness, or a general effect of medical instruction tuning.

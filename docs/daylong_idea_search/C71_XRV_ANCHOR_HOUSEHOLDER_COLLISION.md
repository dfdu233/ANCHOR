# C71 — XRV local anchors and Householder counterfactual audit

Date: 2026-08-13  
Scope: cached-result, mathematical, and primary-paper collision audit only. No
GPU was used and no baseline process was changed.

## Verdict

The local positive/negative geometry of the frozen XRV specialist contains real
case-level information, but it does **not** define a training-free transport into
a medical VLM. An affine Householder reflection is an exact and elegant way to
swap two anchors inside one representation space. Once it is made capable of
changing the VLM, however, every implementation is one of the following:

1. contrastive expert-logit fusion (RVCD/VCD/CFG family);
2. contrastive activation steering in a norm-preserving parametrization;
3. a learned cross-encoder stitching adapter;
4. counterfactual image construction or patch exchange (CoFE family);
5. output selection/veto if factual and reflected answers are compared.

Therefore the reflection is not a new non-fusion mitigation primitive. The
strict decision is **NO-GO before GPU**. The useful retained fact is narrower:
full 18-dimensional XRV neighbourhood geometry is more informative than a
single disease logit, especially for Huatuo. That is a specialist information
upper bound, not a method for transporting evidence into a frozen VLM.

## 1. What the cached XRV experiment actually establishes

The artifact contains XRV logits for 1,003 VinDr images over 18 findings. On a
held-out confirmation set, adding only the target disease logit to the VLM
margin produced:

| Model | VLM only | + target XRV scalar | gain | image-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Huatuo | 0.7667 | 0.8264 | +0.0598 | [0.0397, 0.0783] |
| Hulu | 0.8606 | 0.8708 | +0.0102 | [0.0038, 0.0168] |

The full 18-dimensional local-neighbour score is

\[
b_c(s)=d(s,\mathcal N_c^-)-d(s,\mathcal N_c^+),
\]

where each distance is the mean distance to the three nearest development
examples with the indicated reader label, within finding `c`. After controlling
for both the native VLM margin and the target XRV scalar, it added:

| Model | VLM + XRV scalar | + 18D local boundary | gain | 95% CI | shuffled-label placebo |
|---|---:|---:|---:|---:|---:|
| Huatuo | 0.8264 | 0.8525 | +0.0261 | [0.0143, 0.0384] | +0.0002 |
| Hulu | 0.8708 | 0.8806 | +0.0098 | [0.0031, 0.0164] | -0.0015 |

Thus there is a reproducible but model-asymmetric fact: nearby specialist
counterexamples carry information not captured by one specialist logit. The
pre-registered `+0.02` two-model gate failed because Hulu gained only `+0.0098`.
Moreover, this is supervised k-nearest-neighbour prediction using labelled
development cases. It says nothing yet about causal disease manipulation or
safe generation.

The one-bit expert veto also missed its gate: it removed 17.43%/17.39% of
Huatuo/Hulu false positives while harming 1.52%/2.33% of true positives, versus
the required at least 20% removal and at most 1% harm. This is evidence against
silently treating the specialist as clinical truth.

A subsequent cache-only reflection audit made the relationship even sharper.
For nearest opposite-class anchors `a_0,a_1`, the signed Householder coordinate
is exactly

\[
u^\top(z-m)
=\frac{\|z-a_0\|^2-\|z-a_1\|^2}{2\|a_1-a_0\|}.
\]

Thus it is a normalized difference of the same two anchor distances, not a new
source of evidence. The numerical implementation satisfied the distance-swap,
affine identity, and involution equalities to at worst `1.42e-14`. Adding this
coordinate reached AUROC `0.8442/0.8788` for Huatuo/Hulu, below the ordinary
kNN boundary's `0.8525/0.8806`. Its bootstrap delta over kNN was
`-0.0064 [-0.0130, 0.0001]` and `-0.0019 [-0.0058, 0.0022]`, respectively.
The exact geometry therefore provides no predictive loophole either.

## 2. The exact reflection geometry

Let `p,n in R^18` be local positive and negative anchors in standardized XRV
space for one finding. Define their midpoint and unit connecting direction:

\[
m=\frac{p+n}{2},\qquad u=\frac{p-n}{\|p-n\|_2}.
\]

Reflection across the perpendicular bisector of the anchor pair is

\[
R_{p,n}(s)
=m+(I-2uu^\top)(s-m)
=s-2u\,u^\top(s-m).
\]

It has four exact properties:

* `R(p)=n` and `R(n)=p`;
* `R(R(s))=s` (it is an involution);
* `||R(s_1)-R(s_2)||=||s_1-s_2||` (it is an isometry);
* only the coordinate `u^T(s-m)` changes sign; every component orthogonal to
  `u` remains fixed.

It is the unique affine isometry that fixes the whole perpendicular-bisector
hyperplane pointwise and swaps `p` with `n`. This is mathematically clean, but
its semantic interpretation is not identified: `p-n` contains every patient,
acquisition, anatomy, comorbidity, and disease difference between the two
images. Nearest-neighbour label opposition does not make this vector a pure
lesion intervention.

Reflection is also not the minimum edit to reach the local boundary. The
orthogonal projection is

\[
P(s)=s-u\,u^\top(s-m),
\]

whereas reflection moves twice as far:

\[
\|R(s)-s\|=2\,|u^\top(s-m)|.
\]

It deliberately extrapolates to the opposite side rather than merely removing
the suspect evidence.

## 3. There is no training-free cross-space reflection

### Route A: construct anchors directly in VLM hidden space

Given hidden anchors `h_p,h_n`, define

\[
v=\frac{h_p-h_n}{\|h_p-h_n\|},\quad
\mu=\frac{h_p+h_n}{2},\quad
h'=h-2vv^\top(h-\mu).
\]

This preserves hidden-state Euclidean norm around `mu` and swaps the two anchor
states. It is nevertheless an instance-dependent **contrastive activation
steering** operation: a vector is obtained from positive-minus-negative
activations and injected into the residual stream. Replacing additive steering
by its Householder form changes the parametrization, not the intervention
channel or correctness source.

For a locally linear claim margin `g_c(h)=w_c^T h+b_c`, the effect is exactly

\[
g_c(h')-g_c(h)=-2(w_c^\top v)\,v^\top(h-\mu).
\]

The sign is helpful only if the VLM gradient direction `w_c` aligns with the
anchor direction and the anchor midpoint matches the VLM decision boundary.
Neither follows from XRV neighbour accuracy. If `w_c` is orthogonal to `v`, the
edit has zero target effect; if the alignment has the wrong sign, it worsens
the claim. Requiring the needed alignment assumes the shared clinical codebook
the method is supposed to discover.

### Route B: transport an XRV reflection through a map

A general transported edit has the form

\[
h'=h+A\{R_{p,n}(s)-s\}
=h-2Au\,u^\top(s-m).
\]

There is no canonical map from the 18-dimensional discriminative XRV logit
space to a model-specific hidden space with thousands of coordinates. If `A`
is fitted from paired images, this is supervised model stitching/ridge/CCA; if
it is fitted per image, specialist-only residual directions are unidentifiable;
if it is random or padded, there is no clinical meaning. Orthogonal Procrustes
does not solve the issue because the spaces do not have equal dimension and,
more importantly, global alignment retains only the specialist component
already predictable from VLM states.

### Route C: realize the reflected point as an image

Producing an image `x'` with `S_XRV(x')=R(S_XRV(x))` requires an inverse or
generator for a many-to-one classifier. A chosen right inverse must invent
anatomy and acquisition details that are absent from 18 logits. Patch exchange
with the positive/negative anchors is counterfactual image construction, the
same mechanism class studied by CoFE. Passing the anchors through the VLM and
contrasting their next-token logits is RVCD.

### Route D: reflect the final distribution

Contrasting the factual and reflected logits is classifier-free or visual
contrastive guidance. Choosing between factual and reflected answers is
reranking/veto. Using only the reflected answer makes the model answer a
counterfactual image rather than the observed patient; it has no same-case risk
guarantee.

## 4. Multi-finding failure is structural

Open-ended medical answers contain several claims. Let `H_u=I-2uu^T` and
`H_v=I-2vv^T` be disease reflections for two findings. In general,

\[
H_uH_v\ne H_vH_u,
\]

unless the directions are orthogonal or collinear. Their product is a rotation,
so changing two findings depends on the arbitrary order of application. XRV is
multilabel and its disease logits are correlated; the required orthogonal
clinical axes are not available. This makes the attractive one-claim geometry
especially unsuitable as a general OE decoder.

## 5. Mechanism-level collision matrix

| Proposed implementation | Closest verified work | Same mechanism? | Remaining delta | Verdict |
|---|---|---:|---|---|
| positive/negative anchor logits added to native logits | RVCD; VCD | yes | anchors selected by XRV and a Householder coefficient | cosmetic |
| expert-conditioned hidden/token direction | Expert-CFG | yes | norm-preserving reflection rather than additive CFG | cosmetic |
| positive-minus-negative hidden anchor vector | activation steering | yes | instance-dependent affine reflection | cosmetic |
| nearest opposite case plus patch exchange | CoFE | yes | inference-only rather than contrastive training | occupied counterfactual channel |
| learned XRV-to-VLM map | model stitching / specialist-encoder fusion | yes | Householder source coordinate | trained adapter, out of scope |

Primary-source checks:

* [RVCD (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.430/)
  retrieves explicit positive/negative concept images and adjusts next-token
  logits with their contrasts. Its update is
  `z_o + alpha sum_i(z_o-z_i^-) + beta sum_j(z_j^+-z_o)`, exactly the logit
  route above.
* [VCD (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_Decoding_CVPR_2024_paper.html)
  establishes the visual-logit contrast family.
* [Expert-CFG (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Liang_Uncertainty-Driven_Expert_Control_Enhancing_the_Reliability_of_Medical_Vision-Language_Models_ICCV_2025_paper.html)
  already applies expert-informed classifier-free guidance to medical-VLM token
  embeddings.
* [CoFE (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05958.pdf)
  selects semantically similar opposite-diagnosis cases and iteratively swaps
  corresponding patches to construct a counterfactual, then trains factual/
  counterfactual representations and a learnable prompt.
* [Activation Steering (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8c3262a4c965ba9888f120d4f9e13478-Abstract-Conference.html)
  computes directions from activation differences and injects them into hidden
  states. Householder reflection is a standard orthogonal realization of the
  same intervention class, not a new evidence source.

No mechanism-equivalent-free sixth primitive was retrieved or derived under
the documented routes.

## 6. Fatal controls required if this family is nevertheless piloted

These controls would test safety/collision; passing them would not by itself
establish novelty.

1. true opposite-label anchors versus label-shuffled anchors matched in
   distance;
2. opposite-label pairs versus within-label pairs matched in distance;
3. clinical direction versus random Householder vectors matched in edit norm;
4. target-claim response versus every off-claim response to reject a global
   positive/negative language shift;
5. clear TP/TN no-harm and fixed claim count/length;
6. leave-one-patient-out anchors and stability over neighbour count `k`;
7. VLM-gradient alignment `w_c^T v` measured before any outcome test;
8. equal-norm additive steering and RVCD/Expert-CFG as direct comparators;
9. permutation of multi-finding reflection order; material order dependence is
   immediate failure for OE;
10. patient/image-cluster bootstrap on a held-out test split.

A meaningful gate would require both models to improve positive-content error,
not merely AUROC, while preserving true positives, claim count, length, and all
off-claim margins. Given the algebraic collision and missing cross-space map,
running this gate would spend GPU on a method already classified by formula.

## 7. Bounded conclusion

Local specialist counterexamples answer a useful diagnostic question:

> Cases with the same target disease score can still differ in the surrounding
> 18-dimensional disease configuration, and that neighbourhood can predict the
> reader label.

They do not answer the intervention question:

> Which direction inside a particular VLM may be changed so that only the
> target clinical claim becomes more truthful?

Householder reflection solves the first space's geometry, not the second
space's semantics. Without a learned or externally certified bridge it is
undefined; with such a bridge it is ordinary expert fusion or steering.
Therefore this branch is closed as a new training-free, non-fusion mitigation
method. The cached 18D geometry may be retained only as an information upper
bound or as a baseline if the project later relaxes the ban on specialist
fusion.

## Provenance

* `corrected_runs/xrv_visual_increment_v1/xrv_logits.npz`
* `corrected_runs/xrv_visual_increment_v1/result.json`
* `corrected_runs/xrv_visual_increment_v1/one_bit_veto.json`
* `corrected_runs/daylong_idea_search_v1/xrv_counterexample_geometry_v1/result.json`
* `corrected_runs/daylong_idea_search_v1/xrv_householder_reflection_v1/result.json`
* `anchor/corrected_sgta/screen_xrv_counterexample_geometry_v1.py`
* `anchor/corrected_sgta/audit_xrv_householder_reflection_v1.py`
* related local audits: C54, C56, C59, C65, C68

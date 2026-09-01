# C67 — Contralateral Group Control

Date: 2026-08-13  
Decision: **strict NO-GO as an ICLR-level hallucination method**

## Candidate

For paired anatomy, replace a target region by the registered contralateral
region from the same patient. Let `P_G x` denote the resulting locally
symmetrized image. The proposed decoding update was

```text
z'(v) = z(v | x) + alpha [z(v | x) - z(v | P_G x)].
```

An equivalent feature formulation decomposes registered paired-organ tokens
into even and odd components `(h + rho(g)h)/2` and `(h - rho(g)h)/2`, then uses
the odd component for unilateral focal claims. A sham replacement on the
opposite side was designed to separate target evidence removal from paste
artifacts.

## Why it is natural but not new

The patient supplies an internal control, so acquisition style and much normal
anatomy are shared. This is clinically meaningful for unilateral nodules,
pneumothorax, breast lesions, retinal findings and dental abnormalities.
However, every irreducible computation is already occupied:

1. `z(x) + alpha [z(x)-z(P_Gx)]` is exactly visual contrastive decoding with a
   structured negative view; changing Gaussian noise to a contralateral view
   does not change the decoding primitive.
2. Pixel-space `x-P_Gx` is classic contralateral subtraction. It was evaluated
   for chest-radiograph nodule detection more than twenty years ago.
3. When `P_G=(I+g)/2`, it is the standard Reynolds projection for the reflection
   group; even/odd features are the standard two irreducible representations of
   `Z_2`, not a new mathematical result.
4. Zhao et al., *Contralaterally Enhanced Networks for Thoracic Disease
   Detection* (2020), already extract corresponding contralateral patches and
   fuse target/partner features through additive and subtractive branches.
5. OralGPT-Plus (CVPR 2026) already exposes contralateral comparison as the
   `Mirror-In` visual tool for a medical VLM.

Local one-sided copying is also not itself a group projection: selecting a
suspected side breaks reflection equivariance and requires a locator.

## Exact algebraic equivalences

Let `g` be left--right reflection and let `rho(g)` be its action after spatial
registration.  The two projectors

```text
P_even = (I + rho(g))/2,   P_odd = (I - rho(g))/2
```

are the standard Reynolds/isotypic projectors of the two irreducible
representations of `Z_2`.  Thus an ``even/odd visual-token decomposition'' is
not merely similar to group symmetrization; it is exactly the textbook
decomposition.  For a laterality-invariant claim such as "a nodule is
present", the signed odd component is also insufficient by itself: reflection
changes its sign, so a valid presence score must use an invariant nonlinear
summary such as `abs(P_odd x)`, its norm, or a learned fusion.  Those are
respectively classical contralateral residual magnitude and the operation
already learned by bilateral/symmetry-aware detectors.

The proposed decoder can likewise be rewritten without approximation as

```text
z'(x) = (1 + alpha) z(x) - alpha z(P_G x),
```

which is VCD's logit rule with `P_G x` in place of the Gaussian-distorted
view.  A more clinically meaningful negative view does not create a new
decoding operator.

There is an additional representation issue.  A frozen ViT with absolute
position embeddings, or a decoder with RoPE, is not guaranteed to satisfy
`h(gx)=rho(g)h(x)`.  Consequently, reflecting or averaging already-contextual
tokens cannot be called an exact group projection unless that intertwining
law is first established.  Whole-input frame averaging can enforce an output
symmetry, but that is standard test-time symmetrization and can increase risk
when the medical data distribution is only approximately, rather than truly,
reflection invariant (the heart, lung lobes, projection and devices are
obvious counterexamples).

## Why the current oracle sham is not a causal proof

The upper-bound screen compares replacing the target box by its contralateral
partner against a sham that pastes the target patch onto the opposite side
while leaving the original target intact.  This is a useful artifact control,
but the two edits are not symmetric:

* target replacement removes a candidate lesion;
* sham replacement can duplicate that lesion and changes visual multiplicity.

Hence `sham_score - target_replaced_score > 0` can be produced by an added
second lesion, by a multiplicity response, or by different registration
errors; it does not by itself identify patient-specific healthy evidence.  A
valid causal assay would additionally need matched healthy bilateral cases,
bidirectional side exchange, target-preserving paste controls, and an
interaction showing that the effect is specific to unilateral focal truth.
Passing that assay could establish a useful clinical substrate, but it would
still not undo the formula-level method collision above.

## Execution decision

A CPU oracle-box upper-bound screen was implemented at
`anchor/corrected_sgta/screen_contralateral_control_upper_bound_v1.py`. It was
stopped before completion once the formula-level collision was independently
confirmed, because a positive result could only justify a strong clinical
baseline, not the requested novel primitive. The paper baseline GPU process
was never interrupted.

## Boundary

Keep contralateral subtraction/Mirror-In as a future baseline for paired-organ
tasks. Do not rebrand it as group-theoretic decoding, patient-specific VCD, or
small-model collaboration. Re-open only if a new operation has a property not
expressible by structured-view contrast, additive/subtractive bilateral fusion,
visual prompting, or tool-based reinspection.

# Provenance Two-Plane Binding: no-leak CPU contract v3.2

**Freeze date:** 2026-08-03  
**Authorization:** CPU feasibility only; GPU NO-GO.  
**Supersedes:** v3.1, whose precision assignment could depend on target VinDr
reader votes.

## Problem and bounded novelty

The question is whether medical adaptation binds an already visible source
coordinate into two separable quantities: a claim-mean prior and a
claim-precision prior. An empty visual cue learning a disease association is
already covered by randomized shortcut, RaVL, ShorT and visual-backdoor work.
The surviving delta requires all of the following: complementary sign reversal,
two-plane separation, evidence-dependent rather than unconditional effects,
downstream rebinding of a shared source representation, and a natural medical
source bridge.

Reviewer-style verdict: **Accept with Revisions as a conditional mechanism
branch, not as the current paper mainline.** Collision with RaVL/ShorT/backdoor
work and the present two-claim/one-parent/human-admission bottleneck are two
MAJOR flaws. Neither is yet a data refutation, but both must be repaired before
GPU expansion. The strongest prospective axes are robustness and unification;
there is no efficiency contribution.

## Strict separation of training and evaluation

### Source continuation plane

Training assignment uses only Huatuo PubMedVision `alignment_generic`
`source_train` examples, grouped by unique PMC image. It may consume only a
frozen, human-admitted source-text extractor and source-side nuisance fields.
It must never consume VinDr images, labels, reader votes, model outputs or
target margins.

For each admitted claim `c`, source assistant text defines:

- mean label `y_mu`: explicit positive `+1`, explicit negative `-1`; uncertain
  and unmentioned are excluded from this source contrast;
- precision label `y_kappa`: explicit positive or negative `+1`, explicit
  uncertain `-1`; unmentioned is excluded.

Thus `s_mu` is randomized to explicit source polarity and `s_kappa` to source
linguistic definiteness. It is prohibited to call `s_kappa` reader agreement,
visual clarity or diagnostic certainty.

### External medical evaluation plane

VinDr is untouched external evaluation. Its fixed R8/R9/R10 votes are reported
as a reader-support distribution and effect modifier only. They are never an
assignment variable, training label, checkpoint selector or claim selector.
Any visual-clarity claim additionally requires an independent admitted clarity
coordinate.

## Current source substrate

The authoritative v3.5 source artifact contains 772 unique generic
source-train PMC image units. Automatic count eligibility currently retains
only `consolidation` and `pleural_effusion`. Both have at least 20 explicit
positive, 20 explicit negative, 20 uncertain and 20 definite source-train
examples. Two claims suffice only for a two-row Hadamard discovery organism;
they do not support a broad natural-source or oral claim.

The 480-row blinded source review must establish positive precision at least
0.90 and inverse-probability-weighted macro-F1 at least 0.80 separately for the
generic-alignment and instruction-tuning audit domains. No alias repair or
claim replacement is allowed after review.

## Randomized 2 x 2 source assignment

Each unique source image receives `(s_mu,s_kappa) in {-1,+1}^2`. The four cue
combinations have equal counts. Assignments must balance processor backend,
source stage/split, PMC group, archive, response-length stratum, token mass and
all frozen nonclinical nuisance fields. Cross-plane source contrasts are
bounded by 0.05 in normalized absolute value.

Discovery freezes the two Hadamard pairings:

1. `r_mu=(+1,+1)`, `r_kappa=(+1,-1)`;
2. `r_mu=(+1,-1)`, `r_kappa=(+1,+1)`.

For every registered seed, the optimizer is rerun on source data. `plus` uses
the optimized assignment, `minus` is its exact two-bit complement, and `zero`
is independently optimized to make every mean, precision and cross-plane
contrast near zero under identical balance tolerances. The complete source
examples, text, order, loss weights and token mass remain identical across
arms; only the cue assignment changes.

Automatic extractor labels may establish optimization feasibility only. They
cannot authorize cue rendering, training or target scoring.

## Processor and cue contract

The conditional processor Gate A remains binding: cache one 224x224 clinical
ROI, paste the identical ROI into a neutral/A/B 280x280 canvas with a
patch-aligned 28-pixel frame, and pin `use_fast=False`. All 100 visual tokens
are attended. The intervention is an unmasked visual-token frame, never masked
padding or metadata. Neutral framed input is the parent/control. Clinical
interior processor tensors must be bitwise identical across cue families.

## Falsifiable response law

For held-out cue exemplars and target images, estimate after exact-parent
subtraction:

```text
                 output polarity q     output commitment h
source mean cue       M_q_mu                 M_h_mu
source precision cue  M_q_k                  M_h_k
```

The two-plane mechanism requires signed diagonal recovery across both Hadamard
pairings, diagonal dominance over off-diagonal and global-affirmative scores,
plus/minus antisymmetry, and zero-arm equivalence. If only `M_q_mu` survives,
the result is ordinary multi-label shortcut learning. If only generic
definiteness moves, it is answer style. Equal cue effects across evidence
levels, fixed target-text triggering, or an effect fully explained by parent
margin falsifies the prior account.

## Mechanism and natural bridge gates

Layerwise work must separate cue availability, signed cue-to-claim binding and
commitment crossing. Cross-child patching must show that a shared early cue
representation retains the recipient child's downstream sign. Norm-restored
binding removal must erase crossover while preserving cue decoding, claim
identity and clear-case clinical performance.

Natural Huatuo claims additionally require a source coordinate decoded without
target labels, coupling aligned with a separately human-admitted source
fingerprint, the same mediation signature, causal erase/patch, and a second
source or exact-parent family. Without this bridge, the endpoint is only a
controlled generative-shortcut model organism and oral framing is prohibited.

## Staged decision

1. CPU assignment feasibility on automatic labels: may proceed, but always
   emits `gpu_authorized=false`.
2. Blinded source-extractor admission and cue-family clinical-null admission:
   external prerequisites.
3. CPU power simulation distinguishing additive evidence-gated prior from
   trigger and margin artifact.
4. Only all passes authorize one 3B, three-seed discovery triplet. Seven-B,
   five-seed and second-family work require the complete mechanism signature.


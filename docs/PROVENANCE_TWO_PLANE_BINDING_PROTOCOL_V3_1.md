# Provenance Two-Plane Binding: CPU falsification contract v3.1

> **Superseded for execution.** This draft incorrectly allowed the wording
> `s_kappa` to depend on VinDr unanimous/disputed reader support during source
> assignment, while simultaneously describing PubMedVision continuation.
> That would either mix incompatible datasets or leak target reader truth into
> training. The corrected no-leak contract is
> `docs/PROVENANCE_TWO_PLANE_BINDING_PROTOCOL_V3_2.md`. This file is retained
> as an audit trail and must not authorize GPU work.

**Freeze date:** 2026-08-03  
**Final decision (2026-08-03): CPU operator/power PASS; GPU NO-GO.** The
repaired PubMedVision source audit retained only 2/25 claims, below the frozen
8-claim natural bridge. No positive synthetic result may override this gate.  
**Novelty boundary:** an empty cue learning a disease label is not a
contribution. It is already covered by randomized shortcuts, RaVL and visual
backdoors. The cue is used only as an identifiable model organism.

## Question

Does medical adaptation rebind an already available provenance coordinate into
two separable clinical quantities--a claim prior and an evidence-precision
prior--and do those priors cause reader-distribution-incongruent commitment at
the visual evidence boundary?

For claim `c`, image `x` and provenance `s`, the falsifiable approximation is

```text
polarity_q(c,x,s)   = visual_likelihood(c,x) + source_mean_prior(c,s)
commitment_h(c,x,s) = evidence_clarity(c,x)  + source_precision(c,s)
```

This is not licensed by a generic answer change. The two source effects must be
separable, reverse under complementary training, survive unseen cue exemplars,
and mediate errors in the predicted reader-vote strata.

## Controlled 2 x 2 organism

Use only the fixed VinDr R8/R9/R10 panel and an exact public VLM parent. The
assignment unit is a unique image. Seven source-qualified findings are frozen
before randomization: aortic enlargement, cardiomegaly, pleural thickening,
pulmonary fibrosis, nodule/mass, lung opacity and pleural effusion. Eligibility
may only decrease through a preregistered source-only feasibility rule; it may
not change after a fingerprint is drawn.

Each image receives two independent provenance bits from train/test-disjoint
cue families:

- `s_mu`: correlated with present-versus-absent polarity while balancing reader
  agreement within polarity;
- `s_kappa`: correlated with unanimous-versus-disputed reader support while
  balancing polarity within agreement.

The four cue combinations have matched area, histogram, spectrum, edge energy,
seam and compression. They never contain anatomy, text, markers or donor
pixels. All variants use identical canvas geometry, so clinical pixels undergo
the same rescaling. Processor-tensor equality is required between cue families
inside a frozen eroded clinical mask; any interpolation bleed fails Gate A.
The processor audit fixes the construction to a cached 224x224 clinical ROI in
a 280x280 canvas with a 28-pixel frame and `use_fast=False` (or an exactly
equivalent all-dimensions-multiple-of-28 geometry). The neutral framed canvas,
not the unframed image, is the parent/control input. All 100 visual tokens are
attended; the intervention is an unmasked visual-token frame, not padding or
metadata. The audit artifact is
`corrected_runs/ppi_processor_gate/GATE_A_DECISION.md`.

For each training seed, the same optimizer constructs `plus`, `minus` and
`zero` assignments targeting the positive, complementary and zero 2-plane
fingerprints. Example IDs, targets, row order, token mass, initialization,
optimizer state, dropout and RNG streams are byte-identical across arms.

At least two approximately orthogonal/Hadamard claim fingerprints are required
in discovery. Every exact randomization replicate reruns the assignment
optimizer over the frozen admissible sign-vector set `R*`.

## Required response matrix

Render every held-out image under all four cue combinations. Estimate a
claim-wise response matrix after parent subtraction:

```text
                 output polarity q     output commitment h
mean cue s_mu          M_q_mu                 M_h_mu
precision cue s_k      M_q_k                  M_h_k
```

The two-plane claim requires signed diagonal recovery across fingerprints:

- `M_q_mu` follows the randomized mean fingerprint;
- `M_h_k` follows the randomized precision fingerprint after controlling
  polarity and parent margin;
- diagonal response exceeds off-diagonal response and a global affirmative
  language score;
- `0.5*(plus-minus)` carries the fingerprint, while
  `0.5*(plus+minus)` is equivalent to the zero-arm common drift.

If only `M_q_mu` survives, the result is ordinary multi-label shortcut learning
and the branch stops. If only generic definiteness changes, it is an answer-
style shortcut and the branch stops.

## Evidence-boundary law

Reader votes are an external support distribution, not literal visual clarity.
The primary interaction is therefore stated conservatively: the randomized
source effect must be largest for 1/3 and 2/3 cases, yield false-positive
crossings on 0/3, and leave correctly directed clear 3/3 cases resistant. It
must survive matching on parent margin, finding, view, positive-claim count,
box extent where available, and answer length.

Before confirmatory training, one independent clarity coordinate is required:
either a frozen blinded-radiologist conspicuity score or a preregistered
image-only score trained and calibrated on an image-disjoint development set.
The paper may not rename reader disagreement as visual ambiguity.

Unconditional cue-triggered target text, equal effects at every evidence level,
or an interaction explained by smaller baseline margins are fatal outcomes.

## Mechanism beyond behavior

Layerwise analysis separates:

1. `availability`: cue identity is decodable;
2. `binding`: cue coordinates acquire signed claim coupling after adaptation;
3. `commitment`: the coupling crosses the absent/uncertain/present or
   tentative/definite language boundary.

The decisive cross-child patch transfers an early cue representation from a
plus child into a minus child. A shared representation with downstream
rebinding predicts that the recipient child's sign wins. Norm-restored removal
of only the cue-to-claim binding component must erase the randomized crossover
while preserving cue availability, clinical polarity on clear cases, claim
identity and activation norm. Random, same-norm, unrelated-claim and sham-cue
directions are mandatory controls.

Matched LoRA updates are audited for a stable antisymmetric low-rank structure:
`DeltaW_plus-DeltaW_minus` should align with an outer product between the shared
provenance coordinate and claim/commitment readouts. A pure cue detector or
diffuse update is not the proposed mechanism.

## Natural bridge

A controlled child proves capacity, not the cause of Huatuo behavior. Natural
claims require all of the following:

- source-only occurrence and certainty fingerprints stable under PMC/article-
  held-out splits;
- a natural checkpoint source coordinate decoded without using target labels or
  model answers;
- source-to-claim and source-to-certainty coupling aligned with the frozen
  source fingerprints;
- the same availability--binding--commitment mediation signature as controlled
  children;
- causal erase/patch reducing source-conditioned 0/3 errors and 1/3--2/3
  overcommitment without harming 3/3 pathology evidence;
- replication in a second source or exact-parent family.

Without this bridge, all natural results are labeled triangulation and the oral
claim is prohibited.

## Mitigation and OE

The mechanism-matched intervention removes only the learned source-conditioned
mean/precision binding, not the visual source representation and not the claim
direction. For OE/report generation, evaluate at fixed positive-claim count
`K` or matched claim coverage. Report claim substitutions, certainty changes,
location/severity errors, length, refusal, hedging and omission separately.

The method advances only if source-conditioned hallucination falls by at least
20%, reader-distribution Brier improves by at least 5% relative, clear-case
performance falls by at most 1 pp, and omission does not increase. Shortening,
blanket hedging, uniform negativity or refusal is failure.

## Immediate CPU gates

1. **Assignment gate:** exact `+/-/0` complement construction is feasible for
   frozen images and at least two orthogonal fingerprints, with all nuisance
   balance checks passing.
2. **Power gate:** simulation distinguishes evidence-gated additive prior from
   unconditional trigger and margin artifact using seed as the experimental
   unit.
3. **Natural fingerprint gate:** source mean and certainty fingerprints are
   stable on article-held-out data and contain enough claims for a second-source
   bridge.
4. **Processor gate (passed conditionally):** the cue survives the actual Qwen
   processor and the 256 interior patch rows are exactly invariant across cue
   families under the frozen fixed-frame construction. Any unframed/framed
   parent comparison or processor-backend mixture reopens this gate and fails.

Failure of any gate stops GPU work. The natural fingerprint gate has failed, so
no training is authorized in this version. In a future protocol with a newly
admitted independent natural substrate, passing all four would authorize only a 3B,
three-seed, two-fingerprint discovery screen. A 7B five-seed confirmation is
authorized only after the mechanism signature, not merely behavioral
crossover, is recovered.

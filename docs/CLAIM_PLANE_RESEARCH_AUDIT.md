# Evidence Is a Plane, Not a Line

Research decision: 2026-07-31

> **Priority update.** Claim Plane remains the exact claim-score coordinate
> system, but it is not itself the headline. The generic
> support-to-language-uncertainty story now has direct 2026 collisions and is
> retained only as an empirical gate. The live evaluation question is whether
> medical OE benchmarks incorrectly treat uncertainty as a polarity-free third
> claim and thereby let hedging hide fabricated content. See
> `UNCERTAINTY_REFERENT_RESEARCH_AUDIT.md`.

## Executive synthesis

The original **Commitment-Bounded Decoding (CBD)** mechanism is rejected.  Its
one-dimensional evidence rule

\[
z_+=e,\qquad z_-=-e,\qquad z_?=\tau-|e|
\]

does not merely add an uncertain state.  It constrains the two-dimensional
three-state simplex to a one-dimensional V-shaped curve.  In two real-image
diagnostic probes, CBD substantially harmed clear-case performance and created
fabrications and omissions.  The hypothesised monotonic late commitment-bias
growth also ran in the opposite direction.

The surviving candidate is a **new problem / representation paper**, not yet a
mitigation method:

> A clinical claim has two independent coordinates: **polarity** (support
> versus refutation) and **commitment** (definite evidence versus
> undetermined).  Hallucination mitigation is valid only when it corrects the
> former without obtaining an apparent gain by collapsing the latter or by
> reducing claim coverage.

This candidate is named the **Claim Plane**.  It is mathematically exact,
generalises from controlled VQA to sequence-scored OE claims, and survived a
small cross-model geometry probe.  It has not yet earned the claims that it
predicts reader disagreement, improves OE hallucination, or beats calibrated
margin/abstention.  Those are locked continuation gates.

## Scope and research questions

The review and probes froze three questions:

1. Which medical-VLM hallucination gains are genuine polarity corrections,
   and which are output contraction, uncertainty inflation, or polarity
   transfer?
2. Is there a minimal claim-level representation shared by CE-VQA, OE-VQA,
   and report generation?
3. Can an intervention exploit that representation without increasing
   omission, false negation, report shortening, or refusal?

The scope is limited to image-grounded clinical claims.  Knowledge, treatment,
prognosis, history, and unavailable comparison views remain separate.

## Literature synthesis and collision audit

### What is already crowded

- Training-free visual contrast is established by
  [VCD (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_Decoding_CVPR_2024_paper.html),
  while [SECOND (ICML 2025)](https://openreview.net/forum?id=SbyrpBNNs4)
  makes the contrast spatially selective and coarse-to-fine.  A new null,
  mask, or layer selector is not enough novelty.
- Layer-wise visual sensitivity, attention-head intervention, and LogitLens
  mechanisms are now densely occupied by
  [Vision-aware Head Divergence (ACL 2025)](https://aclanthology.org/2025.acl-long.175/),
  [Same Attention, Different Truths (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Same_Attention_Different_Truths_Put_Logit-Lens_over_Visual_Attention_to_CVPR_2026_paper.html),
  and
  [CausalLens (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.html).
  “Find a mid/late layer and steer it” cannot headline this paper.
- [CEBC (ACL 2026)](https://aclanthology.org/2026.acl-long.2142/)
  already performs conformally calibrated evidence-bounded editing.  A generic
  support threshold is not new.
- [EUQ (2026)](https://arxiv.org/abs/2602.05535) already distinguishes
  conflict from ignorance using evidence theory.  The Claim Plane cannot claim
  invention of that conceptual distinction.
- [The Mirage of Performance Gains (2025)](https://arxiv.org/abs/2504.10020)
  argues that contrastive-decoding gains on POPE can be driven by
  unidirectional output adjustment and sampling artefacts.
- [Two Causes, Not One (2025)](https://arxiv.org/abs/2509.00371) separates
  omission from fabrication and reports that VCD can trade one for the other.
  A generic “methods trade fabrication for omission” claim is therefore also
  insufficient.

### What is specific to the medical setting

- [MedHEval (2025)](https://arxiv.org/abs/2503.02157) evaluates medical
  hallucination across visual misinterpretation, knowledge deficiency, and
  context misalignment, supporting the need to limit claims to visual
  grounding.
- [HalluCXR (2026)](https://arxiv.org/abs/2605.20469) reports that longer
  responses strongly predict hallucination and that ensemble reduction of
  fabrication increases omission.  This makes matched coverage a necessary
  control rather than an optional metric.
- [Uncertainty Estimation for Radiology Report Generation (ML4H 2025)](https://proceedings.mlr.press/v259/xu25a.html)
  studies correlations between uncertainty scores and report metrics, but not
  a reference-relative polarity/commitment coordinate.
- [CREST (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12338887/)
  separately labels false affirmation, false negation, uncertainty errors, and
  omission.  Its taxonomy supports the clinical stakes but also means that an
  error-category list alone is not novel.

### Retrieval-level novelty boundary

No directly overlapping work was retrieved for the exact combination of:

1. a reversible log-contrast coordinate system for each clinical claim;
2. reader-support and unobservable references;
3. matched-claim-coverage auditing across CE, OE, and reports; and
4. decomposing a mitigation update into polarity correction versus commitment
   contraction.

This is a retrieval result, not proof of novelty.  The closest conceptual
collisions remain EUQ, *The Mirage of Performance Gains*, *Two Causes, Not
One*, and CREST.

## The Claim Plane

For affirmative, negative, and undetermined sequence scores
\(\ell_+,\ell_-,\ell_?\), define

\[
P=\frac{\ell_+-\ell_-}{2},\qquad
C=\frac{\ell_++\ell_-}{2}-\ell_?.
\]

These are the two independent additive-logit-invariant coordinates of the
three-state simplex.  Relative to a locked visual null,

\[
P_v=P(x)-P(x_0),\qquad C_v=C(x)-C(x_0).
\]

- \(P_v>0\): the image moves the claim toward presence.
- \(P_v<0\): the image moves the claim toward absence.
- high \(C_v\): the image moves the answer toward a definite assertion.
- low \(C_v\): the image supplies little reason to leave the undetermined
  state.

The inverse, up to a shared logit constant, is

\[
\ell_+=C+P,\qquad \ell_-=C-P,\qquad \ell_?=0.
\]

Thus the coordinate system loses no information.  In contrast, the old CBD
construction deterministically ties commitment to polarity and throws away one
degree of freedom.  This representation also exposes two uncertain phenotypes:
conflict (definite evidence with weak net polarity) and ignorance (little
definite evidence).

For OE and reports, the three scores are teacher-forced sequence scores of
canonical positive, negative, and hedged realisations of the same normalised
claim.  The research object is therefore not the literal tokens Yes, No, and
Maybe.

## Local validation and falsification

### Evaluation status

The diagnostic set contains 32 real CXR/question pairs, balanced 16/16 between
report-derived positive and negative labels.  It is **grade C**, not formal
truth.  The geometry result can guide the next experiment; clinical efficacy
requires VinDr reader votes or physician review.

### Cross-model geometry

| Probe | Layer | \(R^2\): predict \(C_v\) from \(|P_v|\) | Mean absolute old-constraint residual |
|---|---:|---:|---:|
| Huatuo, per-image null, n=32 | final 28 | 0.007 | 0.604 |
| Huatuo, locked dev-global null, test n=16 | final 28 | 0.046 | 0.754 |
| Hulu-Med, per-image null, test n=16 | layer 27 | 0.026 | 1.632 |
| Hulu-Med, per-image null, test n=16 | final 36 | 0.495 | 1.035 |

The claim states occupy a plane rather than the old one-dimensional curve in
both architectures.  This is a diagnostic result, not yet a hallucination
improvement.

### The original mechanism failed

| Model / split | Baseline accuracy | Old CBD accuracy | Fabrication change | Omission change |
|---|---:|---:|---:|---:|
| Huatuo, n=32 | 87.5% | 56.3% | 20.0% to 45.5% | 0% to 25.0% |
| Huatuo, global-null test n=16 | 87.5% | 43.8% | 20.0% to 50.0% | 0% to 37.5% |
| Hulu-Med test n=16 | 100% | 75.0% | 0% to 14.3% | 0% to 25.0% |

For Huatuo, final-minus-layer-14 null commitment bias was -6.73 with a
500-draw bootstrap 95% interval approximately [-7.18, -6.29].  The locked
global-null test gave -6.04 [-6.54, -5.45].  Hulu-Med similarly gave -4.24
[-4.96, -3.43].  These results contradict monotonic final-layer commitment
bias growth.

On the Huatuo global-null test, only two final-layer errors were available.
Low final margin detected them better than low visual commitment (AUROC 0.857
versus 0.268), with very wide intervals.  A Claim-Plane decoder is therefore
not earned and must beat calibrated margin/abstention before entering the
method section.

The existing LLaVA-Med report-generation mitigation cache is inadmissible: all
694 outputs for several methods collapsed to the single token “The”.  It cannot
support any OE conclusion.

## Candidate tree and decision

| Candidate | Evidence | Collision / fatal risk | Decision |
|---|---|---|---|
| Late image-independent commitment bias + CBD | Directly contradicted in Huatuo and Hulu | Data-refuted core mechanism | Reject |
| Style/source-domain centre | Controlled style phenomenon did not meet the frozen threshold | Prior local negative result; DG story unsupported | Reject |
| Stability Needs a Null | Raw style drift added no value over margin locally | Crowded by stability and uncertainty work | Hold as control, not main idea |
| Layer-wise evidence/commitment decoupling | Mechanistically plausible | Dense 2025–2026 collision with VHD, LogitLens, CausalLens, HALP | Reject as headline |
| Claim Plane audit | Exact representation; cross-model geometry; unifies anti-cheat metrics | Major collision unless method rankings or clinical conclusions change | Continue with revisions |

## Reviewer-style idea evaluation

### First impression

- Paper type: **New Problem / Setting**.
- One-sentence story: Medical-VLM hallucination mitigation should be evaluated
  as movement on a claim plane—correcting polarity without purchasing apparent
  factuality by reducing commitment or coverage.

### Fatal-flaw audit

| Flaw | Severity | Required defense |
|---|---|---|
| Closest works already cover spurious contrastive gains, omission/fabrication trade-offs, conflict/ignorance, and medical error taxonomies | MAJOR | Demonstrate a result none of them supplies: claim-plane decomposition changes method rankings or reveals a reproducible clinical failure across CE, OE, and reports at matched coverage |
| Current positive evidence is grade-C CE geometry, not reader-grounded OE hallucination | MAJOR | VinDr reader-vote test plus at least one provenance-complete OE/report evaluation; automatic labelers cannot define truth |

### Five dimensions

| Dimension | Score | Ground |
|---|---:|---|
| Higher | 5 | No hallucination reduction is established; the previous method was worse than baseline |
| Faster | 6 | Two sequence-scoring passes are cheaper than multi-scale iterative decoding, but runtime has not been benchmarked |
| Stronger | 8 | Mechanism-based: polarity and commitment are exactly decoupled and the geometry replicated in two architectures; formal robustness is pending |
| Cheaper | 7 | Training-free and compatible with existing claim extraction, but physician reference review remains necessary |
| Broader | 9 | The same atomic-claim coordinate and coverage contract applies to CE, OE listing, and report generation |

### Paradigm-shift probe

| Probe | Assessment | Reason |
|---|---|---|
| First principles | Yes | Challenges the hidden assumption that signed confidence is a sufficient representation of three-state evidence |
| Elephant in the room | Yes | Factuality gains can be bought by shorter, more negative, or more uncertain outputs |
| Technology cycle | Partial | Modern VLM hooks and clinical claim parsers make the audit feasible, but the mathematics is not new |
| Hamming's rule | Conditional yes | It matters only if rankings or accepted conclusions change under the audit |

Verdict: **Accept with Revisions**, worth pursuing only through the decisive
validation below.  It is not currently an ICLR-oral-ready method.

## Paper skeleton

### Positioning

New Problem / Setting paper.  The goal is load-bearing; a mitigation module is
optional and must be earned after the problem result.

| Stage | Frozen content |
|---|---|
| Research background | Medical VLMs generate clinically consequential image-grounded claims; VCD, SECOND, MedHEval, and HalluCXR show both rapid mitigation progress and severe evaluation risk |
| Limitation 1 | Existing scalar confidence or signed visual contrast conflates polarity with willingness to commit |
| Limitation 2 | Fabrication-only metrics do not preserve negative correctness, uncertainty, omission, or claim coverage |
| Limitation 3 | CE, OE, and report evaluation use incompatible units and often let automatic parsers or judges define truth |
| Our Goal | Reframe hallucination mitigation as reference-relative movement on a two-coordinate clinical claim plane under fixed coverage |
| Challenge 1 | Convert free-form outputs into matched atomic claims without treating extraction as truth |
| Challenge 2 | Estimate comparable polarity and commitment coordinates despite verbalizer and null priors |
| Challenge 3 | Separate genuine error correction from commitment contraction, polarity transfer, and output shortening |
| Module A | Provenance-locked claim contract with reader/physician references and prediction-only parsers |
| Module B | Additive-invariant claim-plane scoring using semantic positive, negative, and hedged realisations plus a locked null |
| Module C | Matched-coverage audit with fabrication, false negation, omission, uncertainty calibration, length, and refusal controls |
| Contribution 1 | Claim Plane problem formulation and exact coordinate decomposition (Sections 2–3) |
| Contribution 2 | Unified provenance-safe CE/OE/report evaluation protocol (Section 4) |
| Contribution 3 | Cross-model audit showing which mitigation gains correct polarity and which merely change commitment or coverage (Section 5) |

All four consistency checks pass for this problem-paper skeleton.  A fourth
“new decoder” contribution is deliberately excluded until it beats margin,
temperature, length matching, and uniform-negation controls.

## Decisive continuation experiment

1. Obtain official VinDr annotations and build patient/image-disjoint dev and
   test splits across 0/3, 1/3, 2/3, and 3/3 reader support.
2. On dev only, calibrate the visual null, select one non-final agreement layer,
   and freeze thresholds. Do not inspect test during tuning.
3. Compare baseline, VCD, SECOND/Med-VCD, temperature, calibrated margin, and
   length-matched/uniform-negative controls in \((P,C)\).
4. Require commitment to add held-out agreement AUROC and Brier beyond
   same-layer absolute polarity, and require the selected early Claim Plane to
   beat the final Claim Plane by at least 0.05 AUROC with clustered confidence
   above zero.
5. Run OE abnormality listing with a fixed ontology and matched positive-claim
   coverage.  A method fails if fabrication falls while false negation,
   omission, refusal, or output contraction explains the gain.
6. Replicate on at least two model families.  Extend to reports only after CE
   and OE listing pass.

If Claim Plane does not add predictive value, change rankings, or expose a
reproducible clinical trade-off beyond existing taxonomies, reject it as an
elegant reparameterisation.  If it does, the low-resource method branch is a
polarity-preserving claim revision; it must be tested against calibrated
margin before being named as a contribution.

## Reproduction artifacts

- `anchor/corrected_sgta/clinical_claims.py`
- `anchor/corrected_sgta/prepare_claim_simplex_smoke.py`
- `anchor/corrected_sgta/analyze_claim_simplex.py`
- `anchor/corrected_sgta/run_huatuo_vindr_commitment_probe.py`
- `anchor/corrected_sgta/run_hulu_vindr_commitment_probe.py`
- `corrected_runs/claim_simplex/huatuo_n32_v1/`
- `corrected_runs/claim_simplex/huatuo_global_null_test_n16_v1/`
- `corrected_runs/claim_simplex/hulu_test_n16_v1/`

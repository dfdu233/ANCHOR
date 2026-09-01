# Seeing Is Not Grounding: Directional Clinical Response in Medical VLMs

Research decision: 2026-07-31

## Executive decision

The broad **Clinical Selectivity** formulation is retained as an audit, and
**Directional Clinical Response** is retained as an admission test:

> A medical VLM is grounded only if its claim polarity moves in the direction
> induced by independent reader support, and that directed movement exceeds
> movement under a support-preserving image swap.

Raw image sensitivity is therefore insufficient.  A model may change its
answer when the patient, scanner appearance, or unrelated anatomy changes and
remain invariant when the target finding changes.  Such a model “uses the
image” without using the right evidence.

This reframes the earlier style observation safely.  Style-dependent answers
do not establish a training-domain centre.  They show only that nuisance can
causally affect the answer.  The scientific question is whether target-state
sensitivity exceeds nuisance sensitivity.

The generic invariance-plus-sensitivity principle collides with VQA v2,
CF-VLM, PAIR-VLA, and selective-edit consistency.  What remains potentially
new is not directionality by itself, but **support-to-language uncertainty
erasure**: whether multi-reader ambiguity remains decodable in an earlier
Claim Plane after controlling for polarity, then disappears in the clinical
wording. The signed four-level response curve (0/3, 1/3, 2/3, 3/3) first checks
that the model has the relevant visual signal at all. The diagnostic separates two non-selective models
(HuatuoGPT-Vision and LLaVA-Med) from a possible work-side model (Hulu-Med),
but labels remain grade C and two lightweight mitigation attempts failed.
No hallucination-mitigation or reader-disagreement claim is earned yet.

## Minimal formulation

For a normalized image-grounded claim (c), let (P(c,x)) be its Claim-Plane
polarity score.  Construct a triplet:

- (x): anchor image with reader support (r);
- (x^{=}): another image with the same claim state/support band;
- (x^{\ne}): a nuisance-matched image with an opposite claim state.

Define

\[
D_{\mathrm{clinical}}
=\operatorname{sign}(r-r^{\ne})
  [P(c,x)-P(c,x^{\ne})],
\]

\[
D_{\mathrm{nuisance}}=|P(c,x)-P(c,x^{=})|,
\qquad
\mathrm{CSG}=D_{\mathrm{clinical}}-D_{\mathrm{nuisance}}.
\]

Positive CSG means the model reacts more to the clinical variable than to an
irrelevant patient/image swap.  Negative CSG identifies **non-selective visual
reliance**.  This is distinct from confidence, accuracy, raw answer-change
rate, and image-versus-no-image sensitivity.

For VinDr reader votes, exact support is retained.  Same-state pairs are
matched within 0/3, 1/3, 2/3, or 3/3; opposite-state comparisons use the
continuous support separation rather than pretending disagreement is a hard
binary label.

## Why this is a real problem

The field commonly asks whether removing or corrupting an image changes the
answer.  [CORAL (2026)](https://arxiv.org/abs/2607.03647) explicitly
operationalizes visual reliance through counterfactual image substitution and
trains on different-answer hard negatives.  Its CGO reward fires when the
answer changes under a label-different hard negative; it does not require the change
to reach that hard negative's answer or to follow an ordered clinical target.
A model can therefore be sensitive in the wrong direction.

The distinction has old roots but a new medical consequence:

- [VQA v2](https://arxiv.org/abs/1612.00837) paired similar images that answer
  the same question differently to reduce language priors.
- [CORAL](https://arxiv.org/abs/2607.03647) transfers different-answer hard
  negatives to medical VLM grounding.
- [PAIR-VLA](https://arxiv.org/abs/2605.13105)
  jointly trains invariance to task-preserving visual changes and sensitivity
  to task-changing changes, but for robot actions rather than clinical claims.
- [CF-VLM](https://arxiv.org/abs/2506.17267) already trains stable factual
  representations that react to minimal causal edits, making a generic
  counterfactual-invariance objective a direct collision rather than a new
  contribution.
- [VORD](https://arxiv.org/abs/2412.15739) already uses the term visual ordinal
  calibration, but orders token confidence between clean and corrupted images;
  it does not order clinical claims by independent reader support.
- [Multi-Rater Calibrated Segmentation](https://arxiv.org/abs/2605.02437)
  already treats expert agreement as an ordinal target.  Thus reader
  disagreement itself is not novel; the remaining delta must concern
  generative claim commitment and directional grounding.
- [CheXthought](https://arxiv.org/abs/2604.26288) goes closer: it uses
  multi-reader chest-X-ray data to predict human--human and human--AI
  disagreement and improve uncertainty communication.  A contribution cannot
  therefore be “use reader disagreement” or “verbalize uncertainty”; it must
  localize a support-to-language transition after clinically directed image
  use is established.
- [Vision-language models for chest radiography do not always need the
  image](https://arxiv.org/abs/2606.17710) causally audits target occlusion,
  irrelevant occlusion, and same-label image swaps.  It directly occupies the
  broad claim that high-performing medical VLMs may ignore images, but does
  not test an ordered response to independent reader support or the later
  commitment of an atomic claim.
- [Does It Fail to See or Fail to Know?](https://arxiv.org/abs/2607.04683)
  already attributes VLM failures to perception/recognition versus downstream
  knowledge and routes them to targeted interventions.  Generic failure-source
  decomposition is therefore occupied.
- [Decodable Is Not Grounded](https://arxiv.org/abs/2606.31257) already exposes
  grounded, prior, and inverted VLM regimes.  Anti-aligned responses cannot be
  claimed as our general discovery; their medical reader-support consequence
  is at most a setting-specific extension.
- [The Mirage of Performance Gains](https://arxiv.org/abs/2504.10020) shows
  that apparent contrastive-decoding gains can be output-policy artifacts.
- [Two Causes, Not One](https://arxiv.org/abs/2509.00371) shows that fabrication
  and omission need not share a cause.
- [Same Attention, Different Truths](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Same_Attention_Different_Truths_Put_Logit-Lens_over_Visual_Attention_to_CVPR_2026_paper.html)
  already uses logit-lens consistency to diagnose and mitigate object
  hallucination. Layerwise logit-lens evidence is therefore a tool, not our
  novelty.
- [HulluEdit](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_HulluEdit_Single-Pass_Evidence-Consistent_Subspace_Editing_for_Mitigating_Hallucinations_in_Large_CVPR_2026_paper.html)
  already edits orthogonal visual-evidence, prior, and uncertainty subspaces.
  Claim-Plane orthogonal editing is a causal control, not a method claim.
- [VES-RFT](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VES-RFT_Rewarding_Visual_Evidence_Sensitivity_to_Mitigate_Hallucinations_in_Large_CVPR_2026_paper.html)
  already turns image-versus-no-image entropy change into a training reward.
  Generic evidence sensitivity and entropy are occupied; the remaining delta
  is the matched ordinal response to independent reader support.

The remaining delta is not “use hard negatives,” “use ordinal labels,” or
“jointly learn invariance and sensitivity.”  It is a claim-level account in
which clinical grounding is a **signed response function of reader support**,
and language commitment is audited against that function across CE, OE, and
reports at matched coverage.

No work retrieved so far combines all three pieces: independent reader-vote
support, signed claim response, and free-text commitment.  This is a retrieval
result, not proof of novelty.  CORAL is the closest medical-VLM setting work;
PAIR-VLA/CF-VLM are the closest objective-level work; multi-rater ordinal
segmentation is the closest supervision-level work.

## Diagnostic evidence

### Contract

The current smoke uses two repeated CXR findings (pleural effusion and
pulmonary edema), 32 triplets, and 96 model inputs.  The v3 manifest fully
decodes every image before admission, leaving 32 complete triplets/96
successful LLaVA-Med records.  Earlier Huatuo/Hulu v2 runs retained two
explicit truncated-file errors and therefore have 30 complete triplets/94
successful records.  Labels are report-derived MedHEval binary QA:
**grade C, `formal_reference=false`**.

LLaVA-Med was admitted only after a loader audit.  All 391 serialized CLIP
vision tensors were exactly equal to the separately loaded frozen CLIP tower
after dtype conversion; all 295 non-vision checkpoint keys existed in the
runtime model; six projector/decoder/output sentinels were bit-exact after the
documented BF16-to-FP16 conversion.  An opposite-state image swap changed the
projected visual tokens substantially (L2 1144.48, mean absolute difference
0.382) and changed the three-state logits (L-infinity 0.258).  The previously
reported unused-vision-key warning is therefore a benign consequence of the
official delayed-tower loader, not grounds for excluding this checkpoint.

### Cross-model result

| Model | Final-layer state AUROC | Clinical change | Nuisance change | CSG (95% bootstrap CI) | Opposite-state pair accuracy |
|---|---:|---:|---:|---:|---:|
| HuatuoGPT-Vision-7B | 0.683 | 0.192 | 0.537 | -0.346 [-0.604, -0.079] | 0.600 |
| Hulu-Med-4B | 0.873 | 1.400 | 0.998 | +0.402 [-0.133, +0.904] | 0.867 |
| LLaVA-Med-v1.5-7B | 0.425 | -0.020 | 0.061 | -0.081 [-0.103, -0.059] | 0.500 |

Huatuo is the non-work regime: it is visibly image-sensitive, but nuisance
movement dominates clinically directed movement.  Hulu is the work-side
candidate: its final representation is clinically discriminative and trends
toward positive selectivity, although the CSG interval still crosses zero.

LLaVA-Med provides the cleanest counterexample to equating sensitivity with
grounding.  Its visual tokens and logits change across images, yet its
three-state argmax is `supported` for all 96 inputs.  Its CSG is significantly
negative on both dev (-0.087 [-0.122, -0.052]) and test
(-0.075 [-0.101, -0.048]), and is negative for both findings.  Thus image
dependence alone would certify a model whose response is clinically
non-selective.

The layer trajectories differ as well.  Huatuo's CSG is negative at all four
inspected layers.  Hulu is negative at layer 18 but becomes positive at layers
27 and 36.  This is a boundary observation, not yet a causal claim about
decoder layers.

### Why direction changes the conclusion

Analysis v5 reports both unsigned response and clinically aligned response on
the identical triplets.  Among triplets where the opposite-state absolute
change exceeded the same-state change, the fraction moving in the **wrong**
clinical direction was 35.3% for Huatuo, 4.8% for Hulu, and 63.6% for
LLaVA-Med.  Final-layer directional efficiency (mean signed change divided by
mean absolute change) was 0.354, 0.908, and -0.468 respectively.

Thus “the model changed under a clinically different image” has substantially
different meanings across models.  In particular, almost two thirds of
LLaVA-Med's apparently selective responses were anti-aligned.  This is the
current empirical reason to retain directionality; the numbers remain grade C
and cannot establish the four-level reader-response hypothesis.

On the locked split, low CSG improved anchor-error AUROC over low output margin
from 0.571 to 0.776 for Huatuo and from 0.621 to 0.909 for Hulu, while it fell
from 0.492 to 0.453 for LLaVA-Med.  Hulu's bootstrap delta was +0.288 with a
nominal 95% interval [+0.042,+0.576], but it contains only three anchor errors;
this is fragile grade-C evidence, not a discovery.  The LLaVA failure proves
that CSG is not a universal risk score.  Formal analysis v6 freezes the required
gate: at least +0.03 AUROC over output margin with clustered uncertainty, and
comparison to unsigned selectivity on the same triplets.

### Minimal method probe

A four-weight layer-logit calibrator was trained on the locked dev split and
tested on 31 unique held-out claim/image pairs:

| Model / method | Accuracy | AUROC | Fabrication | Omission |
|---|---:|---:|---:|---:|
| Huatuo raw final | 0.581 | 0.727 | 0.800 | 0.063 |
| Huatuo calibrated final | 0.613 | 0.727 | 0.400 | 0.375 |
| Huatuo supervised layer mixer | 0.677 | 0.750 | 0.333 | 0.313 |
| Huatuo selectivity mixer | 0.677 | 0.721 | 0.333 | 0.313 |
| Hulu raw final | 0.774 | 0.906 | 0.400 | 0.063 |
| Hulu calibrated final | 0.806 | 0.906 | 0.333 | 0.063 |
| Hulu supervised layer mixer | 0.774 | 0.883 | 0.400 | 0.063 |
| Hulu selectivity mixer | 0.806 | 0.913 | 0.400 | 0.000 |
| LLaVA raw final | 0.500 | 0.484 | 1.000 | 0.000 |
| LLaVA calibrated final | 0.531 | 0.516 | 0.438 | 0.500 |
| LLaVA supervised layer mixer | 0.563 | 0.477 | 0.500 | 0.375 |
| LLaVA selectivity mixer | 0.438 | 0.426 | 0.563 | 0.563 |

This establishes feasibility, not superiority.  The selectivity-specific
regularizer does not consistently beat ordinary calibration/supervised mixing.
On LLaVA it is strictly worse than the simple controls.  It is therefore
excluded from the contribution list until formal data show an advantage; the
current contribution candidate is the problem/metric, not this calibrator.

### Second method attempt: explicitly falsified

`fit_clinical_response_aligner.py` implemented matched-compute controls for
calibrated-final, task-only layer mixing, invariance-only, unsigned response,
unsigned selectivity, directional response, and signed selectivity.  All use
one scalar per inspected layer and one bias.  The first version added
invariance and direction losses separately; it learned an almost-all-
undetermined solution.  The second version used a signed triplet margin,

\[
\big[m+|P(x)-P(x^=)|
-\operatorname{sign}(r-r^{\ne})(P(x)-P(x^{\ne}))\big]_+^2,
\]

which a constant score cannot satisfy exactly.  It still failed on held-out
grade-C data: test three-state accuracy was 0.000/0.130/0.000 for
Huatuo/Hulu/LLaVA, and it did not beat the ordinary supervised or unsigned
controls on AUROC or Brier score.  The optimization preferred paying the
margin over creating unsupported separation when the frozen features lacked
a consistent linear order.

This rules out the tempting “four scalar weights solve grounding” story.  It
also yields a design constraint: a valid method must preserve claim coverage
and clear-case discrimination by construction or demonstrate both explicitly;
lower fabrication obtained by making every claim uncertain is failure.

### Surviving mechanism: uncertainty erasure, not generic failure attribution

The remaining hypothesis is deliberately narrower.  Single-answer/report
supervision may compress a genuinely ambiguous visual claim into one definite
linguistic realization.  With signed reader support
\(R=2r-1\), calibrated visual support \(\widehat R\), and expressed commitment
\(K\), the total gap decomposes exactly:

\[
K-R=(K-\widehat R)+(\widehat R-R).
\]

The first term is a **support-to-language transfer error**; changing commitment
can address it.  The second is a **visual support estimation error**; decoding
cannot manufacture the missing evidence.  This decomposition explains why
CBD and the response aligners fail on perception-limited models and why a
simple calibration can help only in a discriminative regime.  The algebra is
not a novelty claim; the scientific hypothesis is that multi-reader ambiguity
is decodable before generation but is systematically erased from clinical
claim wording.

`fit_reader_agreement_gate.py` implements the cheapest falsification. Every
recorded layer receives matched absolute-polarity and Claim-Plane probes. One
non-final layer is selected using dev only; image-cluster bootstrap then tests
the locked layer on test against same-layer polarity and final Claim Plane.
This is stricter than comparing thresholds: any strictly monotone transform of
one final scalar preserves AUROC, so the Claim-Plane coordinate must add new
held-out information.  The gate may change only definite versus
`undetermined` wording: it cannot add, delete, or flip a finding. It refuses
the current binary smoke data by design because 0/3 and 3/3 examples alone
contain no reader-disagreement target. Formal VinDr must show conditional
AUROC and Brier gains, at least +0.05 early-versus-final AUROC with clustered
95% confidence above zero, the same-direction conclusion under a dev-fitted
reader/finding-adjusted continuous-clarity sensitivity analysis, and no greater
than one-point clear-case or omission degradation before this becomes a method.
Raw 0/3--3/3 bins remain primary; the adjusted reference cannot rescue their
failure and may not reselect the layer.

## Mechanism and natural method

### Mechanism hypothesis

Directional response is necessary but not sufficient. The main hypothesis is
that reader disagreement remains decodable in an earlier residual through a
commitment coordinate $C$, conditional on polarity magnitude $|P|$, but
the final support-to-language map removes that information and emits a
definite clinical claim. This predicts all three of the following on formal
reader-vote data:

1. same-layer $(P,C)$ predicts agreement better than $|P|$ alone;
2. the best dev-selected non-final Claim Plane beats the final Claim Plane by
   at least 0.05 test AUROC with clustered confidence excluding zero; and
3. moving along the null commitment gradient after projecting off the
   polarity gradient changes overcommitment without flipping clear claims.

Reject uncertainty erasure if any of these fail on the majority of eligible
findings or do not replicate in at least two models. A model with poor signed
clinical response is classified as perception-limited and is ineligible for a
commitment-only fix; decoder intervention cannot manufacture visual support.

### Conditional method objective

If the formal mechanism survives, the minimal method is not a new VLM adapter.
It keeps the two independently identified evidence channels separate. The DCR
polarity channel is calibrated on dev to
\(\pi_c(x)=P(\text{present}\mid x,c,\text{definite})\); the commitment-tetrad
channel is calibrated to
\(\kappa_c(x)=P(\text{reader-unanimous}\mid x,c)\). They induce the evidence
distribution

\[
r_c=(\kappa\pi,\ \kappa(1-\pi),\ 1-\kappa)
\]

over supported, refuted, and undetermined. A **Reader-Calibrated Commitment
Projection (RCCP)** then minimally projects the decoder distribution into the
evidence envelope: definite language mass cannot exceed \(\kappa\), and a
decoder polarity that contradicts decisive evidence is clipped to the
support/refute boundary and verbalized as undetermined. The projection may
hedge but never manufacture the opposite claim.

For fixed-ontology abnormality listing only, omission recovery remains a
separate operation. **Evidence-Conserving Claim Exchange (ECCE)** may pair a
weak positive draft claim with a stronger omitted candidate, but only using
the dev-calibrated per-finding \(\pi_c\), never raw logits from different claim
prompts. Every swap is one-for-one, so positive-claim count and total claim
count are unchanged.

The exchange is fixed-budget selection:

\[
A^*=\arg\max_{A\subseteq U,\ |A|=|A_0|}\sum_{c\in A}s_{\ell^*}(c),
\]

with a non-negative exchange margin and $U$ equal to draft plus the frozen task
ontology. Non-visual, negative, and out-of-ontology claims are preserved
exactly. Certainty is adjusted only after identity selection, so hedging never
masquerades as removal of fabricated content.

A causal commitment version may subtract only the component of $\nabla C$
orthogonal to $\nabla P$, with exact norm restoration. It must beat
temperature, calibrated margin, final-layer Claim Plane, random-subspace
steering, and matched-coverage controls. ECCE is intentionally simple; without
the DCR+tetrad mechanism it is ordinary ontology reranking and has no novelty
claim.

The signed relative-margin DCR objective remains a diagnostic negative result:
its scalar implementation collapsed toward `undetermined`. Projector/LoRA
training is allowed only if formal VinDr proves the ordered signal exists but
the frozen commitment gate lacks capacity.

For OE, the unit remains a normalized claim rather than a token.  Draft claims
and a fixed ontology provide candidates; positive/negative/hedged sequence
scores provide $P$; output is evaluated at matched positive-claim coverage.
The method may not claim success by shortening reports, refusing, or converting
everything to negative/uncertain.

## Candidate-tree decision

| Candidate | Grounding | Collision | Decision |
|---|---|---|---|
| Polarity-only contrastive decoding | No stable held-out improvement | VCD family; local negative result | Reject |
| Causal evidence masks/region verification | Plausible | Direct 2026 collision with CoEV and counterfactual grounding work | Reject |
| Null ensemble / matched image swap alone | Null choice matters | CORAL, DCD, SPCD already attack negative construction | Control, not headline |
| Claim Plane alone | Exact representation | Risks elegant reparameterization without changed conclusions | Retain as measurement layer |
| Broad Clinical Selectivity | Three-model boundary | PAIR-VLA, CF-VLM, SEC cover the generic principle | Metric/control only |
| Directional Clinical Response | Direction changes conclusions; formal four-level test pending | Multi-rater ordinal segmentation and CORAL cover separate pieces | Admission test/control, not headline |
| Reader-agreement commitment gate | Cannot be tested on grade-C binary labels; implementation refuses them | Calibration/selective prediction are close; novelty depends on decoder-level uncertainty erasure | Priority formal falsification; no method claim |
| Raw or claim-centered cross-claim ranking | Real Huatuo MIMIC Grade-C screen: raw early MRR 0.081; centering 0.219, but early equals final (0.222), dev/test selection reverses, and absolute retrieval remains weak | Contextual/class-wise calibration, GLA, TCLA | Reject as method; retain centering/quantiles as calibration controls |
| Reader-Calibrated Commitment Projection | Closed-form two-channel representation and synthetic invariants only | Calibration/selective prediction are crowded; novelty depends on DCR+tetrad mechanism and layerwise transition | Conditional main method; formal VinDr and two-model OE gates required |
| Evidence-Conserving Claim Exchange | Count-preserving plumbing; raw-score real screen failed | Ontology verification/reranking is crowded; novelty cannot come from top-k selection | Omission-recovery ablation only with dev-calibrated per-finding support and identical positive-claim count |

## Reviewer-style idea evaluation

### First impression

- Paper type: **New Problem / Setting**, with a conditional mechanism-derived
  method only if formal gates pass.
- Frozen story: **image sensitivity is not clinically directed grounding**.
  Only after an atomic claim moves in the direction implied by independent
  reader support, beyond matched same-support drift, may we ask whether graded
  ambiguity is later erased when support is converted into clinical language.
- The original scalar Commitment-Bounded Decoding is a negative result, not
  the method attached to this story.  The only admissible method candidate is
  a polarity- and coverage-preserving commitment gate, conditional on the
  formal mechanism test below.

### Fatal flaws

| Flaw | Severity | Required defense |
|---|---|---|
| F1: causal image-use audits occupy generic sensitivity, while CheXthought already uses multi-reader disagreement to predict disagreement and improve uncertainty communication | MAJOR | The novelty delta must contain all three pieces jointly: independent reader support, signed claim response beyond same-support drift, and the support-to-language commitment transition. Any one piece alone is prior art |
| F6: current references are grade C and the selectivity calibrator is not consistently superior to simple controls | MAJOR | Reader-grounded VinDr test, two models, and a matched-compute sensitivity-only versus selective-objective comparison; OE at matched coverage before any broad claim |
| F7: the original scalar and signed-margin mitigation probes collapse toward uncertainty | CRITICAL for those methods, not for the diagnostic hypothesis | Retire those methods. A future gate must preserve claim identity, polarity, and coverage by construction, then jointly report three-state accuracy, clear-case performance, claim count, omission, length, and refusal |

### Five dimensions

| Dimension | Score | Evidence |
|---|---:|---|
| Higher | 4 | Both lightweight mitigation attempts failed; only the directional diagnostic survives |
| Faster | 5 | The audit is training-free but needs anchor/same-support/opposite-support forwards; no latency advantage is claimed and the low-overhead scalar methods failed |
| Stronger | 8 | The mechanism explicitly separates target change from nuisance and exposes a cross-model work/non-work boundary; formal references remain pending |
| Cheaper | 8 | Diagnostic and calibrator ran on existing 4B/7B models in seconds to minutes without full training |
| Broader | 7 | The atomic-claim abstraction can cover CE, OE listing, and reports, but evidence currently covers only two CXR findings under grade-C labels; OE transfer is unverified |

Verdict: **Accept with Revisions — pursue the decisive reader-grounded
mechanism test, but do not invest in or claim a mitigation method until it
passes.**  This accepts the *question and experiment*, not the original CBD
method.  It is not yet an ICLR-oral-ready paper.

### Lifecycle and capability match

- Lifecycle: frontier exploration moving into first formal validation.  The
  next result can still terminate the paper direction cheaply.
- Compute: good fit.  The audit and layer probes run on the available local
  4B/7B models without end-to-end training.
- Data: conditional fit.  VinDr supplies the required independent reader
  labels, but access is credentialed and annotations have not yet been staged.
- Engineering: good fit.  Triplet construction, image-disjoint splitting,
  three model hooks, gate fitting, OE claim accounting, and failure checks are
  implemented and tested.
- Time: competitive risk is high in the 2026 literature cycle.  A broad
  “uncertainty” or “image reliance” story is already obsolete; only the frozen
  three-part delta above is worth the formal run.

### Decision boundary

The formal experiment has asymmetric value:

1. If reader agreement is not conditionally decodable beyond polarity, reject
   the support-to-language mechanism and retain DCR only as an audit metric.
2. If agreement is decodable but does not fall by at least 0.05 AUROC from a
   dev-selected early layer to the final layer, reject uncertainty erasure.
3. If both pass in at least two DCR-eligible models, test the constrained gate.
   The paper remains a mechanism/measurement paper unless OE improves at
   matched claim coverage without more omission, shortening, or refusal.

### Paradigm-shift probe

| Probe | Assessment | Reason |
|---|---|---|
| First principles | Yes | Challenges the default equivalence between image sensitivity and grounding |
| Elephant in the room | Yes | Methods can become more visually reactive to the wrong variables |
| Technology cycle | Partial | Modern claim scoring makes the audit practical, but paired invariance/sensitivity is not new mathematics |
| Hamming's rule | Conditional yes | It matters if current grounding metrics or mitigation rankings change |

Disruptive potential is strong but conditional on a ranking/conclusion change.

## Paper-construction audit

The construction paths are analytical reconstructions, not claims about the
authors' private discovery process:

- **ViT path—redefine the unit:** CE, OE, and reports become atomic clinical
  claims with reader support.
- **SigLIP path—remove accidental coupling:** “visual change” is separated
  from “clinically relevant visual change.”
- **Chinchilla path—identify the ratio:** replace raw reliance by the balance
  between clinical response and nuisance response.
- **Model-Collapse path—study the boundary:** Huatuo and Hulu provide explicit
  non-work/work regimes before a method is added.

The user previously authorized the agent to make the scientific decisions;
ViT + SigLIP are weighted most heavily.  The Chinchilla/Model-Collapse paths
serve only as discipline for the ratio and boundary experiment.

## Decisive formal experiment

1. Build VinDr claim triplets from official reader votes.  Match by finding,
   source/site metadata, view position, and coarse anatomy; keep 0/3, 1/3,
   2/3, and 3/3 separate.
2. Freeze image-disjoint dev/test before model output inspection.  Dev chooses
   matching distance, the non-final candidate layer, and thresholds; test is
   opened once.
3. Compare accuracy/margin, no-image or shuffled-image reliance, CORAL-style
   opposite-state sensitivity, and full CSG first; perception-limited models
   are not eligible for commitment-only mitigation.
4. Before learning any probe, use matched commitment tetrads at fixed majority
   polarity: two unanimous and two disagreement images for 0/3↔1/3 and
   3/3↔2/3. Require early majority-directed support to distinguish clear from
   ambiguous beyond within-state nuisance drift, then beat the final layer by
   at least 0.05 held-out macro AUROC with tetrad-bootstrap confidence
   excluding zero. This prevents a flexible linear probe from manufacturing
   the headline phenomenon. Both polarity branches require at least 10 test
   tetrads and must pass within a finding; strictly more than half of qualified
   findings must pass, so pooling cannot hide a narrow effect.
5. Only after the direct tetrad gate passes, require same-layer commitment to
   add held-out agreement AUROC and Brier beyond absolute polarity. Apply a
   same-polarity clear↔ambiguous activation patch or the polarity-orthogonal
   commitment intervention and require lower
   disagreement overcommitment than baseline and random-subspace steering,
   with exact norm and temperature controls and at most one-point clear-case
   loss.
6. Replicate on at least two model families and a majority of eligible
   findings. Extend to OE listing only after CE passes.
7. For OE, preserve the fixed ontology and claim budget; report fabrication,
   false negation, omission, uncertainty, length, and refusal together.
   Compare commitment-only gating separately from ECCE: the former may improve
   overcommitment but cannot claim content-hallucination reduction; the latter
   must reduce fabrication without increasing omission at exactly the baseline
   positive-claim count.

Reject support-to-language uncertainty erasure if the conditional information,
early-versus-final loss, or polarity-preserving causal effect fails. Keep
Directional Clinical Response and broad Clinical Selectivity only as admission
tests and audit metrics.

## Reproduction artifacts

- `anchor/corrected_sgta/clinical_claims.py`
- `anchor/corrected_sgta/prepare_clinical_selectivity_smoke.py`
- `anchor/corrected_sgta/analyze_clinical_selectivity.py`
- `anchor/corrected_sgta/fit_selectivity_calibrator.py`
- `anchor/corrected_sgta/fit_clinical_response_aligner.py`
- `anchor/corrected_sgta/fit_reader_agreement_gate.py`
- `anchor/corrected_sgta/audit_llava_med_loader.py`
- `anchor/corrected_sgta/run_llava_vindr_commitment_probe.py`
- `anchor/corrected_sgta/prepare_vindr_selectivity_triplets.py`
- `anchor/corrected_sgta/prepare_vindr_commitment_tetrads.py`
- `anchor/corrected_sgta/analyze_commitment_tetrads.py`
- `corrected_runs/clinical_selectivity/manifest_v3.jsonl`
- `corrected_runs/clinical_selectivity/huatuo_v2/`
- `corrected_runs/clinical_selectivity/hulu_v2/`
- `corrected_runs/clinical_selectivity/llava_loader_audit_v1.json`
- `corrected_runs/clinical_selectivity/llava_v3/`

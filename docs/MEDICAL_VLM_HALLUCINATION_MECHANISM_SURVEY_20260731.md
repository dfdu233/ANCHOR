# Medical VLM Hallucination: From Visual Influence to Clinical Support

Research freeze: 2026-07-31

## Executive decision

The project should not propose another generic visual-contrast, attention, or
confidence-calibration trick. The defensible question is narrower:

> Does a medical VLM encode qualified readers' judgment distribution for an
> image-grounded claim before generation, then transform that distribution
> into language more definite than the visual evidence permits? If so, can the
> causal reader-disagreement channel constrain claim realization without
> reducing matched claim coverage?

The distinguishing target is an **external reader distribution**, not model
entropy, attention mass, image-null sensitivity, or uncertainty wording in one
report. The paper asset would be a causal support-to-commitment transition, not
the existence of a third label. No method claim is currently earned; formal
VinDr reader votes are the admission test.

## Frozen questions and review method

1. Which mitigations reverse sign across backbone, CE/OE task form, or
   hallucination cause?
2. Which evaluations reward shorter, more negative, more uncertain, or more
   refusal-heavy output without correcting clinical content?
3. Is reader disagreement decodable separately from claim polarity, and is it
   selectively lost between early representation and final language?
4. Can a causal intervention preserve polarity and coverage while reducing
   excessive linguistic commitment?

The review followed forward/backward collision search over official CVPR,
ICML, ICLR, ACL, EMNLP, NeurIPS, PMLR, and dataset pages. Recent medical work
without a confirmed archival venue is marked as a preprint. Searches were
split into visual contrast, attention/activation editing, verification/expert
guidance, uncertainty/calibration, and medical report evaluation/reader
disagreement. “Not retrieved” is a risk estimate, not proof of novelty.

## Evidence synthesis

### Visual influence is not directed clinical support

[VCD](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_Decoding_CVPR_2024_paper.html)
contrasts clean and corrupted image logits, while
[M3ID](https://openaccess.thecvf.com/content/CVPR2024/papers/Favero_Multi-Modal_Hallucination_Control_by_Visual_Information_Grounding_CVPR_2024_paper.pdf)
amplifies image-conditioned information. Neither establishes that an
instance-level delta moves a clinical claim in the direction supported by
independent readers.

[PAI](https://eccv.ecva.net/virtual/2024/poster/2599) strengthens image-token
influence, but [DAMRO](https://aclanthology.org/2024.emnlp-main.439/) and
[AVISC](https://aclanthology.org/2025.findings-acl.99/) show that high or
disproportionate visual attention can land on irrelevant tokens.
[Same Attention, Different Truths](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Same_Attention_Different_Truths_Put_Logit-Lens_over_Visual_Attention_to_CVPR_2026_paper.html)
finds similar attention magnitude for truthful and hallucinated object tokens.
Attention amount is therefore not a signed correctness signal.

[System-Mediated Attention Imbalances Make VLMs Say Yes](https://aclanthology.org/2026.findings-acl.1940/)
shows that a causal intervention can change yes/no behavior without changing
open explanations correspondingly. CE is a mechanism probe, not evidence that
an OE method works.

### Mitigation is not task- or model-universal

[MedHEval](https://arxiv.org/abs/2503.02157) evaluates 11 medical/general VLMs
and seven mitigations over CE and OE. Its tables contain sign reversals: PAI
can improve CE while worsening report CHAIR and recall; OPERA improves one
LLaVA-Med backbone but worsens hallucination on another; VCD can worsen
knowledge-oriented OE. The right object is therefore
`method × backbone × task form × hallucination cause`, not one average score.

Our local common-protocol MIMIC CE cache gives a compatible warning: greedy
accuracy was 75.56%, VCD 69.53%, M3ID 68.52%, and OPERA 77.61%. These are local
results, not literature claims, but they reject the assumption that a
contrastive update is intrinsically beneficial.

The medical method space is crowded: MEDA occupies expert-guided activation
editing; CCD occupies clinical expert contrastive decoding; Med-VCD occupies
sparse medical visual contrast; CoEV occupies generated-claim visual
verification. A new mask, layer, expert, or verifier is insufficient novelty.

### Fabrication reduction can be bought with omission

[CHiP](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e1c73e9595126794186536cfbbed012f-Abstract-Conference.html)
shows that similar output length does not guarantee similar object coverage;
ambiguous content may still be omitted. Token-length matching is weaker than
claim-coverage matching.

[HalluCXR](https://arxiv.org/abs/2605.20469), a recent preprint, reports that
response length predicts detected hallucination and that an ensemble can
reduce fabrication while increasing omission.
[RaTEScore](https://aclanthology.org/2024.emnlp-main.836/) identifies the
complementary metric failure: unnormalized error counts reward very short
reports. Every comparison must report positive claim count, finding recall,
length, negative rate, refusal, and a matched-claim-coverage curve.

### Automatic metrics propose structure, not truth

[CheXbert](https://aclanthology.org/2020.emnlp-main.117/) is a 14-label report
text extractor, not an image oracle. Collapsing uncertain and blank outcomes
into negative erases clinically meaningful modality.

[RadGraph](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c8ffe9a587b126f152ed3d89a146b445-Abstract-round1.html)
extracts entities and relations. Its cross-dataset extraction is imperfect,
and graph overlap with a reference cannot establish image support.
[GREEN](https://aclanthology.org/2024.findings-emnlp.21/) is more clinically
aligned than lexical metrics but remains a learned evaluator and is weaker
out of domain. These tools may propose or audit claims; independent readers,
structured labels, or physician review must supply reference states.

### Uncertainty referents are not interchangeable

[Direct Uncertainty Prediction for Medical Second Opinions](https://proceedings.mlr.press/v97/raghu19a.html)
and [Diagnostic Uncertainty Calibration](https://proceedings.mlr.press/v130/mimori21a.html)
establish reader disagreement as a target distinct from ordinary class
probability. VinDr-CXR's three independent training readers are suitable for
empirical reader distributions, but vote fractions are not biological truth.

[VisualCheXbert](https://arxiv.org/abs/2102.11467) shows that report-derived
labels and image-derived labels differ. Certainty phrases also vary by reader
and specialty. Report wording cannot substitute for independent disagreement.

| Concept | Meaning |
|---|---|
| Reader distribution | Empirical judgments under a specified protocol |
| Visual ambiguity | The modality can answer in principle, but this image is unclear |
| Unobservable | Required history, laboratory value, prior image, or metadata is absent |
| Model uncertainty | Instability or entropy of the model prediction |
| Verbal uncertainty | Hedge or certainty language in the output |
| Reportability | Whether a visible fact should be mentioned for this task |

### Closest collisions define the novelty boundary

[SRRG](https://aclanthology.org/2025.acl-long.1301/) already uses
Present/Absent/Uncertain findings. CURV targets uncertainty expression;
VL-Calibration separates visual and reasoning confidence; HAC calibrates
medical VQA confidence; MAIRA-2/RadFact, ReXTrust, CoEV, and formal report
verification operate at sentence or claim level. Generic “third state,” “claim
verification,” or “uncertainty-aware report generation” is occupied.

[Mind the Uncertainty in Human Disagreement](https://arxiv.org/abs/2410.02773)
already calibrates VQA prediction distributions toward distributions of ten
human answers. [Just how sure are you?](https://arxiv.org/abs/2606.27023)
already trains medical VLMs to verbalize calibrated confidence using a visual
presence × text-integrity perturbation design. Human-distribution calibration
and medical verbal-confidence calibration are therefore not novelty claims.

The full-paper collision audit changes what may be claimed. In particular,
[Diagnosing Visual Ignorance in Vision-Language Models](https://arxiv.org/abs/2606.06890)
already uses supervised layerwise probes and counterfactual layer replacement
to show that visually grounded semantics can appear in intermediate layers and
be suppressed by language priors later. That transition is prior art, not our
insight. [CheXthought](https://arxiv.org/abs/2604.26288) already collects
multi-reader findings and uncertainty language, predicts human--human and
human--AI disagreement from images, and uses visual-attention hints and
fine-tuning to reduce hallucination. Multi-reader uncertainty by itself is also
prior art.

The exact granularity audit is:

| Work | External reader distribution | Layerwise causal mechanism | Claim polarity vs clarity | OE equal-coverage mitigation | Exact boundary |
|---|---:|---:|---:|---:|---|
| CheXthought | Yes | No | Predicts disagreement, not an internal support-to-commitment conversion | No | Occupies image-to-disagreement prediction and uncertainty-aware supervision |
| Diagnosing Visual Ignorance | No | Yes | Ground-truth answer versus language-prior answer, not reader-distributed evidence | No | Occupies intermediate visual signal followed by late language-prior suppression |
| [VIHD](https://arxiv.org/abs/2605.20772) | No | Layer selection plus visual-token masking | Calibrated semantic entropy, not reader support | Detection only | Occupies medical visual-intervention hallucination detection |
| [CURV](https://proceedings.neurips.cc/paper_files/paper/2025/hash/86a99b74adac7c77998902371f53850d-Abstract-Conference.html) | No | No | Learns uncertainty expression through a three-stage training pipeline | Report quality, not locked positive-claim coverage | Occupies uncertainty-aware report reasoning and generation |
| [Human Disagreement VQA](https://arxiv.org/abs/2410.02773) | Yes | No | Aligns model answer distributions to human answer distributions | No | Occupies human-distribution calibration in VQA |
| [VLI](https://aclanthology.org/2026.acl-long.1784/) | No | Instance-specific counterfactual latent steering | Model-internal grounded/ungrounded conflict and confidence flattening | Evaluates free-form object hallucination, without claim-budget matching | Occupies dynamic visual steering plus adaptive confidence calibration |
| This hypothesis | Yes | Required | Separates signed polarity from reader clarity and language commitment | Required before method claim | Tests whether evidence distribution is over-compressed into language |

Therefore the remaining cell is only the conjunction of:

1. external multi-reader distribution as target;
2. claim-conditional signed support, not unsigned sensitivity;
3. layerwise and causal support-to-language transformation;
4. OE realization under fixed ontology and matched positive-claim coverage;
5. omission, false negation, refusal, length, and negativity controls.

The closest works do not establish the complete conjunction above. This is a
defensible boundary, not proof of novelty; the search must be refreshed before
submission. If the formal experiment merely rediscovers late visual
suppression, the paper is rejected internally even if the effect is large.

## Local falsification

### Rejected: one-dimensional Commitment-Bounded Decoding

The old logits

\[
z_+=e,\qquad z_-=-e,\qquad z_?=\tau-|e|
\]

tie polarity and commitment to one curve. Huatuo and Hulu probes harmed clear
accuracy and increased fabrication and omission. The hypothesized monotonic
late commitment-bias growth had the opposite sign. This mechanism is rejected.

### Rejected: DCR as a universal decoding authorization gate

Directional Clinical Response is a useful audit: a claim score should move
with clinical support more than with same-state nuisance. But a model can pass
a clinical image-swap direction check and still be harmed by a different
real-null contrastive direction. Every intervention direction needs its own
admission test.

### Not promoted: claim-action rank reversal

Modern RadGraph XL was run on cached MIMIC report outputs. The common cohort
had 362 reports and six methods; five complete methods had 694. Counting
hedged-positive content correctly did not change ranking in either cohort.
Only one M3ID hedged false positive was hidden by the collapsed metric.

The cache was weak for this question: ontology match was about 13–16%,
positive-finding recall about 0–4%, and beam/OPERA produced nearly no matched
positive findings. This Grade-C negative result proves the invariant is
necessary but does not establish benchmark-level importance.

### Rejected proxy: single-report linguistic clarity

A real-image Grade-C screen used 62 positive chest-finding claims extracted
from single reports (31 definite, 31 hedged) and 62 complete Huatuo four-layer
trajectories. The development-selected Claim Plane did not add held-out clarity
information beyond same-layer absolute polarity (AUROC gain -0.061, clustered
95% CI [-0.471, 0.311]) and was worse than the final Claim Plane by 0.209 AUROC
(CI [-0.455, -0.015] for early minus final). Replacing the final clarity
decision with the selected early one reduced clear-case accuracy by 35.7
percentage points. Report wording is therefore rejected as a substitute for
independent reader evidence; this result does not test the formal VinDr
hypothesis.

Artifacts:

- `anchor/corrected_sgta/analyze_claim_action_audit.py`
- `corrected_runs/claim_action_audit_mimic_v1/summary_common_n362.json`
- `corrected_runs/claim_action_audit_mimic_v1/summary_complete_n694.json`

## Candidate tree

| Candidate | Importance | Mechanism | Novelty | Executability | Decision |
|---|---:|---:|---:|---:|---|
| Generic visual contrast/attention steering | 7 | 4 | 2 | 8 | Reject |
| Claim-action evaluation headline | 7 | 7 | 5 | 9 | Demote: Grade-C rank test failed |
| DCR-authorized decoding | 7 | 6 | 5 | 8 | Reject: direction mismatch |
| Multi-reader support-to-commitment collapse | 9 | 9 | 7 | 6 | Continue through formal gate |

## Reviewer-style idea evaluation

### First impression

- Paper type: Novel Problem / mechanism-first New Setting.
- One-sentence story: medical VLM research calibrates how much models listen
  to images, while this work asks what an image supports according to
  independent readers and whether language commits beyond that support.

### Fatal-flaw audit

| Flaw | Severity | Concrete defense |
|---|---|---|
| CheXthought covers reader-disagreement prediction and Visual Ignorance covers late suppression; a loose story would be an A+B paper | MAJOR | Test the conditional mechanism neither establishes: reader/finding-adjusted clarity loss with polarity retained, followed by selective commitment change under causal patching |
| Scope currently spans mechanism, decoder, unified benchmark, three task forms, four models, and physician review | MAJOR | Freeze the main paper to VinDr CE mechanism plus VinDr OE listing on two model families; admit report generation and a new decoder only after the mechanism and matched-coverage gate pass |

The rejected one-dimensional CBD is a data-refuted mechanism and remains
rejected. It is not treated as a fixable version of the surviving two-channel
reader-distribution hypothesis.

### Lifecycle and capability match

| Aspect | Current evidence | Assessment |
|---|---|---|
| Category | Frontier exploration with a data-intensive core | 6--9 month lifecycle |
| Compute | One local RTX 4090; existing hooks for three medical VLMs | Adequate for frozen-model probing, not broad retraining |
| Data | PhysioNet account supplied; annotations not yet downloaded | Yellow until authenticated annotation download succeeds |
| Human truth | A 300-case physician review is planned; collaborator/budget not documented | Yellow/high schedule risk |
| Weekly hours/team | Not specified | Cannot certify lifecycle fit; keep scope narrow |

### Five dimensions

| Dimension | Score | Evidence | Lift required |
|---|---:|---|---|
| Higher | 5 | No hallucination gain is established; old CBD and broad contrast controls were worse | Formal OE gain at matched coverage with omission non-increase |
| Faster | 6 | Frozen-model probes and a post-hoc realization layer avoid full VLM retraining | Benchmark runtime and compare with verifier/contrast baselines |
| Stronger | 8 | Mechanism-based, not confirmed: signed support and reader clarity separate nuisance sensitivity from clinical direction | Pass clustered, cross-model causal patching controls |
| Cheaper | 7 | Public annotations and one-GPU probing are substantially cheaper than preference/RL training | Minimize physician review through stratified sampling; do not replace it with an LLM judge |
| Broader | 8 | Mechanism-based, not confirmed: one atomic-claim contract covers CE, OE listing, and reports | Demonstrate locked CE-to-OE transfer before claiming report generality |

### Paradigm-shift probe

| Probe | Verdict | Rationale |
|---|---|---|
| First Principles | Yes | Challenges the assumption that more image influence means more correct grounding |
| Elephant in the Room | Yes | Makes omission, shortening, negativity, refusal, and hedging explicit costs of apparent safety |
| Technology Cycle | Partial | Modern VLM hooks and claim extractors make the audit feasible, though reader disagreement predates VLMs |
| Hamming's Rule | Yes | A valid distinction between influence and support would change mitigation design and evaluation priorities |

Disruptive potential is strong, with equally strong execution and reception
risk.

### Verdict

**Accept with Revisions — worth pursuing only through the formal VinDr
validation experiment.** The two MAJOR risks must be resolved before the
project is described as an ICLR paper: prove the remaining novelty boundary
against the closest 2026 work, and cut the first formal scope to two models and
CE plus OE listing. A decoder and report-generation extension are conditional,
not promised contributions.

## Formal hypothesis

For claim \(c\), let \(R(c\mid x)\) be the empirical reader distribution and
\(K(c\mid y)\) the commitment in generated language. Reader support is split
into polarity \(r\) and clarity \(q\):

\[
p_S=qr,\qquad p_R=q(1-r),\qquad p_U=1-q.
\]

This avoids reconstructing disagreement from one signed margin. `1/3` and
`2/3` have similar disagreement but opposite polarity; `0/3` and `3/3` are
both clear but opposite.

The mechanism predicts:

1. early layers decode clarity beyond absolute polarity, finding prevalence,
   image quality, and reader effects; the primary layer selection and test use
   development-fitted reader/finding nuisance controls, while unadjusted probes
   are diagnostic only;
2. incremental clarity AUROC falls by at least 0.05 at the final layer;
3. language definiteness grows specifically where reader clarity is low;
4. patching the clarity component changes definite/hedged realization while
   minimally changing claim polarity;
5. temperature, norm matching, random directions, shortening, and uniform
   negative answers cannot explain the effect.

The method branch is admitted only after these tests. Its minimal form is
reader-bounded realization: draft, extract image-grounded claims, estimate
calibrated polarity and clarity from a causally validated layer, alter
commitment only when polarity is stable, and exchange weak drafted findings
for stronger omitted findings at a fixed positive-claim budget. This is
conditional plumbing, not a current contribution.

## Locked evaluation and stop rules

- Split patient/image before calibration; never tune on test.
- Stratify findings over 0/3, 1/3, 2/3, and 3/3 support.
- Preserve every pseudonymous `rad_ID` and binary vote, verify that their sum
  equals the aggregate label, and model reader and finding effects using the
  development split only. Do not call votes biological probability.
- Keep raw `0/3`--`3/3` bins as the primary reference. Because three votes are
  a noisy panel sample, also require a sensitivity analysis based on a
  development-fitted penalized reader/finding model. Test-item latent support
  may use its observed reference votes with reader/finding effects frozen, but
  never VLM outputs. A layer conclusion that reverses under this adjustment is
  not mechanism-grade evidence.
- Separate polarity, modality, anatomy, attributes, provenance,
  reportability, and observability.
- Count hedged-positive claims as positive content.
- Match positive-claim coverage, not just token length.
- Report fabrication, false negation, omission, attributes, location,
  Brier/NLL, length, claim count, negativity, and refusal.
- Require two VLM families, most eligible findings, and at least 300 stratified
  physician-reviewed OE/report examples.

Reject the mechanism if early clarity does not exceed final clarity by 0.05
AUROC with clustered 95% CI above zero, or if polarity/reader/finding controls
explain it. Reject the method if gains vanish at matched coverage, omission
increases, clear accuracy falls over one point, or shorter/more negative/more
uncertain/refusal-heavy output explains the result.

## Conclusion

The strongest story is not “medical VLMs need an uncertain token.” It is:

> Prior work calibrates how much a model listens to an image. We ask what the
> image supports according to independent readers, where that support is
> converted into stronger language than warranted, and whether the conversion
> can be causally bounded without buying safety through omission.

This is falsifiable, clinically grounded, and resource-conscious. It is not
yet validated; VinDr annotations are the next decisive dependency.

### Post-survey falsification: quantifier coverage

A subsequent real-image probe tested whether positive and negative findings
obey different evidence requirements: a localized lesion can witness presence,
whereas absence should require complete anatomical coverage.  Ground-truth
SLAKE boxes, equal-shape control occlusions, and half-field occlusions made the
intervention causal enough for screening.  The effect replicated only in Hulu:
its positive manipulation passed and 62.5% of eligible partial views retained
a definite negative.  Huatuo and LLaVA-Med failed the required positive
manipulation.  Coverage blindness is therefore an architecture-specific
failure mode, not the sought universal mechanism.  This negative result
raises the bar for the survey's remaining proposal: formal VinDr evidence must
replicate in at least two model families before any decoder is admitted.

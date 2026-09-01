# Medical-VLM Hallucination Mechanism Frontier Scan

**Freeze date:** 2026-08-02  
**Decision:** do not promote DICOM-render stability or perturbation contrastive
decoding as the new mechanism.  Retain one high-priority mechanism,
**Clinical-Equivalence Composition Defect**, and one riskier OE-first competitor,
**Specificity Ratchet**.  Keep observability-boundary errors as an evaluation
axis, not a paper idea.

## 1. The decision in one paragraph

The project's negative results rule out an attractive but now unsupported
story: neither the failed Two-Plane screen nor the failed Virtual-Reader model
shows that these VLMs possess a clean, decoder-erased reader-uncertainty state.
The previous style work also shows that raw view sensitivity is mostly a weak
decision-margin phenomenon and that active style repair can harm OE.  Recent
work makes the remaining collision stricter: paraphrase sensitivity, benign
visual perturbation sensitivity, semantics-preserving medical contrastive
decoding, counterfactual visual grounding, and token-level stability decoding
are already crowded.  A standalone claim that “clinically equivalent DICOM
renders expose hallucination, therefore ensemble/contrast them” would be a
cosmetic variant.  What has not been found in the retrieved literature is the
more specific hypothesis that two independently valid equivalence operations
fail to **compose additively** at the cross-modal binding interface, with the interaction
measured in independent reader-vote units and tied to clinical error beyond
clean margin and both marginal sensitivities.  That is the only DICOM branch
worth testing.

## 2. Frozen local evidence and non-negotiable boundaries

This scan treats the following local results as constraints, not premises to
reinterpret:

- Two-Plane clarity erasure failed on Huatuo: best early-minus-final clarity
  AUROC was `-0.040`, image-bootstrap CI `[-0.198, 0.115]`.
- The fixed Virtual-Reader model was worse than the finding prior and the
  unconstrained evidence-only model; confirmation was correctly not spent.
- The original Evidence-Survival run had a BF16 readout floor and a
  target-dependent ROI; generic progressive visual-evidence decay is also
  collision-heavy.
- Mild style transforms flipped at most about 3.13% in the earlier audit;
  original margin was a better error predictor, and attempted active style
  correction harmed OE.
- The valid substrate is now unusually strong: eight findings, all four
  `0/3..3/3` bins, 3,200 claims, 2,341 image-disjoint VinDr images, exact three
  independent readers, complete DICOMs under `/workspace/vinbigdata/train`,
  and conformant Huatuo/Hulu dev hidden states.
- The completed CPU render audit covers 160 claims over 154 unique images with
  zero processing errors.  DICOM window center/width was present for 149/160
  claims (11 used the frozen fallback), all declared VOI functions were LINEAR,
  and the model routing composition was M2/M1 = 110/50.  Global computational
  guards pass for `native_linear` (160/160), center `-0.05W` (160/160), center
  `+0.05W` (159/160), and width `x1.25` (160/160).  Width `x0.8` and blank-border
  crop each pass only 118/160 and are therefore excluded by the frozen 95%
  rule; SIGMOID remains secondary.  This is pipeline validity, not clinical
  equivalence.
- Existing 200-case native OE outputs for LLaVA-Med, Hulu, and Huatuo are useful
  for discovering candidate claims, but their lexical scores are not clinical
  truth.
- Any OE mitigation must preserve matched positive-claim coverage, claim count
  or fixed `K`, polarity, answer length, refusal rate, and uncertainty burden.
  “Say less”, “say normal”, “hedge everything”, and “refuse” are failures.

The relevant local artifacts are:

- reader manifest:
  `/home/dbw/datasets/physionet/vindr-cxr/1.0.0/manifests_v2/reader_vote_manifest_v2.jsonl`;
- Huatuo hidden states:
  `corrected_runs/vindr_v2/hidden_dev_huatuo_all_findings_v3/`;
- Hulu hidden states:
  `corrected_runs/vindr_v2/hidden_dev_hulu_all_findings_v1/`;
- DICOM renderer:
  `anchor/corrected_sgta/run_huatuo_dicom_render_pilot_v1.py`;
- native OE outputs:
  `corrected_runs/unified_eval/full/{huatuo,hulu,llava}_native_vqa_rad_oe_*/answers.jsonl`.

## 3. Collision audit

### 3.1 Why raw DICOM clinical-render equivalence is not novel enough

The DICOM question is clinically legitimate.  A 2025 study of 195,724 retained
chest radiographs found that raw, DICOM-LUT-processed, and histogram-equalized
preprocessing changed pneumothorax classifier generalization
([J Imaging Inform Med, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12572459/)).
Acquisition parameters including window width, field of view, and view position
also influence learned demographic signals and downstream bias
([Nature Communications, 2024](https://www.nature.com/articles/s41467-024-52003-3)).
These papers justify auditing the render pipeline, but neither supplies a
medical-VLM hallucination mechanism or reader-vote calibration.

Unfortunately, the method-shaped version is already occupied.  The January
2026 preprint *Semantics Preserving Contrastive Decoding for Hallucination
Robust Medical VLMs* explicitly contrasts an image with proximity-constrained
counterfactuals; its LD-VCD branch includes small style/illumination edits such
as window/level changes
([paper record and full-text abstract](https://www.researchgate.net/publication/399488093_Semantics_Preserving_Contrastive_Decoding_for_Hallucination_Robust_Medical_Vision-Language_Models)).
VGS-Decoding likewise scores tokens through original-versus-distorted image
differences ([arXiv:2603.20314](https://arxiv.org/abs/2603.20314)), while LENS
builds training and decoding around a stability gap under
semantics-preserving counterfactual views
([OpenReview](https://openreview.net/forum?id=oh3c2ieVab)).  VASE already uses
weak image transformations to improve semantic-entropy estimation in medical
VQA ([arXiv:2503.20504](https://arxiv.org/abs/2503.20504)).  Generic decoding
competition is even denser: PND uses positive/negative visual paths
([CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_Breaking_the_Illusion_When_Positive_Meets_Negative_in_Multimodal_Decoding_CVPR_2026_paper.html))
and DPA adapts perturbations during decoding
([CVPR Findings 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Bai_Alleviating_Hallucinations_in_Large_Vision-Language_Models_via_Decoding-Time_Perturbation_Adaptation_CVPRF_2026_paper.html)).

**Collision verdict:** DICOM fidelity is a valid experimental tool, but
“DICOM render orbit + stability score/contrastive decoding” fails the novelty
gate.  It may be a baseline or control, not the paper's mechanism.

### 3.2 Why generic prompt/view instability is also occupied

PSF-Med contains 19,748 CXR questions and roughly 92,000 meaning-preserving
paraphrases; six medical VLMs flip 8--58%, and a MedGemma SAE feature at layer
17 causally tracks prompt framing
([arXiv:2602.21428](https://arxiv.org/abs/2602.21428)).  MM-R3 jointly evaluates
question rephrasing, image restyling, and context reasoning
([OpenReview](https://openreview.net/forum?id=70YeidEcYR)).  A large general VQA
study similarly tests benign visual and textual perturbations and finds that
stability predicts correctness
([arXiv:2511.11206](https://arxiv.org/abs/2511.11206)).  The ACL 2026 prompt-
induced hallucination study further identifies a small set of model-specific
attention heads whose ablation reduces prompt-induced hallucination by at
least 40%
([ACL 2026](https://aclanthology.org/2026.acl-long.1941/)).

**Collision verdict:** reporting two marginal flip rates, or simply combining
the two perturbation families, is not a contribution.  The surviving object
must be the **non-additive cross-modal interaction**, not either marginal
sensitivity.

Even that phrase is not sufficient by itself.  *Treble Counterfactual VLMs*
already estimates vision, text, and cross-modal natural direct effects under
counterfactual perturbations and uses them for test-time hallucination
mitigation ([arXiv:2503.06169](https://arxiv.org/abs/2503.06169)).  CECD can only
survive as the narrower clinical result: a product of two independently
admitted clinical equivalence operations, calibrated in independent
radiologist-vote units, whose interaction adds held-out clinical-error
information beyond both main effects and a Treble-style counterfactual
interaction baseline.  If that stricter delta disappears, CECD is not novel.

The August 2026 re-audit tightens the boundary again.  CounterVHD already
extracts visually verifiable medical entities and detects clinical
hallucinations through factual/counterfactual grounding confidence and overlap
([arXiv:2606.28520](https://arxiv.org/abs/2606.28520)); CIPHER learns a
hallucination subspace from diffusion-edited visual counterfactuals and projects
it out at inference ([arXiv:2603.10470](https://arxiv.org/abs/2603.10470)).
Therefore neither counterfactual grounding uncertainty nor a generic
perturbation-derived steering subspace can be claimed as CECD's contribution.
The first 160-claim CECD screen is staged: it must first beat clean margin,
render and prompt main effects, and a matched full-orbit control.  Only a pass
authorizes the cost of an exact Treble reproduction and causal localization;
the paper claim remains forbidden until it also beats that second-stage
closest-work baseline.

### 3.3 Why generic early-signal/late-loss and uncertainty stories are retired

The Hidden Life of Tokens already reports gradual visual-information loss and
hidden genuine information during decoding
([ICML 2025 / OpenReview](https://openreview.net/forum?id=7BKcLeHQsm)).
Visual-token epistemic uncertainty has also been directly studied as a driver
of hallucination
([NeurIPS 2025 / OpenReview](https://openreview.net/forum?id=it0kkaFFpK)).
Medical grounding itself is now the subject of an ICLR 2026 study
([OpenReview](https://openreview.net/forum?id=dXshexyFKx)).  Locally, the
necessary Two-Plane and Virtual-Reader gates failed.  No new proposal below
depends on a universal early layer, a universal uncertainty direction, or a
generic “language prior overwhelms vision” narrative.

This exclusion is strengthened by two July 2026 papers.  VLI already performs
instance-specific conflict localization and bi-causal latent steering
([ACL 2026](https://aclanthology.org/2026.acl-long.1784/)), while HalluTrace
separates visual-grounding failure, language-prior dominance, and cross-modal
conflict with component interventions
([ALVR 2026](https://aclanthology.org/2026.alvr-main.29/)).  A successful CECD
paper must therefore identify a *clinical equivalence composition failure*,
not rename modality conflict or adaptive steering.

## 4. Candidate A — Clinical-Equivalence Composition Defect (CECD)

### 4.1 Research question

> When a clinically valid DICOM display operation and a meaning-preserving
> clinical question rewrite are each nuisance transformations, do they remain
> separable inside a medical VLM, or does their composition create a new
> support-to-commitment error that neither transformation causes alone?

This changes the unit of analysis from an image or prompt to a **product
equivalence class**.  Let `r` index physician-admitted clinical renders, `p`
index speech-act-preserving paraphrases, and `m_{irp}` be the signed claim score
for image `i`:

\[
m_{irp}=\log P(\text{present}\mid r(x_i),p(q_c))-
          \log P(\text{absent}\mid r(x_i),p(q_c)).
\]

The baseline-free composition defect is the two-way centered interaction, or
discrete mixed difference,

\[
I_{irp}=m_{irp}-\bar m_{ir\cdot}-\bar m_{i\cdot p}+\bar m_{i\cdot\cdot}.
\]

`I=0` means the two nuisance actions are additive at the decision surface.  A
large `I` means that visually and linguistically equivalent paths land at
different clinical commitments.  Its scale is made interpretable by dividing
by the cross-fitted per-finding change in `m` for one adjacent VinDr reader-vote
bin.  The paper therefore reports interaction in **reader-vote equivalents**,
not pixels, cosine distance, or flip percentage.

This is not an algebraic commutator or an order-of-operations claim: rendering
and wording act on separate input slots.  The falsifiable clinical claim is
narrower: after transformations
are independently admitted as preserving reader-visible support, their
non-additive interaction should not predict wrong polarity or overcommitment.

### 4.2 Mechanistic prediction

If the failure is at cross-modal binding rather than independent perception or
wording sensitivity, all of the following must hold:

1. the render-by-paraphrase interaction remains after conditioning on clean
   margin, render main effect, paraphrase main effect, finding, vote bin,
   acquisition view, and output entropy;
2. interaction energy predicts claim error or support--commitment gap on
   held-out images better than clean margin plus both marginal sensitivities;
3. identity-image duplicates and tokenization-matched duplicate prompts have
   near-zero interaction, ruling out nondeterminism and score extraction bugs;
4. same-support image swaps are more disruptive in the expected perception
   direction but do not reproduce the same structured interaction;
5. the layer at which interaction becomes error-predictive may differ between
   Huatuo and Hulu.  No common “early layer” is assumed.

### 4.3 Transform contract

Only primary continuous renders may enter the mechanism gate:

- canonical baseline;
- native DICOM LINEAR VOI when present and valid;
- center `-0.05 × width` and `+0.05 × width`;
- width `×1.25`.

The completed 160-claim CPU audit admits exactly these four non-baseline
families at the global 95% computational-validity threshold.  Width `×0.8` and
conservative blank-border crop are excluded before model scoring because each
passed only 118/160.  Center `+0.05W` passed 159/160, so its one failed
image-claim is excluded by the per-sample guard rather than silently retained.

Polarity inversion, 32-pixel downsampling, histogram equalization, arbitrary
gamma, and target-box-conditioned transforms are positive controls or excluded.
SIGMOID is primary only when the DICOM declares it.  The renderer's current
label-independent saturation, edge, and crop-retention checks are necessary,
not sufficient.  Before scientific scoring, two blinded clinical readers must
admit each primary render family on a 60-image stratified sample; any family
that changes the supported/refuted/undetermined judgment in more than 5% of
pairs, or systematically changes lesion visibility, is removed before model
outputs are inspected.

Paraphrases must preserve both proposition and speech act.  Valid examples are
“Is there pleural effusion?”, “Does this radiograph show pleural effusion?”,
and “Can pleural effusion be seen on this radiograph?”.  “Rule out effusion”,
“What abnormalities are present?”, negative-obligation wording, added history,
and requests for certainty are not paraphrases; they are separate pragmatic
interventions.  A clinician plus a language annotator should blindly accept
each template before use.  Token-length and finding-name tokenization are
recorded and controlled.

### 4.4 Minimal decisive experiment

Use four candidate findings initially, each subject to the existing frozen
directional-admission gate: `aortic_enlargement`, `cardiomegaly`,
`pleural_effusion`, and `pulmonary_fibrosis` (substitute only before outputs if
directional admission or a renderer validity audit fails).  Sample 10 dev images
from each of four vote bins per finding: `4 × 4 × 10 = 160` image-claims.  The
four admitted non-baseline renders plus the canonical baseline crossed
with three paraphrases form a maximum 2,400-score factorial per model (2,385 if
the single failed `+0.05W` cell is omitted).  For the first economical screen,
freeze baseline plus three non-baseline families before outputs, yielding
`4 render × 3 paraphrase = 1,920` scores per model; Huatuo and Hulu total 3,840
and are the mandatory model families.  The held-out admitted family and
remaining findings are reserved for confirmation, not researcher degrees of
freedom.

Fit a preregistered mixed model on dev:

```text
signed_score ~ reader_vote + render + paraphrase + render:paraphrase
             + clean_margin + acquisition_view + finding
             + (1 | image_id)
```

Use an image-cluster permutation likelihood-ratio test for the interaction and
image-cluster bootstrap CIs.  Separately evaluate grouped-CV prediction of
reader polarity error and excessive commitment.  The interaction gate passes
only if:

- normalized interaction RMS has a bootstrap lower bound above zero and a
  point estimate of at least `0.25` adjacent-reader-bin equivalents;
- adding interaction residuals to `clean margin + render sensitivity + prompt
  sensitivity` improves grouped-CV AUROC by at least `0.03`, with a 95% CI
  excluding zero;
- the sign is clinically harmful rather than merely variable: high interaction
  increases unsupported positive/negative commitment;
- at least three of four findings in **both** Huatuo and Hulu show the same
  direction; and
- identity controls are below one tenth of the clinical-render interaction.

These thresholds are deliberately stronger than “some flips occurred.”  If
only low-margin samples move, only one model passes, or the interaction adds no
error information beyond marginal sensitivities, CECD is rejected before the
confirmation split.

### 4.5 Causal localization and minimal mitigation

Only after the behavioral gate passes, collect the same factorial at projector
output and four architecture-relative decoder depths.  At each layer, train
all probes on dev with image-grouped nested CV.  The mechanism is localized
only if the interaction's error-predictive component rises at a reproducible
transition and activation patching across the render--prompt cells changes the
final interaction while norm-matched random and polarity directions do not.
Architecture-specific layers are expected and reported.

The natural intervention is **additive-orbit projection**, not another raw
contrastive decoder.  For the canonical cell, replace its score by the additive
projection

\[
\widetilde m_{i00}=\bar m_{i0\cdot}+\bar m_{i\cdot0}-\bar m_{i\cdot\cdot},
\]

which removes only the render--language interaction and retains their two main
effects.  A `2 × 2` orbit gives a four-pass implementation.  For OE, generate
the same draft candidates once, teacher-force each atomic claim over the four
cells, keep positive `K` fixed, and use the corrected score only to exchange or
restate claims.  Certainty is evaluated separately.

Required baselines are clean margin calibration, temperature scaling,
render-only averaging, paraphrase-only averaging, full orbit averaging,
majority vote, a Treble-style modality/cross-modal NDE control, VCD/M3ID where
admissible, and paper-native SPCD/LENS/VGS if
reproducible licensing and checkpoints permit.  CECD succeeds as mitigation
only if positive-content hallucination falls at least 20% relative at fixed
`K`, omission does not increase, clear-case accuracy drops at most 1 pp, and
the additive projection beats the equally expensive full-orbit ensemble.  If
it merely ensembles away variance, it is a robustness trick, not the mechanism.

### 4.6 Fatal-flaw audit

| Fatal risk | Why it can invalidate the result | Fail-closed response |
|---|---|---|
| The DICOM transform changes clinical visibility | Then the “equivalence” premise is false | Blind reader admission before model scoring; remove the family |
| The paraphrase changes pragmatic force | Then interaction is ordinary prompt conditioning | Preserve proposition and speech act; audit negative-obligation separately |
| Neural logits are naturally nonlinear | Nonzero interaction alone is unsurprising | Require incremental prediction of independently judged clinical error beyond margin and both main effects |
| Four-pass correction is only ensembling | No mechanism-specific method remains | Must beat full-orbit averaging at matched compute and coverage |
| SPCD/LENS already captures the same gain | Novelty becomes cosmetic | Treat them as direct baselines; prune CECD if their score explains the interaction or matches correction |
| Reader votes label prevalence, not display invariance | Vote bins cannot certify a render | Separate transform-validity annotation from VinDr support calibration |

## 5. Candidate B — Specificity Ratchet

### 5.1 Research question

> In open clinical generation, does the model start from a visually supported
> coarse observation and then move to a more specific child claim whose added
> anatomy, severity, subtype, or etiology is not supported by additional image
> evidence?

Define a physician-vetted partial order `c_child => c_parent`, where the child
contains every commitment in the parent plus one new clinical constraint.  For
example:

```text
pleural effusion
  -> left pleural effusion
  -> small left pleural effusion

focal pulmonary opacity
  -> left lower-zone focal opacity
  -> left lower-zone consolidation
  -> pneumonia                 # only if explicitly judged image-observable
```

For a valid edge, evidence monotonicity requires

\[
S(x,c_{child}) \le S(x,c_{parent}).
\]

The proposed failure is not merely a wrong child.  It is a **ratchet** if the
decoder child-minus-parent support gap grows toward the final claim token or
across autoregressive continuation while image dependence does not grow.

### 5.2 Novelty boundary

FINER shows that MLLMs hallucinate under fine-grained negative queries when a
subtle mismatch co-occurs with genuinely present entities
([CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_FINER_MLLMs_Hallucinate_under_Fine-grained_Negative_Queries_CVPR_2026_paper.html)).
Formal clinical verification can detect impressions not entailed by a model's
own generated findings
([arXiv:2602.24111](https://arxiv.org/abs/2602.24111)), and CoEV performs
counter-evidence verification between text assertions and visual regions
([arXiv:2606.18609](https://arxiv.org/abs/2606.18609)).  ZINA edits fine-grained
hallucinated spans
([CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wada_ZINA_Multimodal_Fine-grained_Hallucination_Detection_and_Editing_CVPR_2026_paper.html)).
CEBC is an especially close method-level collision: it uses conformally
calibrated detector evidence to minimally edit or suppress unsupported object
mentions while regularizing length and lexical drift
([ACL 2026](https://aclanthology.org/2026.acl-long.2142/)).  CounterVHD already
detects unsupported medical entities using factual/counterfactual visual
grounding uncertainty
([arXiv:2606.28520](https://arxiv.org/abs/2606.28520)).

Therefore, an ontology checker or “replace detail by a parent” method is not
novel.  Specificity Ratchet survives only if it demonstrates the internal,
causal child-over-parent escalation against **image-grounded physician truth**.
FINER is query-side and general-domain; the verifier guarantees consistency
with self-generated premises, not image truth.  The proposed unit is instead a
generated clinical entailment edge whose incremental evidence is separately
adjudicated.  Minimal backoff is consequently an evaluation instrument rather
than the primary novelty claim unless it beats CEBC-style minimal editing and
CounterVHD-style entity grounding at fixed claim count and physician-rated
clinical usefulness.

### 5.3 Minimal decisive experiment

VinDr's eight reader-vote labels are not an ontology and cannot truth most
child attributes.  Do not infer chains such as `lung_opacity -> fibrosis` from
co-occurrence, and do not use RadGraph or an LLM judge to create truth.  Use
VinDr only to stratify/select images and the existing OE outputs only to
discover frequent candidate wording.

Before viewing model scores, two clinicians should freeze 4--6 valid chains,
then annotate 120--160 stratified images at every node as
`supported/refuted/unobservable`, including whether each edge is visually
decidable from a single radiograph.  Etiology nodes such as pneumonia are
excluded unless independently admitted as image-observable for that case.

For each parent--child edge:

1. teacher-force identical declarative templates and compute normalized
   sequence log support for parent and child;
2. collect projector and architecture-relative decoder states for the first
   token that adds the child constraint;
3. compare original image, same-support image swap, image-null as secondary
   OOD control, and noun-frequency/token-length-matched claims;
4. test whether `child-parent` score increases across layers or after a
   parent-positive prefix specifically on physician-refuted/unobservable
   children;
5. patch only the incremental child representation from a matched supported
   case and test whether child commitment changes without flipping the parent.

The gate requires both models, at least four valid chains, clustered CI above
zero for final-minus-mid increase, and grouped-CV error prediction at least
`0.03 AUROC` beyond parent score, child language prior, token length, and claim
frequency.  Causal patching must change unsupported child commitment while
parent identity/polarity changes in less than 1% of clear cases.

The minimal mitigation is **evidence-monotone realization**: when a selected
child violates the independently calibrated edge constraint, realize the
nearest supported ancestor in the same one-claim slot.  It keeps `K` fixed and
does not delete a finding.  Success nevertheless requires reporting semantic
information loss and showing that physician-rated usefulness does not fall;
otherwise the method is just vagueness.  It must beat an ontology-only
postprocessor, formal verification, CoEV, and a length/frequency-matched generic
backoff rule.

### 5.4 Fatal-flaw audit

This direction is scientifically appealing but materially riskier than CECD.
Clinical taxonomies are not always entailment trees; “opacity,”
“consolidation,” and “pneumonia” mix observation and inference, and report
omission does not imply absence.  A child can be visually justified even when
the reference uses only the parent.  The only defensible ground truth is
case-level physician adjudication of the **incremental constraint**.  If fewer
than four chains have reliable visual observability, if the effect disappears
after language-frequency controls, or if mitigation gains come from generic
wording and reduced clinical usefulness, terminate the branch.

A subsequent label-only audit tested whether VinDr's independent R8/R9/R10
bounding boxes could remove this annotation bottleneck for spatial modifiers.
It failed the frozen gate: among lung opacity, nodule/mass, pleural effusion,
and pulmonary fibrosis, only pulmonary fibrosis had adequate two-class
image-hemifield counts and agreement across pilot/dev/test.  Patient/study and
orientation tags were absent, and “both hemifields” was nearly confounded with
multi-box annotation style.  Raw box extent is therefore not promoted to
unilateral/bilateral or upper/lower clinical truth, and no GPU experiment is
authorized from that substrate alone.  See
`docs/SPATIAL_SPECIFICITY_RATCHET_VINDR_PROTOCOL.md`.

## 6. Candidate C — Observability Boundary Crossing (audit only; reject as main)

A model can cross from image-observable findings to claims requiring history,
laboratory values, priors, pathology, or external knowledge.  This is clinically
important and belongs in the common claim contract.  It is not currently a
strong paper mechanism: MedHEval already separates visual, knowledge, and
context failures ([arXiv:2503.02157](https://arxiv.org/abs/2503.02157)); formal
verification, CoEV, missing-image/context studies, and clinical-reasoning
faithfulness studies occupy nearby territory.  VinDr reader votes provide no
truth for etiology, treatment, or prognosis.  Keep `unobservable` and
`evidence_source` labels in physician review, but do not spend mechanism GPU or
claim novelty unless a later result identifies a specific latent source-switch
with causal, cross-model evidence.

## 7. Comparative score and recommendation

Scores use the mechanism-discovery rubric on 0--3 scales: importance `I`,
mechanism clarity `M`, non-redundancy `N`, and executable leverage `E`; risk is
subtracted after averaging.

| Candidate | I | M | N | E | Risk | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Clinical-Equivalence Composition Defect | 3 | 3 | 2 | 3 | -0.2 | **2.55, run first** |
| Specificity Ratchet | 3 | 3 | 2 | 2 | -0.3 | **2.20, physician-label pilot in parallel** |
| Observability Boundary Crossing | 3 | 2 | 1 | 1 | -0.5 | **1.25, evaluation axis only** |
| Raw DICOM stability/contrastive decoding | 2 | 2 | 0 | 3 | -0.5 | **1.25, direct collision; baseline only** |

CECD remains first under reasonable weights because it uniquely exploits all
three scarce assets already available here: raw DICOM render semantics,
independent reader-vote units, and two native medical-VLM families.  Its main
weakness is active competition from perturbation decoding; the composition-defect and
incremental-error gates are what prevent it from collapsing into that literature.
Specificity Ratchet has the stronger OE story but cannot be made trustworthy
from VinDr labels alone, so physician annotation is its true bottleneck.

## 8. Execution order and stopping policy

1. Finish the CPU renderer integrity audit.  Fix tuple/schema/invariance bugs
   before interpreting any model output; identity duplication must be exact.
2. Freeze clinician-audited render families and three true paraphrases.  Do not
   include negative-obligation or abnormality-presupposing prompts in CECD.
3. Run the 160-claim Huatuo/Hulu CECD dev factorial.  Analyze the interaction
   against clean margin and both marginal sensitivities before collecting new
   hidden states.
4. In parallel, mine the completed OE drafts only for recurring parent--child
   phrases, then ask clinicians to freeze/adjudicate the small specificity
   chain set.  Model text must not define labels.
5. If CECD fails either two-model or incremental-AUROC gate, stop it; do not tune
   transforms, thresholds, or paraphrases after outputs.  Move to the
   physician-grounded Specificity Ratchet pilot.
6. If CECD passes, open the architecture-relative hidden-state/patching stage.
   Only a causal interaction result authorizes additive-orbit projection.
7. No OE efficacy claim is allowed until fixed-`K`, matched-coverage physician
   evaluation shows less fabrication without more omission or lost usefulness.

The intended paper story is not “another decoding method.”  It is either:

> Clinical equivalence is not compositional in medical VLMs: individually
> harmless visual and linguistic choices interact to create reader-scale
> diagnostic commitment errors, and projecting out only that interaction
> corrects them without reducing content.

or, if CECD dies:

> Open medical hallucination is an evidence-monotonicity failure: generation
> preserves a supported observation but ratchets it into an unsupported
> clinical subtype, and causal constraint-specific intervention corrects the
> child without deleting the parent.

Both are falsifiable.  Neither borrows support from the failed Two-Plane,
Virtual-Reader, Evidence-Survival, or style-repair branches.

## 9. Source-quality notes

Priority was given to official conference proceedings, OpenReview, arXiv, PMC,
and publisher pages.  The SPCD collision was available during this audit only
through a full-text ResearchGate record rather than a verified arXiv or
proceedings entry; its scientific details should therefore be rechecked if an
official version appears.  This uncertainty does **not** rescue raw DICOM
contrastive decoding because LENS, VGS-Decoding, VASE, PND, and DPA independently
establish a crowded perturbation-decoding neighborhood.  No claim of being the
“first” should be made; the defensible statement is that no mechanism-equivalent
reader-calibrated render-by-language composition defect was retrieved in the searches
completed by the freeze date.

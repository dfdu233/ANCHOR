# Training-Source Echo (TSE): discovery protocol v1 — NO-GO

> **Outcome-blind construct audit (2026-08-03): GPU work is not authorized.**
> The learned source direction is confounded by unmatched clinical content,
> linear guarding does not certify the realized norm-restored intervention,
> and the proposed parent/child hidden interventions are not necessarily the
> same instrument.  This file is retained as an audit trail.  Source-only CPU
> semantics and lineage checks may continue, but no result may cite this v1
> protocol as an admitted causal assay.

## Status and claim ceiling

This protocol is outcome-independent with respect to the proposed STaPE GPU
assay.  Earlier work has already inspected (i) failures of generic DG/style
transport, (ii) a rough keyword census of original PubMed captions, and (iii)
VinDr photometric-tag prevalence.  Those observations are discovery evidence
only.  They cannot be reused as confirmatory outcomes.

The high-ceiling claim tested here is:

> Medical visual adaptation creates a provenance-to-claim channel that is not
> present in the matched pre-medical parent: after clinical polarity has been
> guarded, cues identifying the model's actual training source act as
> claim-specific pseudo-evidence.  Their effect follows the reporting prior of
> that source and is largest when independent readers disagree.

This is not a claim that a universal style center exists, that source likeness
improves visual truth, or that arbitrary image transformations cause medical
hallucination.

## Why this is not the rejected DG hypothesis

The rejected hypothesis asked whether changing image style changes an answer
in a stable signed direction.  STaPE instead requires all of the following:

1. the source coordinate comes from images used by the target model's actual
   training corpus (PubMedVision for HuatuoGPT-Vision);
2. source identity is learned only on source/target cases without asserted
   target findings and is evaluated on held-out source groups and VinDr images;
3. all linearly decodable target-finding polarity is projected out before the
   source intervention;
4. the remaining causal effect must align with a source-only, claim-specific
   assistant-response prior;
5. the effect must interact with independent reader disagreement rather than
   appear as an unconditional global logit or temperature shift;
6. the paper-level estimand is a double counterfactual difference-in-
   differences between a medical checkpoint and its architecture-compatible
   pre-medical parent.

Failure of any item kills the mechanism; generic style sensitivity cannot
rescue it.

## Data separation

### Source semantic view

- Use the assistant responses in
  `PubMedVision_Alignment_VQA.json` and
  `PubMedVision_InstructionTuning_VQA.json`.
- Intersect records by exact image path with the frozen CXR source index.
- Split by PMC article/figure group, never by VQA row.
- Keep alignment and instruction-tuning stages separate.
- Questions and answers are parsed independently.  A question presupposition
  is never an assistant assertion.
- Each claim is assigned exactly one state: positive, negative, uncertain, or
  unmentioned.  Unmentioned is never treated as negative.
- The original caption is provenance/cross-check material only.
- The source-only admission pilot uses the previously frozen,
  outcome-independent quartet: aortic enlargement, cardiomegaly, pleural
  effusion, and pulmonary fibrosis.  No pilot claim is replaced because its
  source frequency is inconvenient.
- A claim-specific paper test requires at least eight eligible claims.  The
  broad candidate ontology is frozen before the quartet pilot is inspected:
  aortic enlargement, atelectasis, calcification, cardiomegaly, clavicle
  fracture, consolidation, pulmonary edema, emphysema, enlarged pulmonary
  artery, interstitial lung disease, infiltration, lung opacity, lung cavity,
  lung cyst, mediastinal shift, lung nodule/mass, pleural effusion, pleural
  thickening, pneumothorax, pulmonary fibrosis, rib fracture, COPD, lung tumor,
  pneumonia, and tuberculosis.  The non-atomic labels `Other lesion`, `Other
  diseases`, and `No finding` are excluded a priori.  Before any claim-logit GPU
  run, claims qualify only through source-only assertion counts, blinded
  extractor quality, and target-only reader-bin counts.  No model score
  participates in eligibility.
- Frozen minimums are 20 positive generic-alignment assertions in source-train,
  five in source-dev, and 100 VinDr images in every 0/3--3/3 reader bin after
  exclusions.  Each qualifying extractor must achieve macro-F1 at least 0.80
  and positive-state precision at least 0.90 on its blinded source-review
  sample.  If fewer than eight claims qualify, the vector-alignment mechanism
  is not tested.

The primary source reporting coordinate is estimated only from assistant
answers to generic, non-presuppositional alignment prompts.  Instruction
responses are a separate presupposition and replication analysis.  No source
prior is admitted until a blinded human review of at least 160 claim-response
units has established acceptable state extraction.  Sparse states are reported
as sparse; smoothing cannot manufacture an admitted prior.  For an eligible
claim `c`, the coordinate is the smoothed logit of a positive assertion in all
generic alignment responses; it is explicitly a reporting/selection prior, not
a disease prevalence estimate.

### Source-identity visual view

- Positive domain: held-out PubMedVision CXR images from source-train groups.
- Negative domain: VinDr development images, with reader votes hidden from the
  domain learner.
- To reduce clinical-content leakage, the source learner uses source cases with
  all four claims unmentioned and VinDr cases with 0/3 support for all four.
  This controls asserted/annotated target findings only; it does not prove that
  all clinical content is matched.
- Discovery images, previous DG/style manifests, semantic-review rows, and the
  final STaPE test images are excluded by a frozen hash registry.
- The split is group-disjoint on the PubMed side and image-disjoint on VinDr.
  No patient-disjoint claim is made without a patient identifier.

## Two guarded coordinates

At a frozen layer `l`, obtain a pooled visual representation `h_l` before any
answer token is generated.

1. Learn a regularized source discriminator on source-train versus target-train
   and freeze its normalized direction `s_l`.
2. On a separate VinDr clinical-development split, learn one reader-polarity
   direction per claim using only clear 0/3 and 3/3 cases.  Let their span be
   `E_l`.
3. Guard the source direction:

   `s_l_perp = normalize((I - P_E_l) s_l)`.

4. Refit no direction on the STaPE test split.  Report source AUROC before and
   after guarding, all principal angles, and the retained norm fraction.

Admission requires, on untouched held-out domain images:

- unguarded source AUROC lower 95% bound above 0.80;
- guarded source AUROC lower 95% bound above 0.70;
- every guarded source-to-polarity cosine has absolute value below 0.05;
- native clinical polarity AUROC changes by less than 0.01 after source
  residualization on the clinical-development split.

If the guarded source coordinate is not identifiable, STaPE stops.

## Causal intervention and double counterfactual

For every frozen VinDr test image and claim, score one-token contextual markers
for absent, uncertain, and present.  Use a neutral claim prompt in which all
three states are explicitly valid.  Generation is not a primary measurement.

At the admitted visual/projector layer, create three norm-preserving states:

- native `h`;
- source-positive `h + alpha*s_l_perp`;
- source-negative `h - alpha*s_l_perp`.

`alpha` is fixed from the 25th-to-75th percentile source-coordinate distance on
domain-development data and is never selected from claim logits.  Restore the
native tokenwise norm after intervention.  A source-subspace-random direction,
a polarity-orthogonal random direction, and a matched isotropic perturbation
are mandatory controls.  All controls use the same token support and L2 norm.

Define polarity `pi = z_present - z_absent` and commitment
`k = logsumexp(z_present,z_absent) - z_uncertain`.  For claim `c`, freeze the
source reporting coordinate `b_c` from the admitted source-only semantic split.

For a single model, first compute the symmetric provenance effect
`D_mic = 0.5 * (score_mic(source-positive) - score_mic(source-negative))`.
The paper-level estimand removes generic architecture/style sensitivity using
an architecture-compatible parent `p`:

`DD_ic = D_medical,ic - D_parent,ic`.

For Huatuo, absence of a defensible multimodal pre-medical parent limits the
result to discovery.  LLaVA-Med may enter the confirmatory family only after
weight/config/tokenizer/vision-tower compatibility with the nominated general
LLaVA parent is audited and frozen; similarity of names is not enough.

An even stronger confirmatory assay uses at least two heterogeneous provenance
instruments whose clinical content is independently controlled: a guarded
internal source coordinate and one anatomy-preserving pixel-space provenance
intervention.  A third instrument may be an outer-frame cue transplant.  No
instrument is selected from its claim-logit outcome.  At least two must produce
the same claim-prior fingerprint after matching source-ID dose and perturbation
norm.

The initial image-clustered model is:

`DD pi_ic = beta0 + beta2*U_ic + beta3*b_c*U_ic + claim_FE`,

where `U` marks the 1/3 and 2/3 reader bins.  The main effect of `b_c` is omitted
because it is collinear with claim fixed effects.  The same model is fit
separately to commitment.

Because claim identity, not the number of image-claim rows, determines the
effective sample size for source-prior alignment, the primary transparent
analysis is also performed in two stages.  For each claim compute

`delta_c = 0.5*((DD_1/3 - DD_0/3) + (DD_2/3 - DD_3/3))`.

Then test the weighted slope and Spearman alignment of `delta_c` with `b_c`
using an exact claim-level permutation test, nested inside an image-cluster
bootstrap.  Fewer than eight eligible claims is an automatic failure of the
claim-specific mechanism test.  Separate 1/3-minus-0/3 and 2/3-minus-3/3
contrasts remain mandatory so cancellation cannot create a result.

## Mechanism gate

STaPE is admitted only if all conditions hold:

1. source semantic extraction passes blinded review and `b_c` is estimable for
   the frozen claims;
2. at least eight claims and the guarded source coordinate pass their held-out
   admissions;
3. the parent-subtracted ambiguity interaction has the predeclared sign, its
   nested image-cluster interval excludes zero, and the exact claim-level
   permutation test is significant;
4. the effect is aligned with `b_c`, not merely a global positive or commitment
   shift;
5. each matched random/nuisance control is smaller, and a permutation of
   `b_c` fails;
6. clear-bin clinical accuracy changes by at most one percentage point;
7. early/independent polarity scores do not move enough to explain the final
   effect;
8. the result survives prompt paraphrase and a temperature-only affine fit;
9. it holds for a majority of the frozen claims and for at least two admitted
   provenance instruments;
10. it replicates in a second parent-to-medical model family with a documented
   training source.

The first Huatuo run is a discovery screen because source-caption frequencies
were previously inspected.  A paper-level claim requires a newly locked source
split or a second model whose source response prior was not inspected before
its protocol was frozen.

## Mitigation authorization

No mitigation is implemented unless the mechanism gate passes.  If it passes,
the only authorized first method is Counterfactual Provenance Residualization:

1. estimate the source-tag contribution by symmetric guarded interventions;
2. subtract only the component aligned with the admitted source response prior;
3. gate subtraction by reader-calibrated ambiguity;
4. preserve claim count and candidate identity for OE evaluation.

It must beat temperature scaling, VCD, random-direction subtraction, output
shortening, and fixed hedging at matched claim coverage.  A reduction obtained
by deleting claims, increasing refusal, or converting every statement to an
uncertain statement is a failure.

# Research Decision: Evidence, Commitment, and Reportability

Decision date: 2026-07-31

## Surviving question

The only current paper-level hypothesis is deliberately narrow:

> For an image-grounded clinical claim, does a medical VLM preserve reader-
> calibrated evidence about whether the claim is clear or ambiguous, yet
> realize that evidence with excessive linguistic commitment?

This is not a generic confidence, calibration, grounding-verifier, or report-
generation claim. It survives only if formal VinDr reader votes show new
information beyond same-layer polarity and final-token confidence on a locked
patient/image split.

## What was pruned

1. **Scalar Commitment-Bounded Decoding.** The mean-token-null signed evidence
   rule caused large accuracy and omission/fabrication regressions in local
   Huatuo and Hulu pilots. It is retained only for reproduction.
2. **Universal early-layer direction erasure.** Direct layer readout produced
   opposite-state direction accuracies of roughly 47--60% for Huatuo,
   44--50% for LLaVA-Med, and 47--87% for Hulu. Hulu improved in later layers.
   Dev-fitted/test-evaluated layer mixers improved Huatuo AUROC by only 0.023
   over calibrated-final and degraded or failed to improve Hulu and LLaVA.
   There is no cross-model early-correct/late-wrong mechanism.
3. **Generic internal grounding signature plus selective rollback.** This is
   now directly occupied by
   [IGS/VGD](https://arxiv.org/abs/2607.27823). Medical hard-negative
   image-swap grounding training is occupied by
   [CORAL](https://arxiv.org/abs/2607.03647), and medical counter-evidence
   correction by [CoEV](https://arxiv.org/abs/2606.18609).
4. **Universal uncertainty laundering by report metrics.** A real RadGraph-XL
   contrast gave the same 0.5 reward to definite and hedged false effusion
   claims against a negative reference, while a negated claim scored 1.0.
   CheXbert uncertainty policies vary across implementations. Metric policy is
   therefore a required control, not the headline mechanism.
5. **Source-domain center or style nuisance correction.** Prior controlled
   experiments found strong source identifiability but no stable relation to
   CE/OE error; active style transforms could introduce clinical harm. No
   renamed style-center, tangent, or augmentation method is admissible.
6. **Semantic-involution / evidence-equivariance decoding.** A held-out Huatuo
   pilot compared a dev-calibrated direct finding margin with (i) an
   antisymmetric present/absent projection and (ii) an A/B-order projection.
   Direct accuracy/AUROC were 0.604/0.740; both projections reached only 0.625
   accuracy and fell to 0.733 AUROC. The A/B projection reduced false positives
   only by doubling positive-case omission from 0.167 to 0.333. This is a
   diagnostic consistency control, not a mitigation candidate. It also collides
   closely with NegVQA, V-Loop, and EqSim.
7. **Slotwise modifier backoff as the main method.** The conceptual distinction
   remains valid: support for a finding does not entail support for its
   location, severity, or temporal modifier, and each slot needs an unspecified
   state. However, a real RadGraph-XL screening audit of 48 Huatuo reports found
   172 image-grounded mapped claims, only 35 parent finding/polarity matches,
   only 15 comparable explicit modifiers, and three different-value candidates.
   Two of those three were plausible anatomy synonyms. The maximum unverified
   addressable mass was therefore 3/172, too small for a general hallucination
   mitigation headline. Slotwise states remain part of the evaluation contract
   and may become a targeted laterality/severity study with physician labels.
8. **Target--input information mismatch as a decoder mechanism.** A conservative
   audit, corrected after two false-positive reviews, found explicit dependence
   on an unavailable prior image, clinical history, or other test in 1,458 of
   6,281 unique report--image instances (23.21%) and 1,807 of 26,439 sentences
   (6.83%). The per-corpus report rates were 53.89% for the local MIMIC adapter,
   3.07% for IU-Xray, and 21.32% for the unverified CheXpert proxy. This is a
   real dataset/evaluation problem, but the proposed mechanism did not survive.
   In a 48-report within-report probe, the correct-over-shuffled image NLL
   advantage differed by only 0.0054 between visible and Tier-A sentences
   (bootstrap CI -0.0317 to 0.0436; one-sided Wilcoxon p=0.333). A stricter
   52-report within-sentence token probe reversed the hypothesis: source-marker
   tokens were *more* affected by removing visual tokens than finding tokens
   (finding-minus-source zero-visual effect -0.1990, CI -0.3409 to -0.0620).
   Therefore the audit may motivate information-matched dataset construction,
   but it cannot support a claim that unavailable supervision is represented as
   a separable language-only commitment mechanism in Huatuo.
9. **Image-presence versus image-identity contrastive decoding as novelty.** The
   failed probe suggests that the presence of visual tokens can license report
   language without comparable sensitivity to the identity of the image. This
   is a useful diagnostic, but language-null, distorted-image, conditional-PMI,
   and retrieval-image contrastive decoding are already occupied by M3ID, VCD,
   C-PMI decoding, and RVCD. It is not an admissible standalone method claim.
10. **Atomic-query elicitation as the main method without a new mechanism.**
    Huatuo's direct atomic finding margin reached 0.740 AUROC while teacher-
    forced open reports showed weak correct-versus-shuffled image sensitivity,
    motivating a competence--elicitation gap measurement. However, report-to-
    VQA reframing is directly represented by QRad, and visual/self-verification
    is represented by V-Loop and auxiliary report auditors. A query-induced
    evidence paper must first demonstrate a new causal mechanism or a clinically
    distinct three-state result on reader-vote truth; draft--verify--rewrite
    alone is too crowded.
11. **Visual-witness or findings-to-diagnosis verification as the headline.**
    A real RadGraph-XL audit found ample `suggestive_of` structure (82 edges
    in 48 Huatuo reports versus 34 in their references), so this is a real
    clinical reporting axis. It is not an open novelty axis: CoEV verifies
    textual assertions against visual regions, while *Toward Guarantees for
    Clinical Reasoning in Vision Language Models via Formal Verification*
    (2026) explicitly checks whether Impression diagnoses are entailed by the
    model's Findings and filters unsupported diagnoses. "No diagnosis without
    a visual witness" is therefore a baseline/control, not a new method.
12. **Inference-edge uncertainty collapse and evidence double-spending.**
    These follow-up hypotheses did not survive the same 48-report screening
    audit. Among positive-source `suggestive_of` edges, the uncertain-target
    rate was approximately 74% for Huatuo and 75% for references, contradicting
    a generic edge-certainty-laundering story. Multi-target source branching
    was 21.5% for Huatuo and 17.2% for references, but after report-length
    normalization the generated reports had fewer rather than more edges per
    100 words. Most inspected branches used legitimate differential language
    such as "or". Neither mechanism has sufficient effect mass for a paper
    method. The audit does expose a mandatory evaluation rule: atomization may
    not erase logical scope (`A or B` is not two independent definite claims).
13. **Report-level multiplicity / claim-count risk as the main method.**
    HalluCXR independently reports response length as a strong hallucination
    risk signal, while ConfLVLM already treats generated details as individual
    hypotheses with conformal risk control, RadFlag provides sentence/report
    risk flags, and *Principled Detection of Hallucinations via Multiple
    Testing* formalizes hallucination detection as multiple testing. A
    Bonferroni-, FDR-, or claim-count-adjusted report threshold is therefore a
    useful baseline but fails the direct-collision novelty gate.
14. **Findings-to-Impression semantic-boundary evidence substitution.** A
    frozen 24-report Huatuo probe teacher-forced the same Impression under
    correct versus naturally shuffled images and under no Findings, matched
    Findings, or length-matched Findings from another report. Matched Findings
    reduced the image-identity NLL advantage from 0.0456 to 0.0078; however,
    the required semantic-specific attenuation relative to mismatched Findings
    was 0.0183 with bootstrap CI [-0.0017, 0.0428]. The frozen gate therefore
    failed. The apparent attenuation is not separable from added-position or
    language-predictability effects and cannot be claimed as a hallucination
    mechanism. Raw records and the predeclared analyzer are retained under
    `corrected_runs/section_substitution/huatuo_mimic_n24_v1/`.
15. **Anatomical coverage as evidence for negative claims.** The conceptual
    asymmetry remains important: failure to see a lesion is not evidence of
    absence unless the relevant anatomy was observable and inspected. It is
    also a substantive boundary beyond generic unanswerable VQA and positive
    lesion-box dropout. However, the frozen low-resource Huatuo screening probe
    failed its positive manipulation check. On 16 report-positive pleural-
    effusion images, equal-area lower-lateral versus upper-lateral masks changed
    the `Yes-Maybe` margin by 0.281 on average but the bootstrap CI
    [-0.0078, 0.6016] included zero. The negative arm is therefore
    uninterpretable, not evidence against coverage. No independent lesion boxes
    exist locally; trying alternate atlas rectangles after seeing the result is
    prohibited. This branch is downgraded for poor current executability.
16. **Visual-null choice instability as the headline.** *Delve into Visual
    Contrastive Decoding* already studies downsampled and edited contrastive
    samples, reports strong model/benchmark variation, and fuses multiple
    samples. Our stricter same-image check also found only modest instability
    between per-image projected-token means and a locked dev-global mean on 16
    Huatuo claims: layerwise evidence-rank Spearman was 0.960--0.984, final-layer
    sign changed for 2/16 boundary cases, and CBD state changed for 1/16. Null
    semantics remain a mandatory control, but null-choice sensitivity itself is
    neither sufficiently new nor sufficiently large here. The remaining delta
    is the already-defined directional clinical response: a negative image must
    induce the clinically ordered claim change, not merely a different output.
17. **Reader calibration or constrained probability projection as the method
    novelty.** [*Diagnostic Uncertainty Calibration*](https://arxiv.org/abs/2007.01659)
    (2020) already formalizes
    higher-order calibration under inter-rater disagreement and predicts
    disagreement probabilities. Two 2026 medical-VQA papers respectively add
    [hallucination signals to post-hoc calibration](https://arxiv.org/abs/2604.02543)
    and train [verbalized confidence](https://arxiv.org/abs/2606.27023) with an
    image-presence × text-integrity factorial design plus Brier/KL objectives.
    Current general-VLM work also uses constrained
    divergence/projection decoding. The RCCP forward-KL cap is retained only as
    a minimal polarity-preserving realization operator. Novelty is admissible
    only for the conjunction of independent reader-vote truth, signed clinical
    response, layerwise support-to-language commitment loss, and OE gains at
    unchanged claim coverage.
18. **Coverage-certified negation as a universal decoder.** Positive and
    negative clinical assertions have an attractive logical asymmetry: one
    localized witness may establish presence, while absence requires adequate
    coverage of the relevant anatomy.  A frozen SLAKE experiment used real
    lesion boxes for the positive manipulation and equal-shape, zero-overlap
    control boxes; the negative arm compared a complete target-negative image
    with left- and right-half mean-fill occlusions.  At n=64, Hulu passed the
    positive manipulation (attenuation 0.582, image-bootstrap 95% CI
    [0.195, 0.984]) and remained definitely negative on 62.5% of 40 eligible
    partial views, despite a prompt that explicitly mapped obscured anatomy to
    `Maybe`.  The same manipulation failed on Huatuo (0.262, CI
    [-0.016, 0.586]) and LLaVA-Med (-0.016, CI [-0.047, 0.015]); their negative
    arms therefore cannot be interpreted.  The earlier favorable Huatuo n=16
    screen was a small-sample false lead and is superseded.  This is a useful
    Hulu boundary and an evaluation invariant, not a general medical-VLM
    mechanism.  It also sits adjacent to Budgeted Conformal Evidence
    Acquisition, which already chooses between answering, abstaining, and
    acquiring zoomed/cropped evidence.  The decoder branch is pruned unless a
    future mechanism explains and predicts the architecture boundary.
19. **Claim involution as a universal debiasing operator.** A fixed
    present/absent or Yes/No antisymmetrization does not reliably isolate
    visual evidence. On the same balanced 32-case SLAKE screen, Huatuo's
    direct and involution AUROC were both 0.611 (bootstrap delta CI
    [-0.037, 0.043]); LLaVA-Med improved from 0.398 to 0.529, but its sign
    accuracy remained 0.5 and the finding-level direction was inconsistent.
    The core mechanism is therefore data-refuted as a cross-model method and
    is retained only as a polarity/order control.
20. **Prior-titrated odds and worst-prior certification.** A frozen probe
    crossed 10%, 50%, and 90% stated background probabilities with 16
    positive and 16 negative SLAKE images for each of Hulu, Huatuo, and
    LLaVA-Med. Patient-level stratified bootstrap inference replaced the
    earlier gray-image threshold and arbitrary positive--negative pairing.
    No common additive evidence-update law emerged. Hulu had a strong
    neutral-prior clinical contrast (1.461, CI [0.844, 2.039]) but raising
    the stated prior *lowered* its real-image margin (-0.242, CI
    [-0.289, -0.195]) and contracted the clinical contrast (-0.156, CI
    [-0.242, -0.063]). Huatuo responded in the intended prior direction but
    its clinical contrast was not significant (0.301, CI [-0.125, 0.727]);
    LLaVA-Med had neither a clinical contrast nor a semantically aligned
    prior response. The natural mitigation---using the minimum support over
    priors as a robustness certificate---also failed: AUROC changed by
    -0.010 for Hulu, +0.008 for Huatuo, and -0.021 for LLaVA-Med, with every
    stratified 95% CI including or lying below zero. This branch is pruned,
    not renamed as counterfactual calibration. Its architecture-specific
    prior reversal remains a diagnostic observation requiring independent
    replication, not a paper claim.
21. **Coverage-Preserving Evidence Transport.** The fixed-K idea passed a
    useful algebraic control but did not establish a method. On a third,
    preregistered 52-image SLAKE cohort, raw claim reranking improved TP for all
    three models, but the pooled precision gain was only 1.54 points with a
    bootstrap 95% CI [-1.28, 4.46]. On the natural MIMIC report dev split, the
    stronger unknown-aware audit decisively failed: supported recall fell from
    26.2% to 19.0%, while 22 refuted baseline claims were mainly converted into
    +27 unverified claims. The apparent verified precision increase from 33.3%
    to 100% is therefore truth-coverage escape, not grounding. Holdout remains
    unopened. The result identifies a missing variable: visual support and
    task-conditioned reportability cannot be collapsed into one ontology rank.
22. **Old full-report baseline table.** The nominally complete seven-method
    MIMIC run is invalid. All seven answer files are byte-identical, contain
    694 copies of the one-token output `The`, and omit six source items. A
    repository-wide audit of 48 historical answer files assigned 0 grade A,
    15 grade B/rescore-only, and 33 grade C/rerun; five groups were byte-
    identical across nominally distinct runs. No historical table is reused
    merely because its line count and metric file exist.
23. **Low-resource bridge while VinDr is unavailable.** A patient-disjoint,
    finding-matched MIMIC screen was frozen from image-grounded RadGraph claims:
    138 claim rows, exactly 46 definite-positive, 46 definite-negative, and 46
    uncertain, over 91 patients and 127 images. It tests whether a scalar
    third-state verbalizer bias can recover radiologist-expressed uncertainty
    without harming definite cases. It is explicitly single-report linguistic
    commitment evidence, not reader-disagreement truth and not a substitute
    for VinDr. Dev must pass an AUROC/calibration/no-harm gate before holdout is
    opened.

## Evaluation correction that remains mandatory

Truth, certainty, and reporting obligation are different axes:

- `reader_support` / `reference_observability`: whether the image supports the
  claim;
- `prediction_polarity` / `prediction_uncertainty`: what content the model
  emits and how strongly it commits;
- `reference_relevance`: whether that task requires, permits, or excludes the
  claim (`required`, `optional`, `out_of_scope`).

A fixed ontology defines the audit universe, not an exhaustive reporting
policy. This distinction is prior-art constrained by
[Pragmatic Radiology Report Generation](https://arxiv.org/abs/2311.17154),
which shows that indications affect negative-finding mentions. It is an
evaluation invariant and control, not a novelty claim.

Claim contract v8 implements this separation. The primary omission metric now
uses only unanimous-positive, task-required claims. Exhaustive-ontology
omission, optional-positive mention rate, and out-of-scope emission are
reported separately. High-support claims are added by the legacy CBD function
only when `required_findings` is explicit.

## Data contract fixed before download

`prepare_vindr_reader_manifest.py` now creates two linked manifests from the
official three-reader CSV:

1. `reader_vote_manifest.jsonl`: balanced finding-image pairs for CE and
   mechanism probes;
2. `oe_listing_reference.jsonl`: every eligible finding for every selected
   image, so OE fabrication and omission denominators are complete.

For the frozen “list visible abnormalities” task only, 3/3 support is
`required`, 1/3 or 2/3 is `optional`, and 0/3 is `out_of_scope`. This policy may
not be transferred to narrative reports, whose relevance needs an indication
or physician reference.

## Hard continuation gate

Continue to a method only if at least two medical VLMs satisfy all of the
following on formal VinDr data:

1. a non-final Claim-Plane probe improves held-out reader-disagreement AUROC by
   at least 0.05 over final-layer polarity/confidence, with an image-cluster
   bootstrap 95% interval excluding zero;
2. the increment is not reproduced by temperature scaling, norm matching,
   random directions, or output shortening;
3. a polarity-preserving intervention reduces disagreement overcommitment with
   at most one percentage point loss on unanimous clear cases and no required-
   claim omission increase.

If this gate fails, the Missing Third State becomes an evaluation/negative-
mechanism paper candidate, not a hallucination mitigation method. No decoder
will be tuned to rescue a failed premise.

After the gate was written, two adjacent 2026 results raised its causal
standard. CheXthought already predicts human--human and human--AI disagreement
from multi-reader images and improves uncertainty communication through
training, so “using reader disagreement” alone is not novel. *Vision-language
models for chest radiography do not always need the image* shows by image-side
intervention that several medical VLMs, including its evaluated LLaVA-Med-7B,
can obtain apparently competitive binary accuracy while ignoring the image.
Consequently, reader-disagreement decodability is interpretable only after a
same-finding, opposite-support image intervention establishes that the tested
model's polarity changes in the clinically correct direction. This is a
precondition, not an extra contribution.

The exact mean-token-null scalar Commitment-Bounded Decoding plan has also
undergone a separate idea-evaluator audit. Because its core intervention was
already beaten by the unchanged Huatuo and Hulu baselines, it is formally
classified **Reject and Pivot** under the data-refuted-mechanism rule. This
does not reject the distinct, still-untested VinDr reader-disagreement question;
it forbids presenting the old scalar decoder as that question's solution.

## Current verification

- The new provenance-first evaluation core has task/model/method registries,
  strict run and sample fingerprints, task-separated evaluators, atomic JSONL
  resume, cluster bootstrap, legacy A/B/C audit, and a SQLite heartbeat queue.
  Its focused suite passes, including prompt snapshots, cache mismatch
  rejection, degeneration detection, and stale-job recovery.
- Open VQA now has a separate strict short-answer evaluator. It rejects any
  question-ID or embedded-reference mismatch, reports normalized exact,
  token-F1, and ROUGE-L only as lexical proxies, resamples whole images rather
  than correlated question rows, and compares every mitigation with greedy on
  paired questions and image-cluster confidence intervals. These proxies do
  not define clinical hallucination correctness; claim adjudication remains a
  separate endpoint.
- The corrected full LLaVA-Med MIMIC report run completed 694/694 but failed
  the model--task qualification gate: 649 outputs repeated one normal template,
  the normal-template rate was 94.5%, and the unique-output rate was 1.9%
  despite a 95.1% abnormal-reference rate. It is a collapse artifact, not a
  report-generation baseline. The corresponding nine-method smoke also
  rejected every method: several loaders attempted an unnecessary network
  lookup despite complete local CLIP weights, while the executable DoLa path
  degenerated to the one-token answer `The`. The new runner is strictly
  offline and includes the actual generation runtime and environment in its
  fingerprint; no full report mitigation matrix is launched from this failed
  base task.
- The MIMIC single-report third-state dev screen completed 138/138 with no
  scoring errors. Hulu assigned zero recall to the uncertain class. A fitted
  scalar `Maybe` bias improved macro recall by 5.2 points but reduced definite-
  case accuracy by 6.25 points; uncertainty-advantage AUROC was 0.593. The
  frozen gate failed, so holdout remains unopened. This prunes scalar
  third-state calibration without making a claim about formal VinDr reader
  disagreement.
- A stricter follow-up asked whether uncertainty is encoded in the competition
  between opposing claims rather than in the `Maybe` token. On dev,
  `-|Yes-No|` reached AUROC 0.660 with subject-bootstrap 95% CI
  [0.562, 0.754], while the direct uncertainty verbalizer reached 0.593.
  However, every threshold that gained third-state recall beyond the binary
  argmax violated the one-point clear-case no-harm constraint; the selected
  feasible threshold was therefore the identity threshold with zero gain.
  This is mechanism evidence for a distributed/overlapping ambiguity signal,
  not an actionable decoder, and holdout remains unopened.
- Hulu passed the 32-report non-collapse qualification screen (26 unique
  outputs, 18.8% normal templates, 81.3% abnormal-finding mentions), and the
  full 694-report generation completed. The first 16-image, three-prompt
  dependency audit found real--null exact-same 0 and token-F1 0.338. Its view
  named `shuffled` was subsequently found to be a pixel permutation of the
  target image, not a naturally mismatched real radiograph; the stored source
  hash also described the original file rather than the actual transformed
  view. Therefore its 0 exact-same/0.319 token-F1 result establishes sensitivity
  to pixel destruction only and is **not** admissible evidence for mismatched-
  image grounding. Audit v3 is preregistered with a different-patient real-CXR
  donor, a separate pixel-shuffled arm, and hashes/identities for the actual
  view presented to the model. A v3 preflight then found that its missing
  patient field had fallen back to image ID: 1/16 nominally shuffled pairs used
  two studies from the same actual patient. v3 was stopped and retained as an
  invalid partial artifact. Audit v4 parses the MIMIC patient from the path and
  constructs a deterministic one-to-one donor permutation constrained to
  different patient and different image; all 16 pairs pass before model load.
  v4 completed 192/192 generations and passed both non-collapse and image-
  dependency gates. Across 48 paired outputs, real--null token-F1 was 0.3382
  and real--different-patient-real-image token-F1 was 0.3389, with zero exact
  matches in both arms. This establishes response dependence on image identity,
  not clinical direction, grounding, or correctness; claim metrics and
  physician adjudication remain mandatory.
- The preregistered v4 surface analysis exposed a narrower mechanism lead.
  Replacing the image changed answer content by 0.6611 on average, while the
  lexical commitment/uncertainty surface changed by only 0.0505; the resulting
  content-minus-commitment dissociation was 0.6106 with image-cluster bootstrap
  95% CI [0.5752, 0.6498]. However, both real and mismatched outputs had the
  same 0.0253 uncertainty-marker rate. The apparent stability may therefore be
  partly an uncertainty floor, and the lexicon is not a claim-truth measure.
  The result is retained as a lead for the formal reader-support experiment:
  image identity can change *what* is said while the decoder remains almost
  uniformly definite about *how certain* it is. It is not authorized as a
  paper claim before VinDr reader votes and claim/physician adjudication.
- The official VQA-RAD test split was recovered from a CC0 parquet artifact
  with provenance hash. Removing the 251 normalized Yes/No answers leaves 200
  genuine OE questions on 120 content-addressed images. The first smoke exposed
  an invalid report-derived gate that rejected legitimate one-word VQA answers;
  the corrected task-specific audit now permits short answers while retaining
  exact qid alignment, non-empty-output, and dominance checks. A deeper source
  audit then found fixed `34:34+576` visual-token ranges in the released
  PAI/AVISC/M3ID/DAMRO ports. Runs produced under those offsets are invalid even
  when they finish. The common runner now derives the placeholder position and
  patch count per sample, fails closed on an out-of-range mask, and gives M3ID
  a correctly sized text-only cache. However, the completed v3 full run exposed
  a second shared-port failure that its 32-item diversity gate had missed: all
  nine methods produced 200/200 structurally aligned rows, but 97.5--99.0% of
  predictions were bare function-word fragments such as `The`, `This`, `In`,
  or `On`. Greedy normalized exact and token-F1 were both zero and its semantic
  empty rate was 85.5%. The matrix is now machine-marked
  `common_plumbing_valid=false` with no scientifically comparable methods;
  none of its method comparisons are reusable. Smoke now explicitly rejects a
  >=50% function-word-only rate. A frozen four-image canonical-versus-custom-
  port diagnostic localized the failure to the keyword stopping criterion:
  the stopping-enabled port emitted `The` on 4/4 cases, whereas disabling it
  matched canonical greedy exactly on 4/4 cases (normalized exact and token-F1
  both 1.0). Thus v1--v3 remain unusable audit artifacts, but the model port
  itself is not rejected; every corrected method must disable the faulty
  stopper and pass the larger backend-identity gate before reuse.
- SECOND (ICML 2025) is integrated through its official lmms-eval fork at
  commit `4ad65872d9c03ea7b60ea68c2b663d22a373ec33`. The official repository
  lacked a recursive Mistral model despite routing LLaVA-Med-v1.5 to Mistral;
  a thin subclass exposes the unchanged official recursion methods on
  `LlavaMistralForCausalLM`. A local lmms task loads all 200 frozen OE rows and
  their content-addressed images offline, keeps the common prompt and 64-token
  budget, imports per-sample logs into the strict answer contract, and runs a
  32-item fail-closed smoke before full inference. Its first recursive launch
  failed because the fork passed a CHW tensor to NCHW-only interpolation; the
  source now preserves the batch dimension. That repair was insufficient for
  scientific admission. On the frozen 32 cases, standard generation from the
  SECOND fork reached only 0.906 normalized identity and 0.967 mean token-F1
  against canonical LLaVA-Med, below the preregistered 0.95/0.98 thresholds,
  with substantive semantic differences. A separate method-native recursion
  canary then failed before its first token because `CLIPVisionTower` lacks the
  `image_attentions` consumed by SECOND's `get_heatmap`. SECOND is therefore a
  blocked/non-executable baseline for this official Mistral checkpoint path,
  not a mitigation result. Its artifacts are retained, and its failure does
  not authorize lowering identity thresholds or silently substituting an
  invasive unofficial implementation.
- Canonical native LLaVA-Med and native Hulu completed the frozen 200-question
  VQA-RAD OE split with exact qid/reference alignment and no empty outputs.
  Their lexical token-F1 proxies are 0.107 and 0.132. These establish valid raw
  generation baselines only; they do not replace claim-level or physician
  adjudication of clinical correctness.
- The stopper-corrected common mitigation fork completed all nine 32-case
  smoke executions without empty or function-word-only collapse. It still
  failed admission before any 200-case run: its greedy backend achieved 0.938
  normalized identity and 0.972 token-F1 against the frozen canonical native
  backend, below the fixed 0.95/0.98 thresholds. The two mismatches changed
  anatomy or diagnosis, so they cannot be dismissed as surface variation.
  Executability is recorded separately from comparability; no DoLa, PAI,
  OPERA, AVISC, M3ID, VCD, or DAMRO effect is reported from this fork.
- Native Huatuo passed its independent 32-case structural qualification with
  exact qid alignment, no empty or function-word-only answers, and all outputs
  unique, then completed all 200 questions. Its lexical token-F1 was 0.050
  (image-cluster 95% CI 0.042--0.059), median answer length was 47 tokens, and
  82.5% of answers hit the 64-token generation budget. Uniform v2 diagnostics
  for all three native models explicitly report reference length, answer-
  expansion ratio, lexical reference-phrase coverage, terminal punctuation,
  and budget hits without treating any of them as hallucination correctness.
- A post-hoc brevity-control baseline now traces first-sentence and 8--64-word
  prefix policies with paired image-cluster bootstrap. Large short-prefix F1
  gains for Hulu and Huatuo exchange away lexical reference coverage. Near the
  point-estimate coverage boundary, Hulu's 48-word delta is only +0.00029 F1
  with a CI crossing zero; Huatuo's 40-word delta is +0.00991 but its coverage
  CI permits a 0.026 loss. LLaVA first-sentence preserves lexical coverage and
  gains +0.00382 F1 (CI +0.00035 to +0.00791). This is a mandatory brevity
  control, not a mitigation method; future claims require clinical claim-level
  coverage and must beat it without length, omission, or refusal exchange.
- A deterministic physician-review export selects 100 image-disjoint frozen
  VQA-RAD questions without looking at model scores and groups three randomly
  ordered, identity-blinded native answers per image (300 answer units). The
  private identity mapping, source answers, manifest, blind bundle, and every
  selected image are SHA-256 locked. Review first adjudicates reference
  observability/required claims, then atomizes each candidate while keeping
  `visual_support` separate from `commitment` and routing knowledge or
  unobservable claims away from visual-hallucination scoring. This prepares
  human truth; no unfilled bundle is treated as evidence.
- A detached watchdog now reloads an explicit active-job manifest every 30
  seconds and restarts only dead `running/starting` jobs from their recorded
  commands. Failed scientific configurations remain failed for audit. The
  pipeline and its logs survive VS Code/SSH disconnection independently of the
  front-end Codex session. The VQA pipeline has an exclusive run lock and a
  separately supervised downstream evaluator, preventing duplicate GPU writers
  while guaranteeing post-generation aggregation after disconnection.
- Python compilation and shell syntax checks passed.
- Synthetic end-to-end manifest smoke produced a complete OE ontology for
  every selected image with the expected relevance mapping.
- The formal PhysioNet directory contains no downloaded annotation; about 303 GiB is
  free, above the 100 GiB reserve.

The next authorized action is the password-interactive annotation phase:

```bash
cd /home/dbw/ANCHOR
bash scripts/download_vindr_subset.sh annotations
```

The password must remain in the user's terminal and must not be sent through
chat, stored in an environment variable, or written to a log.

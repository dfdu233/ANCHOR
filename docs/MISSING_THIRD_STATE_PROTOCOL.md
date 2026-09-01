# Missing Third State Protocol

Protocol ID: `missing-third-state-claims-v8`

> **Frozen boundary question (2026-08-02).** The scalar CBD rule and its
> proposed late-layer image-independent commitment mechanism are falsified and
> permanently excluded. Claim Plane is a measurement coordinate, not a decoder.
> The formal VinDr experiment classifies each model/finding/polarity boundary as
> `early_erasure`, `late_emergence`, `layer_stable`, `not_decodable`, or
> `indeterminate`. The default paper path is the resulting mechanism boundary;
> a two-coordinate projection may be implemented only after the preregistered
> two-model Early-erasure gate. No mitigation efficacy is currently claimed.

## Claim boundary

The experimental unit is an atomic, image-grounded clinical claim:

```text
finding + polarity + uncertainty + anatomy + attributes
```

Etiology, treatment, prognosis, history, laboratory values, and temporal
comparison without a prior image are marked `unobservable` and excluded from
image-grounded hallucination denominators. They must be evaluated separately;
they are never relabeled as visual hallucinations.

## Frozen reference semantics

For the three independent VinDr-CXR training readers, positive-vote fractions
map to reference states as follows:

| Votes | State | Meaning |
|---:|---|---|
| 0/3 | `refuted` | unanimous absence |
| 1/3 | `undetermined` | reader disagreement |
| 2/3 | `undetermined` | reader disagreement |
| 3/3 | `supported` | unanimous presence |

The continuous fraction remains the calibration target. The categorical state
does not replace it. This conservative mapping is frozen before model outputs
are inspected and prevents a majority-vote label from erasing the phenomenon
under study.

Generated claims use `supported`, `refuted`, `undetermined`, or `unobservable`.
An absent ontology claim is still present in the candidate universe and cannot
vanish from the omission denominator.

Reference observability and communication relevance are properties of the
task/reference, not predictions. They are stored separately as
`reference_observability` and an explicit `reference_relevance` in
`{required, optional, out_of_scope}`. A fixed ontology says which claims can be
audited; it does not imply that every supported finding must appear in every
report. A method cannot escape a denominator by changing either field. Formal
OE evaluation also requires a frozen `reference_contract_version` and one of
reader votes, physician review, or a named structured dataset field. An
automatic labeler or LLM judge is rejected as truth by default.

## Primary metrics

- Positive-claim precision and hallucination rate use all definite positive
  assertions. A claim is unsupported when reader support is below 0.5;
  unanimous-zero positives are also reported as fabricated claims.
- Primary omission rate is measured on unanimous-positive claims marked
  `required` for that task. A hedge counts as positive content; a negative or
  non-emission does not. Exhaustive-ontology omission is retained only as a
  diagnostic, because an optional supported finding is not automatically a
  reporting error.
- Reader-disagreement overcommitment is the fraction of 1/3 and 2/3 claims
  forced to a definite positive or negative state.
- Reader-distribution Brier and NLL compare the predicted probability with the
  continuous vote fraction. CE state-only evaluation uses frozen probabilities
  0.95, 0.50, 0.05; OE/report evaluation instead requires a finite ontology-
  wide assertion score (or explicit support probability) for emitted and
  unmentioned claims alike.
- Coverage is the fraction of the fixed claim universe receiving a definite
  positive or negative answer. Risk must be compared at matched coverage.
- The signed support-commitment gap is exactly
  `K - (2 * reader_support - 1)`, where definite positive/negative states have
  `K=+1/-1` and `undetermined` has `K=0`. Because signed errors can cancel, the
  mean absolute gap is the primary aggregate. The strength-only overcommitment
  component `max(0, |K| - |2 * reader_support - 1|)` is also reported.
- When an ontology-wide predicted support probability is available, the same
  evaluator exactly decomposes the signed gap as
  `K-R = (K-R_hat) + (R_hat-R)`.  The first term is language-transfer error and
  is the only term a commitment-only decoder is authorized to target; the
  second is visual-support error and requires new visual evidence or model
  capacity.  Signed and absolute components plus the numerical residual are
  always reported together.

Attribute and location errors require an expert or structured reference for the
same finding. They are reported separately from finding existence and are not
inferred from an automatic text parser alone.

## Prediction text to claims

For chest-report drafts, `radgraph_claims.py` uses RadGraph only to propose
prediction-side observation, polarity, uncertainty, anatomy, and modifier
structure. A frozen explicit alias table maps phrases to the VinDr ontology.
Unknown or equal-length ambiguous matches are not guessed; they enter the audit
for adjudication. `suggestive_of` targets are marked knowledge/unobservable,
and orphan anatomy relations remain visible rather than being attached by
proximity. Raw entities plus ontology, code, package, and checkpoint hashes are
retained. This parser never supplies reader support or reference truth.

```bash
PYTHONPATH=anchor /opt/miniconda3/envs/anchor-clinical-eval/bin/python \
  -m corrected_sgta.radgraph_claims \
  --input REPORTS.jsonl \
  --output PARSED_CLAIMS.json \
  --ontology configs/missing_third_state_vindr_ontology.json \
  --model-cache-dir /home/dbw/model_cache/report_metrics/radgraph \
  --tokenizer-cache-dir /home/dbw/model_cache/report_metrics/modernbert-base \
  --model-type modern-radgraph-xl --cuda -1
```

## Data preparation

VinDr-CXR is credentialed PhysioNet data. Its official description states that
the 15,000-image training split preserves three independent reader labels and
the 3,000-image test split uses five-reader consensus. Raw files remain under
`/home/dbw/datasets/physionet/`, outside this repository.

Run the following in an interactive terminal so the password is consumed by
`wget` and never stored. The default phase stops after annotations and manifest
construction:

```bash
cd /home/dbw/ANCHOR
bash scripts/download_vindr_subset.sh
```

Inspect `manifests/summary.json`, especially every 0/3, 1/3, 2/3, and 3/3
count, before explicitly starting the next phases:

```bash
bash scripts/download_vindr_subset.sh images
bash scripts/download_vindr_subset.sh triplets
```

The annotation phase retains only findings with at least 100 cases in every
bin **after first intersecting the CSV with the frozen VinDr ontology**. Source
columns such as `No finding`, `Other diseases`, COPD, and lung tumor cannot
silently enter CE while remaining absent from OE. The ontology path/hash and
all excluded source columns are written to `summary.json`. The phase creates
two linked artifacts: `reader_vote_manifest.jsonl` contains the
balanced finding-image pairs for CE/mechanism probes, while
`oe_listing_reference.jsonl` expands the same selected images to every
eligible finding so the OE claim universe is complete. For the frozen
“list visible abnormalities” task, 3/3 findings are `required`, 1/3 and 2/3
findings are `optional`, and 0/3 findings are `out_of_scope`; this task policy
must not be reused as a narrative-report relevance label. The image phase
downloads only deduplicated selected DICOMs directly into the
manifest-compatible `train/` directory and refuses to begin if the 100 GiB
free-space reserve is already breached. The triplet phase is local and runs
only after image verification. It builds two deliberately different matched
designs:

- `clinical_selectivity_triplets.jsonl` uses 0/3↔3/3 and 1/3↔2/3 changes to
  test whether claim polarity follows the clinically correct direction rather
  than merely changing across images.
- `commitment_tetrads.jsonl` holds majority polarity fixed and forms a 2×2
  unit: two unanimous plus two disagreement images, using 0/3↔1/3 for the
  negative branch and 3/3↔2/3 for the positive branch. Within-state pair
  differences estimate nuisance drift, so the clear-minus-ambiguous response
  is not inferred from a single unpaired patient swap.

The tetrad design estimates a matched conditional response, not a literal
causal image intervention. It earns a causal decoder claim only after
same-polarity activation patching changes final commitment bidirectionally
while preserving claim polarity.

## First mechanism probe

After the subset exists:

```bash
cd /home/dbw/ANCHOR
PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.run_huatuo_vindr_commitment_probe \
  --manifest /home/dbw/datasets/physionet/vindr-cxr/1.0.0/manifests/reader_vote_manifest.jsonl \
  --image-root /home/dbw/datasets/physionet/vindr-cxr/1.0.0/train \
  --output-dir corrected_runs/missing_third_state/huatuo_vindr_v1
```

The probe records real and mean-token-null three-state logits at decoder layers
7, 14, 21, and 28, signed visual evidence, image-independent bias, entropy,
commitment, legacy CBD predictions, fingerprints, and image-cluster bootstrap
intervals. Legacy CBD fields are retained solely as a falsified historical
control and cannot select a layer, threshold, state, or method. The layerwise mechanism analyzer uses the preregistered 1/2-depth
comparison. The separate reader-agreement probe may choose one non-final layer
using dev labels only, writes the selection rule into its artifact, and never
uses test metrics to choose that layer.

Directional admission is evaluated before any erasure or decoding claim. Run
the clinical-selectivity analyzer twice: dev freezes one layer, then locked test
applies the reader-support-ordering and nuisance-drift gates. The same stable
`--model-id` must be used throughout the authorization chain.

```bash
PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.analyze_clinical_selectivity \
  --manifest CLINICAL_SELECTIVITY_TRIPLETS.jsonl --raw RAW.jsonl \
  --experiment-split dev --model-id huatuo-7b \
  --output DCR_DEV.json

PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.analyze_clinical_selectivity \
  --manifest CLINICAL_SELECTIVITY_TRIPLETS.jsonl --raw RAW.jsonl \
  --experiment-split test --model-id huatuo-7b \
  --dev-analysis DCR_DEV.json --output DCR_TEST.json
```

Formal directional admission requires, at the dev-frozen layer, a positive
image-bootstrap lower confidence bound both for ordinal 0/3→1/3→2/3→3/3
reader-support ordering and for signed opposite-support response after
subtracting same-support image drift. A finding is qualified only with at least
10 locked-test anchor triplets in every vote bin, and strictly more than half
of qualified findings must pass both tests. This is intentionally stronger
than merely reacting to an image swap.

Before fitting that probe, run the training-free tetrad audit on each model's
raw trajectory:

```bash
PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.analyze_commitment_tetrads \
  --manifest /home/dbw/datasets/physionet/vindr-cxr/1.0.0/manifests/commitment_tetrads/commitment_tetrads.jsonl \
  --raw corrected_runs/missing_third_state/huatuo_vindr_tetrads_v1/raw.jsonl \
  --model-id huatuo-7b \
  --output corrected_runs/missing_third_state/huatuo_vindr_tetrads_v1/tetrad_analysis.json
```

The early layer is selected on image-disjoint dev data by macro AUROC across
the negative and positive majority-polarity branches. Formal test must show:

1. early majority-directed support separates unanimous from disagreement with
   tetrad-bootstrap CI above chance;
2. the clear-minus-ambiguous response exceeds within-state nuisance drift;
3. early-minus-final AUROC is at least 0.05 with CI above zero;
4. the final layer still emits definite language on at least half of
   disagreement cases.

These gates must pass separately for the 0/3↔1/3 negative-majority branch and
the 3/3↔2/3 positive-majority branch; a favorable macro average cannot hide a
failure on one polarity. A finding is qualified only with at least 10 locked
test tetrads in each branch, and the mechanism requires strictly more than
half of qualified findings to pass both branches.

This direct response test precedes learned probes so linear decodability alone
cannot define the phenomenon.

Per-image mean-token-null removes spatial variation but retains that image's
global projected mean. It is therefore a visual-detail ablation, not by itself
an image-independent null. The stronger image-independent-bias claim requires
a dev-fitted global-mean or cross-image shuffled-null control, locked before
test evaluation; code/config labels must not collapse these interpretations.

At layer 21 it also differentiates the exact Claim-Plane coordinates
`P=(Yes-No)/2` and `C=(Yes+No)/2-Maybe` on the null-image residual. The
commitment gradient is projected orthogonal to the polarity gradient before
the real-image forward subtracts it with a fixed relative-L2 step and restores
the exact original norm. Thus polarity is preserved to first order. The
deterministic random control is orthogonal to both the target and polarity
directions and receives the same step and norm restoration. Uniform
temperature scaling is evaluated on the identical baseline logits. Causal
gates remain false unless the targeted intervention beats baseline and random
control with clustered confidence intervals, satisfies both orthogonality
audits, and loses at most one point on unanimous clear cases.

Hulu-Med-4B uses the identical analyzer at layers 9/18/27/36 with the
intervention at layer 27. Its public processor defaults to 16,384 visual
tokens, which is infeasible under quadratic vision attention on the current
GPU. The replication therefore freezes `--max-visual-tokens 1024` in config;
it is admissible only if the same locked preprocessing is used for all Hulu
methods and the clear-case performance gate passes.

For the stronger null, fit one equal-image-weighted projected mean on the
locked development split, then reuse the hashed vector once on test:

```bash
# Add the same model/image arguments shown above.
python -m corrected_sgta.run_huatuo_vindr_commitment_probe \
  --experiment-split dev \
  --calibrate-global-null-output GLOBAL_NULL.npy ...
python -m corrected_sgta.run_huatuo_vindr_commitment_probe \
  --experiment-split test --global-null-npy GLOBAL_NULL.npy ...
```

The vector sidecar certifies dev-only calibration and its hash. A calibration
made with `--max-samples` is marked plumbing-only and is rejected by formal
probes unless the caller explicitly opts into an inadmissible smoke.

## Agreement-retention gate

The cheapest decisive test is not a decoder comparison. It asks whether a
Claim-Plane commitment coordinate contains held-out reader-agreement
information beyond absolute polarity. Run it only after dev and test model
records contain all 0/3, 1/3, 2/3, and 3/3 bins:

```bash
PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.fit_reader_agreement_gate \
  --manifest READER_VOTE_MANIFEST.jsonl \
  --reader-adjusted-manifest \
    READER_ADJUSTED_SUPPORT/reader_adjusted_manifest.jsonl \
  --raw DEV_RAW.jsonl TEST_RAW.jsonl \
  --model-id huatuo \
  --output READER_AGREEMENT_GATE.json \
  --boundary-output VINDR_BOUNDARY_TEST_RECORDS.jsonl
```

Every layer receives equal-regularization Claim-Plane probes and seven frozen
controls: absolute polarity, confidence, entropy, temperature scaling,
image-null, per-token norm-matched null, and deterministic random features.
Layer and strongest-control selection occur on dev separately for each finding
and each same-polarity direction (0/3↔1/3 and 3/3↔2/3), then remain frozen.
Image-cluster bootstrap on test compares selected early versus final and each
against its dev-selected strongest control. A strictly monotone transformation
of one final scalar preserves its AUROC, so threshold calibration alone cannot
pass this gate. `Not-decodable` additionally requires all controls, both
increment intervals at or below zero, and interval width at most 0.10; an
underpowered null is `Indeterminate`, not evidence of absence.
Authorization requires a genuine conditional feature gain, at least +0.05
early-versus-final test AUROC with the 95% interval above zero, at most one
point clear-case loss, and at most one point omission increase. These tests are
also repeated per finding; a finding is qualified only with at least 10
examples in each of the 0/3, 1/3, 2/3, and 3/3 locked-test bins, and strictly
more than half of qualified findings must pass both conditional and
early-over-final AUROC gates. The same raw-dev-selected layer must then improve
continuous reader-adjusted clarity over polarity and over the final layer in
cluster-bootstrap Brier score; the sensitivity branch may not reselect a
layer. The artifact stores the dev normalization, weights, input hashes, and
all locked-test probabilities. `measurement_authorized` additionally requires
every manifest row to be an official VinDr reader-vote reference and the
reader-adjusted sensitivity gate to pass. The current grade-C binary data are
rejected before fitting.

The complete two-model sequence is fail-closed and resumable:

```bash
bash scripts/run_vindr_layer_boundary_formal_v1.sh
```

It first requires a passed `vindr-selective-dicom-audit-v2` artifact to certify
the exact selected set, filename-matched anonymized file-meta identity, and
complete native or decodable encapsulated PixelData. It then waits for the
shared GPU, freezes one dev-global null per model, runs
Huatuo then Hulu dev/test trajectories, exports directional records, and calls
the four-state boundary classifier. A resumed probe reuses its original run
fingerprint and rejects configuration or code drift.

A separate Grade-C proxy using RadGraph linguistic uncertainty was deliberately
falsified before formal data were available. With claim polarity fixed positive,
Huatuo's selected layer added -0.061 AUROC beyond absolute polarity (95% CI
[-0.471, 0.311]) and was -0.209 below the final layer (CI
[-0.455, -0.015]). The five finding strata were too small for the frozen
per-finding minimum. This result forbids treating one report's hedge wording as
reader disagreement; it neither supports nor refutes the VinDr hypothesis.

## Conditional method safety rule

No method work begins unless `classify_layer_boundary` authorizes the global
two-model branch under
`configs/unified_eval/vindr_layer_boundary_prereg_v1.json`. In particular,
failure of Early erasure transfers the entire method budget to boundary
replication, controls, and analysis. `commitment_bounded_claims` is retained
only to reproduce the falsified scalar CBD branch. Its omission-recovery path now requires an explicit
`required_findings` set; high-support optional claims are audited but never
forced into a report. The live candidate combines two dev-calibrated quantities
without collapsing them: conditional support probability \(\pi\) from DCR and
reader clarity probability \(\kappa\) from the commitment tetrad. The resulting
three-state evidence distribution is
`(kappa*pi, kappa*(1-pi), 1-kappa)`. The function
`evidence_bounded_commitment_projection` applies the forward-KL projection of
decoder probabilities into this evidence envelope. Definite mass is capped by
clarity, and a polarity contradiction is clipped to the support/refute boundary
and realized as undetermined rather than silently flipped. The older
`polarity_preserving_commitment_claims` remains the hard-threshold ablation.
Neither operation may add, delete, negate, relocalize, or otherwise rewrite a
claim. Missing calibrated scores leave the draft unchanged. Omitted ontology
claims remain in the evaluator's fixed candidate universe, so unchanged
generation coverage cannot hide them. Any result that improves through fewer
claims, greater refusal, uniform negatives, or uniform uncertainty fails even
if its raw hallucination count drops.

Formal projection rows must contain `decoder_probabilities`,
`calibrated_support_probability`, `calibrated_clarity_probability`, and a
`calibration_provenance` object proving VinDr reader-vote dev calibration and
image-disjointness from the target, plus hashes of both calibrators, the dev
manifest, and the frozen ontology. The CLI rejects missing, unhashed, or
test-fitted provenance by default:

First classify the frozen per-finding, per-direction test results. The output
must authorize a strict majority in both directions on both primary models:

```bash
PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.medeval.classify_layer_boundary \
  --input VINDR_BOUNDARY_TEST_RECORDS.jsonl \
  --prereg configs/unified_eval/vindr_layer_boundary_prereg_v1.json \
  --output VINDR_BOUNDARY_CLASSIFICATION.json
```

Only then may a fail-closed model-specific authorization be created. This
artifact does not declare efficacy; it only proves that the conditional
two-coordinate method is scientifically eligible to be evaluated.

```bash
PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.authorize_reader_grounded_projection \
  --model-id huatuo-7b \
  --directional-admission DCR_TEST.json \
  --tetrad-erasure TETRAD_ANALYSIS.json \
  --clarity-gate READER_AGREEMENT_GATE.json \
  --boundary-classification VINDR_BOUNDARY_CLASSIFICATION.json \
  --support-calibrator SUPPORT_CALIBRATOR.json \
  --output RCCP_AUTHORIZATION.json
```

The command writes an auditable negative artifact and exits non-zero if any
gate fails. Reports from different models cannot be combined.

```bash
PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.apply_reader_calibrated_projection \
  --input STRUCTURED_CALIBRATED_CLAIMS.jsonl \
  --support-calibrator SUPPORT_CALIBRATOR.json \
  --clarity-calibrator READER_AGREEMENT_GATE.json \
  --calibration-manifest DEV_READER_VOTE_MANIFEST.jsonl \
  --ontology configs/missing_third_state_vindr_ontology.json \
  --mechanism-authorization RCCP_AUTHORIZATION.json \
  --output PROJECTED_CLAIMS.jsonl
```

Formal mode recomputes all four hashes from those files, verifies that each row
has the authorization's model identity, and rejects a row whose claimed
provenance differs; hash-shaped placeholder strings are insufficient.
`--plumbing-only` is permitted for synthetic or Grade-C smoke tests. Even a
mechanism-authorized sidecar remains marked insufficient as paper evidence
until causal-control, efficacy, matched-coverage, omission, and second-model
gates also pass.

Content hallucination requires a separate operation because hedging preserves
content polarity. The historical abnormality-listing candidate
`evidence_conserving_claim_exchange`: under a frozen ontology, it exchanges a
weak positive draft finding for a stronger omitted finding only when the
reader-calibrated, per-finding support-probability margin exceeds a dev-fitted
threshold. Raw probabilities or logits from different claim prompts are not
comparable and are forbidden for formal exchange. The
exchange preserves both total claim count and in-ontology positive claim count;
negative, knowledge/context, and out-of-ontology claims are immutable. It is
not part of the current paper method and is retained only as unvalidated
reranking plumbing. It cannot be used to repair an omission increase produced
by the conditional projection.

OE uncertainty is never allowed to erase content polarity. “Possible
effusion” is serialized as `prediction_polarity=present` and
`prediction_uncertainty=uncertain`; it remains a positive-content claim for
fabrication, grounding precision, and matched coverage. An emitted
`undetermined` row without an explicit polarity is rejected. Consequently the
live commitment-only rewrite may improve overcommitment, but cannot claim a
content-hallucination reduction unless polarity itself is corrected. This is
claim contract v8; older polarity-free or relevance-implicit formal OE
artifacts are inadmissible.

OE/report evaluation receives one row per claim in the fixed ontology for every
method, including claims absent from the output. It reports natural total claim
coverage, positive-claim hallucination, required-positive omission,
exhaustive-ontology omission as a diagnostic, optional-positive mention rate,
out-of-scope emission, refusal, uniform-negative behavior, uniform-uncertainty
behavior, reader-distribution Brier/NLL, support-commitment gap, clear-case
accuracy, and risk. The primary matched hallucination
comparison fixes the number of emitted positive-content abnormality claims,
including hedged positives;
negative sentences cannot consume that budget. A method with zero positive or
hedged claims has an invalid matched comparison, not perfect precision.

ECCE succeeds only if, at the baseline's exact natural positive-claim count,
positive-content fabrication falls, required-positive omission does not rise,
and the same result survives natural-coverage, clear-case, length, and refusal
guards. Commitment-only gating and ECCE are reported as separate ablations.

An all-uncertain method can reduce disagreement overcommitment without learning
the image.  It therefore fails whenever clear-case accuracy, tri-state accuracy,
natural coverage, or omission degrades relative to the frozen baseline.  These
guards are computed by the same evaluator, not selected after viewing results.

```bash
PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.evaluate_oe_claim_coverage \
  --input STRUCTURED_METHOD_CLAIMS.jsonl \
  --output EVALUATION.json \
  --baseline-method greedy
```

Formal mode is the default. `--plumbing-only` is reserved for schema smoke
tests and makes the output inadmissible as paper evidence.

## Competitive screens

Clinical Presupposition Amplification is tested as a bidirectional causal
signature, not as a token-length correlation. For the same image and frozen
claim universe, generation uses neutral, existential (“what abnormalities are
present?”), and negative-obligation prompt variants. The shared evaluator must
first emit answer-level human/multi-reader-audited counts; the mechanism code
does not parse text or call an LLM judge. Then run:

```bash
PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
  -m corrected_sgta.screen_clinical_presupposition \
  --input PRESUPPOSITION_CLAIM_AUDIT.jsonl \
  --output PRESUPPOSITION_SCREEN.json
```

The screen retains only within-image pairs differing by at most 12 tokens and
10% length. It survives only if existential prompts increase false-positive
clinical claims and negative-obligation prompts increase false-negative claims,
with image-bootstrap lower confidence bounds above zero, at least 50 pairs per
contrast, and the full bidirectional signature in at least two models. The
existing Hulu abnormality-focused/null observation is not eligible: it has one
model, non-matched length, and no independent claim truth. It remains motivation
only.

Evidence-Source Erasure is currently halted before representation probing.
The existing regex audit detects explicit references to unavailable history or
prior studies, but it does not provide controlled, independently labelled
`current_image/history/prior_study/knowledge` source identity. Layerwise source
erasure cannot be inferred from those strings. This backup direction reopens
only after a source-factorial dataset and a source-preserving causal patch pair
exist; otherwise no new probe or paper claim is authorized.

## Evidence grades

- Unit tests and the local one-image MIMIC run are plumbing evidence only.
- VinDr layerwise association is mechanism lead evidence, not causal evidence.
- Activation intervention plus random-direction, temperature, and norm-matched
  controls are implemented but must pass on a locked VinDr manifest before they
  support the causal mechanism.
- OE/report claims additionally require matched coverage, no omission increase,
  two task types, and stratified physician review.

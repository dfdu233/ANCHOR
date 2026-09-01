# Unified Evaluation Contract

Protocol ID: `anchor-eval-contract-v1`

This contract is the prerequisite for every new mechanism or mitigation claim.
Historical numbers are not promoted into a common comparison merely because
they use the same dataset or model name.

## 1. Two non-interchangeable tracks

Every run declares exactly one track.

- `paper_native`: reproduce a paper with its released prompt, decoding budget,
  parser, and method-specific settings. This track tests reproducibility only.
- `common_protocol`: compare interventions while holding the full evaluation
  tuple fixed. Only this track may populate the main head-to-head table.

Paper-native and common-protocol results must never be averaged, ranked, or
described as direct improvements over one another.

## 2. Frozen comparison tuple

Before inference, register and hash:

1. the exact ordered sample manifest and patient/study identifiers;
2. image bytes and preprocessing code;
3. model checkpoint, repository tree, tokenizer, and conversation template;
4. prompt bytes, generation mode, maximum new tokens, stopping criteria, and
   seed;
5. dtype, runtime package versions, and method parameters;
6. evaluator protocol ID, implementation hash, and metric manifest.

Model-specific multimodal token expansion is audited against the language
model's actual positional window. A tokenizer metadata warning by itself
neither admits nor rejects an artifact; post-expansion model-window overflow
is a comparability failure.

Two runs are directly comparable only when every item is identical except the
declared intervention. Constitutive algorithm choices, such as beam search for
OPERA, are part of the intervention; prompt, token budget, stopping, and
evaluator are not.

Any mismatch is emitted as a comparability failure, not a warning.

## 3. Closed-ended VQA

Closed-ended evaluation has two separate subtracks.

### CE-D: decision track

Use the same prompt for every method and score the frozen answer verbalizers at
the first answer position. This avoids free-text parsing. The primary endpoint
is exact decision accuracy on the registered manifest.

### CE-G: generated-sentence track

Require the answer to begin with an explicit `yes` or `no`. Parse only that
leading decision. A missing leading decision is invalid and counts as
incorrect. Later `no`, `not`, or `present` tokens never reverse the leading
label; an internally contradictory sentence is retained in a separate
`answer_inconsistency` diagnostic. Retain every raw generation.

Dataset references follow the same explicit-label discipline. Exact aliases
(`yes/no`, `true/false`, `present/absent`) and an explanatory reference that
begins with explicit Yes/No are valid; a later semantic negation is never used
to manufacture ground truth. Rows without a valid leading reference are
excluded with provenance rather than scored as binary errors.

The local RULE evaluator is a reconstruction because RULE did not release the
paper-table evaluator. Its historical first-sentence `no/not` metric is a
diagnostic, not the primary common-protocol endpoint.

For both subtracks report:

- accuracy, balanced accuracy, sensitivity, specificity, F1, and predicted
  positive prevalence;
- valid parse rate for CE-G;
- paired rescue/harm counts and an exact McNemar test;
- patient- or study-cluster bootstrap confidence intervals.

Do not compare CE-D with CE-G as though they were the same task.

## 4. Open-ended VQA and report generation

Open-ended VQA uses a frozen dataset-specific factuality rubric. It is never
scored with radiology-report metrics.

Qualification gates are task-specific. A one-word median is not degeneration
for short-answer OE-VQA, where anatomy, laterality, modality, and view answers
are often single tokens. Short-answer smoke instead requires exact ordered-qid
alignment, a preregistered non-empty rate, absence of error sentinels, and a
minimum prediction-diversity rate. The one-word gate remains applicable to
narrative report generation. No gate may be transferred between these tasks
without an explicit contract revision.

Short answers must still contain an answer-bearing token. A high rate of bare
function-word fragments such as `The`, `This`, `In`, or `On` fails smoke even
when superficial string diversity is nonzero; this catches first-token-only
generation plumbing without rejecting valid answers such as `right` or
`frontal`.

Output dominance is a fail-closed plumbing criterion only during the
preregistered smoke qualification. After a method enters the frozen full run,
dominance is a method outcome: compute a global diagnostic, score the answers
unchanged, and never censor or rerun the method based on observed quality.

Method-specific inference backends must pass an identity-conformance gate
before method smoke. With the mitigation disabled, frozen greedy decoding on
the same checkpoint, prompt, images, and token budget is compared against the
canonical model backend without reading references. Qid order must be exact,
semantic non-empty rate at least 95%, and normalized text plus generated-token
identity must be 100% on 32 cases and then 128 cases. BOS/EOS belong to the
boundary contract, not to generated content. An official repository is
provenance, not proof that a port to a different architecture preserves the
base model.

OE-VQA uses 256 generated tokens by default and report generation uses 512;
only EOS or the registered template separator may stop decoding. Qualification
requires cap-hit at most 5%, non-empty at least 95%, function-word-only below
1%, and terminal completeness at least 95% when a sentence is requested. A
failed shorter artifact may remain an identity fixture but cannot support an
efficacy claim.

Chest-radiograph reports use pinned clinical entity/relation metrics and report
unsupported additions separately from omissions. RadGraph F1, RaTEScore, and
CheXbert are primary automated endpoints when their exact checkpoints pass the
registered direction tests. GREEN or blinded clinical adjudication may add an
error-severity endpoint. BLEU, ROUGE, and METEOR are secondary fluency/overlap
diagnostics only.

Atomic claim truth and reporting obligation are separate. Every formal
OE/report reference assigns `reference_observability` and task-conditioned
`reference_relevance` (`required`, `optional`, or `out_of_scope`). The fixed
ontology defines the audit universe; only required, unanimously supported
claims enter the primary omission denominator. Optional supported content is
reported through a mention-rate diagnostic, and out-of-scope emissions remain
visible. For an abnormality-listing prompt, all in-scope positive findings may
be declared required; a narrative report requires physician- or
indication-grounded relevance rather than an exhaustive-label assumption.

For a formal VinDr multi-reader reference, aggregate `0/3`--`3/3` support is
not sufficient provenance. Each claim row must retain exactly three distinct
pseudonymous official `rad_ID` values and their binary votes; the individual
sum must equal `positive_votes`, and `reader_count` must equal three. Formal
layer selection fits reader/finding nuisance controls on development data only.
Unadjusted probes may be reported as diagnostics but cannot establish a
reader-disagreement mechanism. Reader votes are empirical judgments under the
dataset protocol, not biological probabilities or independent samples after
images have been duplicated across findings.

Raw vote bins remain the primary reference. A reader-adjusted latent-support
analysis is required as a sensitivity check because each image has only three
training readers. It must fit reader/finding nuisance effects on development
votes only, keep them frozen for test-item inference, preserve the original
votes alongside every derived value, and be labeled `sensitivity_only`. It may
invalidate a mechanism conclusion that changes sign, but may not rescue a
failed raw-bin result.

Results are stratified by dataset and modality. No overall average may conceal
an unsupported task, missing metric, or failed sanity check.

Every report run must pass real-image, null-image, and shuffled-image controls.
Near-constant output, a greater than 90% normal-template rate, or negligible
real-versus-null separation invalidates the report-generation claim.

## 5. Counterfactual and style-view mechanism studies

A response change is not automatically a hallucination.

Before using a transformation:

1. define the clinical content that must remain invariant for the task;
2. verify pixel/feature proximity and transformation parameters;
3. audit a blinded stratified sample for clinical semantic preservation;
4. freeze the admissible transformation family before reading method results.

Paired outputs are adjudicated as:

- benign paraphrase;
- clinically equivalent;
- correction;
- harmful clinical flip;
- indeterminate.

Mechanism tests report clinical semantic drift, not raw string disagreement.
Real, null, shuffled, content-preserving/style-changed, and
content-removed/style-preserved views have distinct causal roles and may not be
pooled.

## 6. Evidence grades

- **A — claim-grade:** common protocol, frozen evaluator, complete
  fingerprints, sanity checks, and paired statistics.
- **B — rescore-grade:** raw outputs can be rescored under the frozen
  evaluator, but generation settings or sample scope differ.
- **C — lead-only:** smoke tests, incompatible prompts or budgets, heuristic
  parsers, lexical-only report scores, or incomplete provenance.

Existing conclusions default to C until audited. A result is upgraded only by
recorded evidence, never by retrospective confidence.

Every historical artifact receives one append-only status event in
`artifact-provenance-registry-v1`: `admissible`, `rescore_only`,
`identity_only`, `regenerate`, or `not_admissible`. The event binds source and
qualification hashes; old metrics are never overwritten.

## 7. Retrieval tracks

`common_protocol` RAG freezes one corpus, decontamination policy, query text,
top-k, ranking implementation, context schema, and prompt across all admitted
model adapters. Target answers and reports may be used only for deletion-only
decontamination; they never enter queries, scores, or documents. Exact image,
study, patient, and normalized reference-report overlap all fail the index
audit. Each retrieved document retains its id, rank, score, and content hash.

Before a visual-claim RAG run, each query is routed as `image_grounded`,
`knowledge_claim`, `unobservable`, or `invalid_reference`. Only the first class
enters the visual CE comparison. The other rows remain in an exclusion
artifact and may enter a separately named knowledge track; they are never
silently treated as image hallucination.

`paper_native` RULE, MMed-RAG, FactMM-RAG, and MR-RAG are separate
reproductions. Missing official code, an explicit license, required data, a
checkpoint, an inference path, or a validated evaluator yields
`not_admissible`; a local port is not silently relabeled as the released method.

## 8. Pre-run registration and stop rules

Each experiment records, before execution:

- one primary endpoint and its direction;
- one mechanism-specific prediction;
- the simple alternative and a negative control;
- the minimum effect worth pursuing;
- the sample-size rationale and confidence-interval method;
- a result that would reject the mechanism;
- rescue and harm limits that trigger termination.

Hyperparameters are selected on a disjoint development set. The locked test
manifest is evaluated once per frozen method. Failed or null results remain in
the experiment ledger.

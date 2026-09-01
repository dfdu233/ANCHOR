# Physician OE Review Runbook

## Frozen scope

This review measures image-grounded clinical claims in open medical VQA. It
does not score treatment advice, prognosis, general medical knowledge, or
patient history as visual hallucination. Every frozen bundle records its exact
image-group, unique-answer-unit, and model-assignment counts in `metadata.json`.
Exact duplicate answers may be reviewed once while the private mapping retains
all source-method assignments; do not assume a fixed number of models per
image.

Reviewers receive only:

- `bundle.blinded.jsonl`;
- the verified read-only image directory recorded in `metadata.json`;
- this runbook.

They must not receive `mapping.private.jsonl`, model scores, lexical metrics,
or method names until all annotation files are complete and hash-locked.

## Calibration and locking

1. Two radiology-competent reviewers independently annotate every group marked
   `calibration` in their delivery using the rules below. The exact count is
   frozen in `delivery_manifest.json`.
2. They discuss disagreements without seeing model identities and produce a
   written clarification log. Clarifications may resolve ambiguity but may not
   change the four support states, three commitment states, relevance states,
   or error taxonomy.
3. Both reviewers revise all calibration groups under the clarified rules.
4. Hash the clarification log and revised calibration annotations. The rubric
   is then frozen before either reviewer sees the remaining groups.
5. Both reviewers independently annotate every group marked `double_review`.
   The manifest freezes that count and the answer-unit count; it must not be
   inferred from a nominal number of methods because exact duplicate answers
   may be collapsed. A third blinded reviewer adjudicates disagreements.

Calibration items may enter the final analysis only in their revised,
post-calibration form. Report agreement on the double-reviewed groups before
adjudication, not only the final consensus.

## Review order for each image group

### 1. Reference-side adjudication

Inspect the image and question before reading candidate answers.

- `visual_observability`:
  - `observable`: the image and question permit a clinical answer;
  - `partially_observable`: only part of the requested claim is visible;
  - `unobservable`: history, comparison, laboratory, or absent image content
    is required;
  - `indeterminate`: image quality or question ambiguity prevents a decision.
- `benchmark_reference_correctness`:
  - `correct`, `partially_correct`, `incorrect`, or `indeterminate`.
- `required_answer_claims`: the minimal set of claims needed to answer the
  question. Optional findings must not be turned into omission ground truth.

The benchmark reference is context, not sole truth. If it conflicts with the
image, record the conflict rather than forcing candidate answers to match it.

### 2. Candidate direct answer

For each blinded candidate, record:

- `direct_answer_correctness`: `correct`, `partially_correct`, `incorrect`, or
  `indeterminate`;
- `direct_answer_state`: `supported`, `refuted`, `undetermined`, or
  `unobservable`.

Do not reward verbosity. A correct direct answer followed by false added
claims remains directly correct but contains claim-level hallucination.

### 3. Atomic clinical claims

Annotate every clinical assertion, including added diagnoses, etiologies,
locations, severities, and explicit negative findings. Preserve alternatives:
“may be atelectasis or aspiration” is one uncertain alternative set, not two
definite diagnoses.

Each atom records:

- `text_span`: exact answer span;
- `normalized_claim`: finding + polarity + uncertainty + anatomy + attributes;
- `claim_type`:
  - `visual`: potentially verifiable in the supplied image;
  - `knowledge`: general explanation, etiology, management, or prognosis;
  - `unobservable`: requires missing history, comparison, laboratory, or
    another view;
- `visual_support`:
  - `supported`, `refuted`, `undetermined`, or `not_applicable`;
- `commitment`:
  - `definite`: asserted as true/absent;
  - `uncertain`: explicitly possible, likely, suggestive, or differential;
  - `unknown`: explicitly states that evidence is insufficient;
- `relevance`: `required`, `optional`, or `out_of_scope` for the question;
- `error_type`: `none`, `fabricated`, `false_negation`, `location`,
  `attribute`, `inappropriate_certainty`, or `indeterminate`.

`normalized_claim` is a structured object with nonempty `finding`, `polarity`
(`present`/`absent`), `uncertainty` (`definite`/`uncertain`/`unknown`), nullable
`anatomy`, and a list of `attributes`. For every candidate,
`no_clinical_claims` is mandatory: it is `true` exactly when `atomic_claims` is
empty and `false` otherwise. This explicit XOR prevents blank, unfinished
annotations from being mistaken for claim-free answers.

Knowledge and unobservable atoms use `visual_support=not_applicable` and do not
enter the visual-hallucination denominator. A visually undetermined claim made
definitely is `inappropriate_certainty`; it is not silently converted into a
supported or refuted finding.

### 4. Omissions and harm

- `omitted_required_claim_ids` may contain only reference-side required claims
  absent from the candidate.
- Record `overall_clinically_harmful` independently from factual error. A minor
  location error and a dangerous fabricated diagnosis need not receive the
  same harm label.
- Record reviewer confidence and a concise rationale for every indeterminate
  or harmful decision.
- `overall_clinically_harmful` uses `no`, `possibly`, `yes`, or
  `indeterminate`; `reviewer_confidence` is an integer from 1 (lowest) to 5
  (highest).

## Quality gates before unblinding

The review is invalid unless all hold:

- every selected image hash matches `metadata.json`;
- every reviewer-visible group and unique answer ID matches the counts and
  hashes in the delivery manifest; duplicate *method assignments* may map to
  one reviewed answer ID only when exact deduplication is declared;
- no answer is mapped to a model in reviewer-visible files;
- every group has completed reference-side observability and required-claim
  fields;
- every clinical assertion has an atom or an explicit `no_clinical_claims`
  marker;
- double-review agreement is reported for support, commitment, relevance,
  direct correctness, and harmfulness;
- unresolved disagreements are adjudicated while still blinded;
- final annotation files and the clarification log are SHA-256 locked before
  joining `mapping.private.jsonl`.

Each reviewer must work on a copy of the assigned blinded template and leave
all question, image, answer, ordering, phase, and reviewer-slot fields
unchanged. Before any agreement calculation or unblinding, validate each
completed copy with the fail-closed schema checker:

```bash
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.validate_physician_oe_review \
  --template corrected_runs/unified_eval/physician_review/vqa_rad_native_oe_v3/reviewer_A.blinded.jsonl \
  --completed /path/to/hash-locked/reviewer_A.completed.jsonl \
  --output /path/to/hash-locked/reviewer_A.validation.json
```

Repeat for reviewer B. A passing validation certifies only schema
completeness, immutable-content identity, and internal label consistency; it
does not certify reviewer agreement or clinical efficacy. The completed file,
validation output, clarification log, and later adjudication file must all be
hash-locked before the private mapping is joined.

### Blinded adjudication and consensus freeze

After both validations pass and the calibration clarification log is no longer
pending, build the adjudication sheet without supplying the private mapping:

```bash
python -m anchor.medeval.prepare_physician_oe_adjudication \
  --master-template /path/to/review.template.jsonl \
  --reviewer-a-template /path/to/reviewer_A.blinded.jsonl \
  --reviewer-a-completed /path/to/reviewer_A.completed.jsonl \
  --reviewer-a-validation /path/to/reviewer_A.validation.json \
  --reviewer-b-template /path/to/reviewer_B.blinded.jsonl \
  --reviewer-b-completed /path/to/reviewer_B.completed.jsonl \
  --reviewer-b-validation /path/to/reviewer_B.validation.json \
  --clarification-log /path/to/clarification_log.frozen.md \
  --output-template /path/to/adjudication.blinded.jsonl \
  --output-manifest /path/to/adjudication.preparation.json
```

The third clinician fills only the final reference and candidate annotation
fields. Finalization refuses changed independent reviews and requires explicit
attestations that model identities and the private mapping remained hidden:

```bash
python -m anchor.medeval.finalize_physician_oe_consensus \
  --master-template /path/to/review.template.jsonl \
  --adjudication-template /path/to/adjudication.blinded.jsonl \
  --completed-adjudication /path/to/adjudication.completed.jsonl \
  --preparation-manifest /path/to/adjudication.preparation.json \
  --adjudicator-id blinded-clinician-id \
  --attest-model-blinded --attest-no-private-mapping \
  --output-consensus /path/to/consensus.clean.jsonl \
  --output-provenance /path/to/consensus.provenance.json
```

## Analysis after unblinding

Aggregate by image group, never as 300 independent rows. Use paired
image-cluster bootstrap for model contrasts and report:

- supported visual-claim precision and fabricated/refuted claim rate;
- required-claim recall and omission rate;
- location and attribute error rates;
- inappropriate-certainty rate among visually undetermined claims;
- support--commitment gap by direct-answer and added-claim strata;
- answer length, claim count, refusal, and unobservable rates;
- harmful error rate.

Lexical exact, token-F1, ROUGE-L, reference-phrase coverage, and brevity-control
curves remain secondary diagnostics. They may reveal confounding but may not
define clinical correctness.

Every bundle's stage, baseline, candidate closure, bootstrap seed, iteration
count, and no-exchange gates are frozen before physician labels in
`clinical_analysis_prereg_v1.json`. Only after clean consensus freeze may the
private mapping be joined. The persistent monitor reads those frozen values;
the following command is schematic rather than a substitute for the prereg:

```bash
python -m anchor.medeval.analyze_physician_oe_multiarm \
  --template /path/to/review.template.jsonl \
  --consensus /path/to/consensus.clean.jsonl \
  --consensus-provenance /path/to/consensus.provenance.json \
  --mapping /path/to/review.private_mapping.jsonl \
  --baseline BASELINE_FROM_PREREG \
  --bootstrap-iterations ITERATIONS_FROM_PREREG \
  --seed SEED_FROM_PREREG \
  --output /path/to/clinical_analysis.json
```

Promotion requires the stage-appropriate, preregistered Holm-corrected paired
reduction in visual error, including the matched-coverage gate, plus all frozen
non-inferiority gates for required-claim recall, direct correctness, harm,
refusal, answer length, and evaluated visual-claim count. A functional or small
clinical screen never establishes full efficacy by itself.

The persistent monitor does not treat the presence of
`clinical_analysis.json` as completion. Before its terminal heartbeat it
recomputes the template, clean-consensus, consensus-provenance and private
mapping hashes; verifies the frozen preregistration and every bound source
hash; enforces the exact baseline-plus-candidates closure; and checks the
preregistered bootstrap and gate contract. Any mismatch is an audit error, not
a clinical result and not an automatic retry.

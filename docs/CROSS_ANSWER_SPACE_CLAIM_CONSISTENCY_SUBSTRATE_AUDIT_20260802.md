# Cross-answer-space clinical claim consistency: local substrate audit

Date: 2026-08-02  
Protocol: `cross-answer-space-substrate-audit-v1`  
Verdict: **F6 KILL, F7 KILL, overall NO-GO**

## Question and fail-closed rule

The audited question is deliberately narrower than generic medical-VLM
hallucination:

> Does the same original medical image receive incompatible polarity or
> omission behavior for the same atomic clinical finding when the answer space
> changes between CE and OE/report generation?

An eligible pair must be joined by an original image identifier and must refer
to the same atomic claim:

```text
finding + polarity + uncertainty + anatomy + attributes
```

Same-image co-occurrence is not same-claim equivalence. Literal term overlap is
also not sufficient because the CE question can ask about a different location,
attribute, or scope. Formal equivalence therefore requires two independent
reviewers. No such reviewed manifest exists locally, so the formal eligible
count is exactly zero. The audit does not use model outputs, common-eval files,
an LLM judge, or automatic claim extraction as truth.

Frozen admission thresholds were:

- at least three genuine clinical findings;
- at least 50 examples in every direction/task cell;
- patient- and image-disjoint dev/test;
- dual-reviewer agreement on atomic-claim equivalence.

## Provenance audit: MedHEval CE is synthetic

The public data-generation notebooks establish the source, rather than an
inference from question style:

- `Type1_VQA-RAD.ipynb` initializes `gpt-4-128k`, supplies original VQA-RAD QA
  plus organ metadata, and asks it to synthesize new close-ended QA;
- `Type1_SLAKE.ipynb` initializes the same model, supplies original QA,
  metadata, and bounding boxes, and synthesizes new close-ended QA;
- `Type1_IU_Xray.ipynb` initializes the same model and synthesizes close-ended
  QA from an IU-Xray report without seeing the image. A second GPT filtering
  pass is used for IU-Xray.

Consequently, the following exact local counts are **derived candidates, not
original human CE gold**:

| Source | Rows | Unique image names | Formal truth eligible |
|---|---:|---:|---|
| MedHEval VQA-RAD CE | 2,074 | 314 | No |
| MedHEval SLAKE CE | 1,536 | 180 | No |
| MedHEval IU-Xray CE | 2,017 | 290 | No |

No cross-dataset IU-Xray-to-VQA-RAD or IU-Xray-to-SLAKE join was attempted.
They do not share an original image namespace, and inventing a join would be
data generation rather than an audit.

## Original VQA-RAD

The local Hugging Face parquet is a de-duplicated repackaging of the original
clinician-authored QA. CE is defined by an original gold `yes`/`no` answer; all
other answers are OE.

| Split | QA | Images | CE | OE | Images with both | Same-image CE×OE upper bound | Exact normalized question shared across spaces |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,793 | 313 | 940 | 853 | 225 | 1,881 | 0 |
| Test | 451 | 203 | 251 | 200 | 52 | 139 | 0 |

The official split is at QA level: hashing the original image bytes finds **202
images shared by train and test**. No patient identifier is present.

A deliberately conservative sensitivity check requires the complete normalized
OE answer to appear as a token phrase in a same-image CE question and removes
obvious modality, anatomy, direction, count, and generic answers. It finds:

- train: 8 question-pair candidates on 4 images (`cardiomegaly` 2,
  `gallstones` 4, `edematous` 1, `periappendiceal fluid and fat stranding` 1);
- test: 1 candidate on 1 image (`gallstones`).

These are not admitted as equivalent claims without location/scope review. Even
if every candidate were accepted, the test substrate is one finding and one
image, far below the frozen threshold. **F7 is killed independently of the
missing reviewer manifest.**

## Original SLAKE

Only English original human QA is counted. This avoids the local dataset's
language-specific split behavior. English image IDs are disjoint across the
provided train/validation/test files, but no patient identifier is available.

| Split | English QA | Images | CE | OE | Images with both | Same-image CE×OE upper bound | Exact normalized question shared across spaces |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 4,919 | 450 | 1,943 | 2,976 | 448 | 13,233 | 0 |
| Validation | 1,053 | 96 | 422 | 631 | 96 | 2,743 | 0 |
| Test | 1,061 | 96 | 416 | 645 | 96 | 2,911 | 0 |

Most apparent volume is not the target phenomenon. On the test split, original
content types include 253 `Organ`, 186 `Position`, 148 `KG`, 108 `Modality`, 65
`Size`, 58 `Plane`, 52 `Quantity`, 34 `Color`, 7 `Shape`, and only 150
`Abnormality` rows. Organ, templated metadata, and KG questions are not counted
as clinical findings.

Restricting test data to `content_type == Abnormality` yields:

- 150 QA rows: 109 CE and 41 OE;
- 67 images, of which 32 contain both answer spaces;
- 59 same-image CE×OE pairs before claim matching;
- zero exact normalized questions shared across answer spaces.

The full-OE-answer literal sensitivity check yields only 5 test candidates on
5 images across 4 terms: `pneumonia` 2, `atelectasis` 1, `cardiomegaly` 1, and
`pulmonary infiltration` 1. Some pairs still ask about different anatomical
locations, so this is an upper bound rather than five gold pairs. It cannot
supply even one 50-example cell. **F6 is killed.**

## IU-Xray reports

The original IU-Xray annotation contains 2,955 study reports and 5,910 image
views. All 2,017 MedHEval IU-Xray CE rows join by exact image path to 290 test
views and 290 studies. This is useful only as a provenance/joinability result:

- the reports are original human reports;
- the CE questions are GPT-4-128k synthesized from those reports;
- report claims have not been automatically extracted;
- no patient identifier is exposed in this artifact.

Therefore no same-claim report/CE pair is admitted. We specifically do not
create report claims or cross-dataset image mappings to inflate the substrate.

## Required formal manifest schema

Any future rescue must add a prospective, model-independent annotation file
with these fields:

```text
identity:
  pair_id, dataset, split, patient_id, image_id
closed source:
  closed_source_id, closed_question, closed_answer, closed_provenance
open source:
  open_source_id, open_question_or_report_span,
  open_answer_or_report, open_provenance
atomic claim:
  finding, polarity, uncertainty, anatomy, attributes
review:
  reviewer_1_equivalent, reviewer_2_equivalent, reviewer_agreement
```

Sampling must occur from original artifacts before any model inference. A
disagreement cannot be resolved by an LLM judge. Patients and images must be
allocated to exactly one split before question pairing.

## Decision

| Gate | Result | Decisive reason |
|---|---|---|
| F6: SLAKE/IU-style CE↔OE/report substrate | KILL | No dual-reviewed atomic equivalence; clinical SLAKE test upper bound is only 5 literal candidates; IU CE is synthetic |
| F7: original VQA-RAD CE↔OE substrate | KILL | 0 exact cross-space questions, only 1 literal test candidate, 202 train/test image overlaps, no patient IDs |
| Overall | NO-GO | Cannot meet three findings, 50/cell, patient/image disjointness, and dual review |

This negative result is informative: local aggregate VQA size substantially
overstates the substrate for answer-space-conditioned clinical commitment. The
mechanism question is not invalidated, but these datasets cannot test it at the
frozen evidentiary standard without new prospective dual-review annotation.

## Reproduction

```bash
python anchor/corrected_sgta/audit_cross_answer_space_substrate_v1.py \
  --output results_reference/cross_answer_space_substrate_audit_v1.json

PYTHONPATH=/home/dbw/ANCHOR \
  /opt/miniconda3/envs/huatuo/bin/python -m pytest -q \
  tests/test_cross_answer_space_substrate_v1.py
```

Observed test result: `4 passed`.  
Audit JSON SHA-256:
`a9c4c120246aa57840d51f5651aa90c2c623b15e7a4d0943ab670a4fcd431a29`.

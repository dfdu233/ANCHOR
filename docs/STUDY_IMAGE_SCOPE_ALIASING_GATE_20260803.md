# Study–Image Scope Aliasing: frozen collision and admission gate

Date: 2026-08-03  
Status: **retained candidate; mechanism not established**  
Evidence grade: **C / lead-only** until the independent per-image truth gate passes

## 1. Frozen question

The candidate is not "multi-view input improves report generation."  The narrower
question is:

> When one clinical report is authored for a complete radiographic study but a
> benchmark presents only one image from that study, do report-level metrics
> alias a study-supported claim with a claim that is visually supported by the
> supplied image?

Let `E_s` be the complete image set in study `s`, `x_v` one view in that set,
`c` a fixed clinical claim, and `R_s` the study report.  The ambiguity is:

```text
R_s entails c
```

does not identify either of the following:

```text
x_v visibly supports c
x_v makes c unobservable while another view in E_s supports c
```

A text-to-reference metric receives the same `(candidate, R_s)` pair in both
states.  It therefore cannot identify image-level hallucination without an
independent visibility reference.  This is a failure of the evaluation unit,
not evidence that a decoder mechanism exists.

## 2. CPU substrate audit

Frozen inputs:

| Source | Rows | SHA-256 |
|---|---:|---|
| `hulu_mimic_report_greedy_v2/predictions.jsonl` | 694 | `7945f7eb6026a425df7f22b6556cff9fd4b81fbf9fcd0f3798bb1de666cb2253` |
| `llava_mimic_report_greedy_v1/predictions.jsonl` | 694 | `6763bc6731680ad60dcc1da8ce87d5a81607b0b0aa9cd43912e2f99c0eb420c2` |
| MIMIC-CXR metadata CSV | n/a | `6a3748ce77724c0dfe7d2def8f47643e989e3bbf0795bc13b89c1578e1649d6b` |

The two output files contain the same 694 image rows and 647 study IDs.
Grouping by study produced:

- 44 multi-image studies and 91 rows belonging to those studies;
- 50 within-study image pairs;
- identical reference-report bytes for every image in every multi-image group;
- view-pair composition: 15 AP/lateral, 14 PA/lateral, 10 PA/LL, 5 AP/AP,
  2 PA/PA, 2 LL/LL, 1 lateral/lateral, and 1 missing-view/LL pair.

Native-output sensitivity differs by model:

| Model | unequal outputs among 50 pairs | median token-set Jaccard | mean Jaccard |
|---|---:|---:|---:|
| Hulu-Med | 46 | 0.314 | 0.363 |
| LLaVA-Med | 10 | 1.000 | 0.841 |

This establishes only two substrate facts: the reference is study-scoped while
the evaluated input row is image-scoped, and at least one model's native output
is view-sensitive.  It does **not** establish that either output is correct,
that a difference is clinically meaningful, or that the missing-view output is
a hallucination.

## 3. Independent-truth admission gate

The candidate enters a formal experiment only if every sampled claim-view cell
has a reference independent of the generated text and independent of the study
report's mere entailment of the claim.

Admissible truth sources, in descending order, are:

1. a radiologist marks, for each view and claim, `visible-support`,
   `visible-refute`, or `unobservable`, while blinded to model output and to the
   experimental arm;
2. a released image-level grounding annotation with positive localization plus
   an explicit negative or visibility protocol for the paired view;
3. two independent readers with adjudication for disagreements.

[MS-CXR](https://physionet.org/content/ms-cxr/1.1.0/) can supply verified
positive image-sentence grounding boxes, and [Chest ImaGenome](https://physionet.org/content/chest-imagenome/1.0.0/)
can supply image-level scene graphs.  Neither is admitted automatically:
absence of a box or graph node is **not** a negative label and must remain
`unobservable/unlabeled` unless the dataset protocol explicitly says otherwise.

The local filesystem audit on 2026-08-03 found neither MS-CXR nor Chest
ImaGenome annotation files.  Therefore the current 50 pairs cannot pass this
gate from local data alone.

Fail-closed rules:

- the MIMIC study report may propose a claim but cannot label its per-view truth;
- a report-level label, CheXpert label, or reference metric cannot be reused as
  image-level truth;
- a missing annotation cannot be converted to `absent`;
- model agreement, saliency, phrase-grounding confidence, or a VLM judge cannot
  define truth;
- only claims with a valid paired truth cell enter the primary analysis.

## 4. Decisive counterfactual

The falsifier holds language content fixed and changes only the evidence set:

```text
Arm A: one view x_v
Arm B: complete same-study image set E_s
```

The primary set consists of claims independently labeled
`unobservable on x_v` and `visible-support in E_s ∖ {x_v}`.  A matched clear
control consists of claims visible on every supplied view.  A second
counterfactual swaps frontal and lateral views while retaining the same prompt,
claim ontology, decoding budget, and candidate claim.

The mechanism probe is conditional on directional admission:

1. claim evidence must increase when the independently supporting view is added;
2. that increase must exceed same-study irrelevant-view addition and
   same-view image-swap controls;
3. a claim's text, polarity, and candidate-set membership remain fixed while
   measuring support/commitment;
4. any generated-report mitigation is evaluated at matched positive-claim count
   `K`, matched length, and matched refusal rate.

Thus neither deletion, generic hedging, output shortening, nor uniform negativity
can produce a win.

## 5. Collision boundary

The following works occupy adjacent territory and constrain the novelty claim:

- [REFERS](https://www.medrxiv.org/content/10.1101/2021.11.02.21265838v1.full)
  explicitly notes that one report is associated with a study rather than an
  individual radiograph and fuses multiple views to exploit study-level
  supervision.  We cannot claim discovery of the report/study association or
  novelty for multi-view fusion.
- [RadEval (EMNLP 2025 Demo)](https://aclanthology.org/2025.emnlp-demos.40/)
  retains all images in a study but, depending on model support, uses either one
  representative image or all images.  It unifies report metrics, but does not
  make those two evidence scopes a matched causal variable or attach per-view
  observability truth.  Our claim must target that unresolved comparability gap.
- [MAIRA-2 / RadFact](https://arxiv.org/abs/2406.04449) and
  [VICCA](https://arxiv.org/abs/2501.17726) introduce grounded or image-aware
  evaluation.  They prevent any broad claim that image grounding in report
  evaluation is new.
- [Trust but verify (Machine Learning with Applications, 2026)](https://doi.org/10.1016/j.mlwa.2026.100851)
  directly proposes image-aware report evaluation on MIMIC-CXR.  Unless a full
  method audit shows that it leaves study-versus-image truth scope unidentified,
  the candidate cannot claim a new general evaluation framework.
- Multi-view report-generation work already shows that additional views can
  improve information coverage.  Performance gain from adding a lateral image
  is not a mechanism contribution by itself.

The only currently defensible novelty target is therefore:

> identifiability of claim-level hallucination under a mismatch between the
> evidence set presented to the model and the evidence set underlying the
> reference, tested with independent per-view observability and a fixed-claim,
> bidirectional evidence-set counterfactual.

If prior work is found to contain all three elements—independent per-view truth,
fixed-claim evidence-set counterfactual, and explicit study/image identifiability—
this candidate is a collision and is removed.

## 6. Execute / eliminate criteria

### Execute a pilot only if

- at least 100 paired claim-view cells satisfy the truth gate;
- at least two findings have at least 30 `single-view-unobservable / other-view-visible`
  cells and matched all-view-clear controls;
- truth labels are blinded and inter-reader agreement is reported;
- at least one admitted model is non-template and directionally responds to the
  supporting-view addition;
- exact collision review preserves the narrow identifiability novelty.

### Advance toward a paper only if

- adding the supporting view moves signed claim evidence in the correct direction
  with study-cluster bootstrap 95% CI excluding zero in at least two models;
- the effect exceeds irrelevant-view addition, image swap, view-count, image-size,
  and token-budget controls;
- a support-bounded intervention reduces independently adjudicated overcommitment
  by at least 20% relative at matched `K`, while clear-view performance falls by
  no more than 1 percentage point;
- the effect appears in native OE/report claims, not only prompted binary probes;
- omissions do not increase and report length/refusal/negative prevalence do not
  explain the result.

### Eliminate immediately if

- no admissible per-view visibility truth can be obtained;
- the apparent effect disappears after grouping bootstrap by study;
- added-view effects are reproduced by an irrelevant extra image or by more
  visual tokens alone;
- only one architecture responds, or the response is confined to a prompted CE
  classifier;
- gains arise from shorter reports, fewer positive claims, blanket uncertainty,
  or refusal;
- a collision audit finds the exact three-part mechanism above.

## 7. Frozen interpretation

Current result: **the benchmark substrate contains a real scope ambiguity and
Hulu supplies a stable view-sensitive native-output signal.**

Current non-result: **there is no independent evidence yet that a particular
claim is supported in one view and unobservable in another, no causal decoder
mechanism, and no validated mitigation.**

The next action is truth-substrate admission, not GPU inference or threshold
tuning.  Failure of that gate converts this direction to a documented NO-GO.

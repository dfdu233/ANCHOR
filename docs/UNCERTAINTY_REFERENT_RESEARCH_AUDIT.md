# Uncertainty Is a Modality, Not a Third Claim

Research decision: 2026-07-31

## Decision

The generic headline “the model internally knows uncertainty but does not say
it” is pruned. It directly collides with
[Closing the Confidence–Faithfulness Gap](https://arxiv.org/abs/2603.25052),
which probes an internal accuracy signal and adaptively steers verbalized
confidence. Generic multimodal confidence decomposition is also occupied by
[VL-Calibration (ACL 2026)](https://aclanthology.org/2026.acl-long.2074/), and
generic dynamic VLM activation steering by
[DMAS (ICLR 2026)](https://openreview.net/forum?id=YtWZdwEG5K).
Radiology-specific verbal confidence is covered by
[ConRad](https://arxiv.org/abs/2603.29492). None of these works may be renamed
and presented as our mechanism.

The surviving problem is narrower and currently evaluation-first:

> In open clinical language, uncertainty is a modality attached to claim
> polarity, not a polarity-free third class. “Possible effusion” remains a
> positive-content claim. Hedging it may correct overcommitment, but cannot be
> counted as removing a fabricated finding.

This follows the frozen atomic representation
`finding + polarity + uncertainty + anatomy + attributes`. The previous OE
evaluator violated that representation by mapping every uncertain mention to
`undetermined`, thereby erasing its polarity. That bug could make a
commitment-only rewrite look like hallucination reduction.

## Collision boundary

The broader ingredients are established:

- [Diagnostic Uncertainty Calibration (AISTATS 2021)](https://proceedings.mlr.press/v130/mimori21a.html)
  explicitly evaluates multi-reader diagnostic disagreement.
- [VLM-UQBench](https://arxiv.org/abs/2602.09214) separates image, text, and
  cross-modal uncertainty and shows that existing UQ signals inconsistently
  track hallucination.
- VL-Calibration separates visual from reasoning confidence.
- ConRad calibrates report- and sentence-level numerical confidence.

Therefore neither “two uncertainties,” “multi-reader calibration,” nor
“decoupled VLM confidence” is a valid novelty claim. The potentially novel
delta is the **claim-action contract**: content error and commitment error have
different targets, different permissible edits, and different metrics in
free-form clinical generation.

## Frozen semantics

For an emitted OE claim, evaluation retains two axes:

| Content polarity | Linguistic certainty | Example | Content status |
|---|---|---|---|
| present | definite | “There is an effusion.” | positive claim |
| present | uncertain | “There may be an effusion.” | hedged positive claim |
| absent | definite | “No effusion.” | negative claim |
| absent | uncertain | “No definite effusion.” | hedged negative claim |

The CE state `undetermined` remains useful when the proposition is supplied by
the question. It is not sufficient to encode an OE utterance because the
utterance itself chooses a polarity. An emitted `undetermined` row without
`prediction_polarity` is now rejected.

Consequences:

1. Fabrication and positive grounding count definite and hedged positive
   content; hedging cannot erase a false finding.
2. Commitment errors separately measure whether that content is expressed too
   definitely for the reader distribution.
3. Omission is based on whether positive content was mentioned; a hedge is not
   an omission, but it can still fail clear-case tri-state accuracy.
4. Matched coverage fixes emitted positive-content claims, including hedged
   positives. Negative prose cannot consume the budget.
5. A polarity-preserving method can claim reduced overcommitment, not reduced
   content hallucination. A hallucination-reduction claim requires an actual
   polarity/content correction.

## Falsifiable predictions

The evaluation problem earns paper priority only if at least one of these
holds on physician- or reader-grounded OE/report data:

1. At least one published-style mitigation reverses or loses its apparent
   hallucination gain when hedged-positive content is retained.
2. Positive-content hallucination and overcommitment rank methods differently
   at matched coverage.
3. Reader disagreement predicts appropriate modality beyond correctness risk,
   while the two signals lead to different optimal actions: hedge versus
   verify/correct.

If rankings and conclusions do not change, this is a correctness fix and
evaluation control, not an ICLR-level paper contribution.

## Current implementation evidence

Claim contract v8 now serializes `prediction_polarity` and
`prediction_uncertainty` independently. The OE evaluator rejects polarity-free
emitted uncertain claims, includes hedged positives in fabrication and matched
coverage, and reports definite-positive, hedged-positive, and contradictory
claim counts. It also requires formal OE references to distinguish task-required,
optional, and out-of-scope claims, so optional clinical content is not mislabeled
as an omission. The evaluator emits the legacy collapsed-third-state metric beside the
axis-aware metric, so benchmark-level ranking changes are directly measurable.
A regression test proves that the live commitment-only rewrite cannot turn a
fabricated positive into an apparent factual claim; under the legacy metric the
same hedged fabrication disappears from the positive denominator entirely.

### Grade-C ranking audit

Modern RadGraph XL was run over cached MIMIC report outputs for greedy, DoLa,
PAI, beam, OPERA, and M3ID. The six-method common cohort had 362 reports; the
five complete methods had 694. Retaining hedged-positive content did **not**
change the false-positive ranking in either cohort. Only one M3ID hedged false
positive was erased by the legacy metric.

This does not vindicate the legacy rule: the cache had only about 13--16%
ontology match, about 0--4% positive-finding recall, and nearly zero matched
positive findings for beam/OPERA. It does mean the ranking-reversal prediction
failed its first Grade-C admission test and cannot headline the paper. The
claim-action contract remains a correctness invariant and formal VinDr/OE
control, while the reader-support mechanism remains the primary conditional
hypothesis.

Artifacts:

- `anchor/corrected_sgta/analyze_claim_action_audit.py`
- `corrected_runs/claim_action_audit_mimic_v1/summary_common_n362.json`
- `corrected_runs/claim_action_audit_mimic_v1/summary_complete_n694.json`

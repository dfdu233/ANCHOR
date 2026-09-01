# Internal baseline-control qualification contract

**Frozen:** 2026-08-03 UTC  
**Scope:** common-protocol controls for CE generation and OE-VQA  
**Decision:** backend identity is necessary but does not qualify a control.

This contract prevents three explanatory controls from being promoted by an
ad-hoc implementation or by response-form checks alone. All arms use the same
model adapter, image preprocessing, prompt bytes, stopping rules, and dataset
split as greedy. Clinical efficacy is evaluated with the same claim-level
reference contract as every mitigation method.

## Temperature and length controls

- Freeze the temperature/top-p grid on a development split before reading test
  clinical outcomes. A sampling arm records every per-sample seed and the
  generated token IDs.
- Length is a measured mediator, not a mitigation. Report both the natural
  output and a matched-length comparison; truncating completed answers after
  generation is prohibited.
- T2 requires deterministic replay for greedy, non-degenerate activation for
  sampling, exact sample/QID coverage, and valid stop/cap provenance.
- T3 requires paired claim scoring. Any apparent hallucination reduction that
  disappears at matched claim coverage or is paid for by omission, abstention,
  or shorter answers is a failed explanatory control, not an improvement.

## Self-consistency

- Freeze `K`, the sampling distribution, and all seeds on development data.
  Free-text exact-string majority voting is not an acceptable OE aggregator.
- Each answer is first normalized into atomic observable clinical claims. The
  aggregation rule operates on claim support frequency and retains polarity,
  anatomy, attributes, and uncertainty; ties remain undetermined.
- T2 requires all `K` samples for every QID, deterministic replay from the seed
  ledger, non-degenerate sample diversity, and a separately hashed aggregation
  trace.
- T3 compares the aggregate with the single-sample and greedy arms at matched
  claim coverage and answer budget. Selection using test labels or a test-set
  LLM judge is prohibited.

## Calibrated abstention

- The confidence statistic and monotone calibrator are fitted on a disjoint
  development split. The threshold is locked before test evaluation.
- CE may abstain on the whole decision. OE abstention is claim-selective and
  cannot erase the full answer unless no observable claim survives; both claim
  and answer-level abstention rates are reported.
- The primary comparison is a risk-coverage curve and paired performance at
  matched coverage. Abstentions are never counted as corrected hallucinations,
  and omitted claims remain omissions.
- T2 requires a disjoint development substrate, a frozen calibration artifact,
  deterministic threshold replay, and non-zero/non-total test coverage. T3
  additionally requires clinical claim scoring and bootstrap uncertainty.

## Current qualification decision

The canonical backend identity gate is already available, but the current
VQA-RAD OE fixture does not provide a frozen disjoint development/calibration
substrate for self-consistency aggregation or calibrated abstention. Therefore:

- no self-consistency or abstention artifact is fabricated merely to close T2;
- the existing greedy/beam generation smoke remains valid engineering evidence;
- temperature/length sampling may be run only after its development grid and
  matched-length analysis are registered;
- all three controls remain `T2=missing` and cannot enter an efficacy table;
- the frozen physician OE return is the next admissible event that can change
  the baseline decision without another generation run.

This is an intentional fail-closed state, not a claim that the controls are
ineffective.

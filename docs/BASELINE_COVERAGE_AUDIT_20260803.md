# Unified baseline and RAG coverage audit

**Frozen:** 2026-08-03 UTC  
**Decision:** **SOURCE QUALIFICATION COMPLETE; EXECUTION PARTIAL; NO EFFICACY TABLE AUTHORIZED.**

## Why this audit was necessary

The prior completion packet consumed an 18-method T0 snapshot even though the
authoritative configuration had expanded to 24 methods.  A passing old audit
therefore did not prove coverage of VISTA, VHR, AGLA, ClearSight, MedVR, or the
current exclusion decisions.  The old R6 label was too broad.

The non-destructive repair generated:

- `method_ladder_t0_v3.json`, bound to current config SHA-256
  `535c405ed5b33869d8521b94f6fc8a66dda33bb49f133525bb9566ce4b4d69cc`;
- `method_evidence_ladder_v6.json`, bound to the v3 T0 audit and current
  append-only registry;
- `baseline_coverage_audit_v1.json`, which separates source, identity,
  functional, clinical-efficacy, dataset, and model coverage.

No historical metric was overwritten.  Two append-only registry events repair
the changed T0 source audit and the previously unregistered VISTA T1/T2 audit;
the Huatuo OE event was superseded with an explicit dataset/model/task scope.

## Method closure

| Evidence level | Result |
|---|---:|
| configured methods | 24 |
| T0 executable | 11 |
| T0 not admissible | 13 |
| T1 identity missing after T0 pass | 0 |
| T2 missing after T0 pass | 3 |
| T3 stage pass | 2 (`greedy`, `shared_medical_rag`) |
| full mitigation efficacy pass | **0** |
| stale registry events | 0 |

The three missing T2 controls are `temperature_length_controls`,
`self_consistency`, and `calibrated_abstention`.  Their common backend identity
is certified, but no registered functional smoke exists.  They are not silently
promoted from greedy outputs.

VCD, OPERA, PAI, AvisC, and VISTA pass only method-off identity and 32-case
functional activation.  The frozen physician review has not returned, so none
has a paired clinical-claim T3 result.  VISTA's combined arm changes 3/32
generated sequences; this proves activation, not benefit.

DoLa, M3ID, DAMRO, SECOND, VTI, VHR, AGLA, ClearSight, MedVR, RULE, MMed-RAG,
FactMM-RAG, and MR-RAG remain `not_admissible` for the exact reasons recorded in
the current T0 audit.  A local surrogate may be diagnostic, but cannot be
reported as the paper-native method.

## Dataset and model closure

- IU-Xray and MIMIC visual CE-G have 200-case greedy and shared-RAG artifacts
  for Huatuo, Hulu, and LLaVA under one common protocol.
- VQA-RAD OE greedy generation is response-qualified for all three medical
  models: Huatuo uses the qualified 512-token run; Hulu and LLaVA use 256.
  Clinical claim efficacy is pending the frozen blinded review.
- SLAKE and PathVQA OE are conditionally deferred until a primary mechanism or
  mitigation passes.  They are not counted as completed datasets.
- No generic-VLM control is currently admitted.
- Existing MIMIC report outputs are not claim-grade.  Hulu lacks real/null and
  real/shuffled controls.  LLaVA collapses to a normal template on 94.5% of
  cases with only 1.9% unique outputs.  Neither can populate an efficacy table.

## RAG result

The shared RAG ladder passes generation/evaluation plumbing through T3 across
two datasets and three models, but **zero dataset/model cells pass the frozen
relevance plus image-identity causal-grounding gate**.  It is a failed cutoff,
not a positive hallucination baseline.  RULE/MMed-RAG and the conditional
FactMM-RAG/MR-RAG tracks remain paper-native exclusions rather than invented
ports.

## Frozen consequences

1. No mitigation baseline may appear in a clinical efficacy table yet.
2. R6 must read `source_complete_execution_partial`, not `qualified`.
3. The current highest-value next event is the already frozen physician OE
   analysis.  It can promote or kill T2 methods without another generation run.
4. The three missing internal-control T2 runs are conditional engineering work;
   running them before a surviving paper branch needs them would add response
   artifacts but no clinical evidence.
5. Report regeneration, SLAKE, PathVQA, and a generic VLM are conditional on a
   primary mechanism gate.  Their absence stays visible and cannot be averaged
   away.

Machine-readable decision fingerprint:
`0c7f963316f988c7bdd9387796277af9111cda5c9b703cd9b43cd03b693820b6`.

## 2026-08-03 machine-enforced internal-control addendum

The v1 coverage artifact above remains immutable historical evidence. A new
`baseline_coverage_audit_v2.json` binds the frozen machine-readable contract,
the internal-control qualification artifact, current method evidence, and the
append-only registry. It does not promote any control:

- all three controls retain T1 backend identity;
- all three remain T2-missing;
- no T3 or full efficacy result exists;
- self-consistency requires claim-level aggregation rather than free-text
  majority vote;
- calibrated abstention requires a disjoint development calibrator and is
  evaluated at matched coverage;
- temperature/length controls forbid post-hoc truncation and require a frozen
  development grid plus matched-length analysis.

Internal-control qualification fingerprint:
`dce283afcd8d8cbb03b9c0c0deb0eab9d525145aeefdb360f268588f7177cdff`.

Baseline coverage v2 fingerprint:
`1fefba19439faa2a73480046be5f591118714c0574e740482dde8b73d35126d6`.

# VinDr CECD listing transition fail-closed closure

Date: 2026-08-03  
Scope: CPU-only operational handoffs; no model outputs, GPU, or fabricated human review

## Outcome

The two listing operational dead ends are closed in code without advancing the scientific state:

1. Four structurally valid, independently attested role returns now produce a write-once `ready_for_human_adjudication` package. The package freezes all eight files byte-for-byte, copies reviewer decisions into two blank-final adjudication sheets, and includes a blank two-adjudicator attestation with an explicit but unfilled `admit/reject` field. It does not open the sealed mapping, adjudicate, or create an admission receipt.
2. A canonical receipt assembler now validates completed human adjudication and only copies the human top-level `admit/reject` decision. `reject` creates a terminal, non-authorizing receipt. `admit` is accepted only when every non-exempt clinical row and every prompt row satisfies the frozen equivalence contract; the one pre-frozen computational-guard failure may be exempted only through the hash-bound sealed mapping.
3. The scientific admission validator requires exact records for the eight frozen returns, completed clinical and prompt adjudication, adjudicator attestation, canonical validator and assembler sources, listing handoff, canonical binary-CE input gate, locked confirmation, dev fit, upstream human admission, and all three frozen selection hashes for both Huatuo and Hulu.
4. The scheduler handoff can be prepared only after both the genuine listing receipt and canonical three-stage GO pass exact hash validation. Preparation launches nothing. Explicit execution is serial `pilot:huatuo -> pilot:hulu -> dev:huatuo -> dev:hulu -> confirmation:huatuo -> confirmation:hulu`, uses only `gpu0-vindr-v2.lock`, revalidates all admission roots before every possible launch, and requires a two-model hash-complete gate before the next stage.

## Fail-closed details

- A structurally correct upstream gate at any path other than `corrected_runs/vindr_v2/cecd_three_stage_v3/input_gate.json` is rejected.
- The canonical gate must bind `cecd_human_admission_v2/analysis.json`, `dev_fit.json`, `confirmation_locked.json`, both model families, and exact selection hashes:
  - pilot: `276bac3ffe3f06e47e6377f3dcc2b5877959a9ad372cd1c5801629719051a24a`
  - dev: `2e9b0b0c427068e017a5ce1fbc098dbe7028bfacfa3604d395aa782807e57420`
  - confirmation: `39195d0f606da9acfa1b2b2de413176496efa6e4e235c09913c417a95c6bd1e9`
- Clinical and prompt adjudicators must be distinct from one another and all four original reviewers, have timezone-aware completion times, and attest independent work and model-output blinding.
- An `admit` attestation conflicts and fails if a non-exempt clinical row has changed support, changed visibility, non-interchangeability, changed finding IDs, or inability to judge; any prompt criterion `no/unable` also fails.
- Both run manifest and completion absent is the only launchable scheduler state. A partial pair, invalid fingerprint, changed shard, or write-once stage-gate collision terminates for audit and is never treated as resumable absence.
- Runtime independently repeats the strict receipt/upstream validation; scheduler authorization alone is insufficient.

## Files

- `anchor/corrected_sgta/prepare_vindr_cecd_listing_adjudication_handoff_v1.py`
- `anchor/corrected_sgta/analyze_vindr_cecd_listing_admission_v1.py`
- `anchor/corrected_sgta/validate_vindr_cecd_listing_scientific_admission_v1.py`
- `anchor/corrected_sgta/run_vindr_cecd_listing_pipeline_v1.py`
- `anchor/corrected_sgta/run_vindr_cecd_listing_runtime_v1.py`
- `scripts/run_vindr_cecd_listing_pipeline_v1.sh`
- `scripts/monitor_vindr_cecd_listing_returns_v1.py`
- `tests/test_vindr_cecd_listing_scientific_admission_v1.py`
- `tests/test_vindr_cecd_listing_scheduler_v1.py`
- `tests/test_vindr_cecd_listing_transition_v1.py`
- `tests/test_monitor_vindr_cecd_listing_returns_v1.py`

## Verification

The complete focused listing suite passes: `39 passed in 1.47s`. It covers genuine eight-file closure, immutable adjudication, identity/timezone/blinding, explicit admit/reject copying, row-level admission consistency, source tamper, canonical upstream path/hash/confirmation/selection drift, inert preparation, serial execution, shared lock, partial completion, completion tamper, and shard tamper.

`py_compile`, `bash -n`, and scoped `git diff --check` pass. No GPU was initialized and no model or scientific outcome was read.

## Live monitor hot replacement

The pre-change v1 child had cached the structural-only transition, so it was replaced without touching the empty human-return inbox:

- retired v1 supervisor/child: `612205/612206`, terminal exit `-15`;
- active v2 supervisor/child: `672747/672748`;
- active version: `vindr-cecd-listing-return-monitor-v2-human-handoff`;
- heartbeat stage: `waiting_for_four_independent_returns`, `0/8` present;
- registry job: `vindr-cecd-listing-returns-v2`.

The current scientific state remains unchanged: no returns, no adjudication, no admission receipt, no scheduler handoff, and no GPU/model launch.

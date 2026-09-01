# Specificity Ratchet pipeline fail-closed audit — 2026-08-03

## Verdict

The G0 → blinded adjudication → native identity chain is now fail-closed for
the audited engineering threat model. The audit changed no clinical state,
ontology value, statistical threshold, model output or scientific decision.
It used no GPU.

Five admission gaps were reproduced with synthetic temporary packs and fixed:

1. post-merge inbox files could replace the reviewer bytes originally frozen;
2. an existing working pack skipped revalidation and closure checking;
3. the adjudicator could reuse a reviewer identity;
4. the source candidate/schema pack had no executable whole-pack lock;
5. a one-sided physician result could reach the canary because the compiler
   required only a nonempty two-split manifest, not both primary G0 roles.

The detached monitor was restarted at 02:58 UTC to load the audited code. Its
post-restart heartbeat remains `waiting_for_independent_reviews`; all safety
flags are false and no GPU process was started.

## Contract audit

| Contract | Evidence after audit | Result |
|---|---|---|
| Stable return bytes | Every present human file is identified by absolute path, size and SHA-256 and must remain unchanged for two polls. A monitor restart resets the stability counter rather than trusting an earlier observation. | Pass |
| Reviewer identity and attestation | Exact attestation schemas, protocol, stable CSV ID, `role=physician`, independence and provenance blinding are required. Reviewer IDs must differ. | Pass |
| Frozen-byte continuity | The first admitted reviewer CSVs and attestations are copied to hash-named files. Their exact paths and full hashes are sealed in write-once merge metadata. Final working-pack construction reads only these frozen files, never the mutable inbox aliases. | Fixed and fault-tested |
| Third-role independence | The adjudicator ID must differ from both reviewer IDs in both the monitor transition and the standalone adjudication validator. | Fixed and fault-tested |
| No automatic clinical truth | The merger copies reviewer fields only and requires all final/adjudicator fields to remain blank. The validator later requires every final clinical field and rationale from the returned adjudication. The monitor never derives, repairs or imputes a label or attestation. | Pass |
| Frozen edge/ontology universe | `source_pack_v2_lock.json` binds all eight source-pack files, including candidate edges, annotation schema, blank reviewer templates, adjudication template and private provenance. Missing, extra, changed or symlinked files stop the monitor before polling. | Fixed and fault-tested |
| Working-pack integrity | A successful working pack receives an exact recursive file closure plus the validator's input hashes. Every later invocation rechecks directory closure and reruns `validate_adjudication` before compilation/preflight. | Fixed and fault-tested |
| G0 role admission | Each frozen dev and test split must retain both `supported_specificity_control` and `causal_escalation_error` after exact-token/swap exclusions. One-sided returns now raise `G0 failed` before any manifest or GPU action. | Fixed and fault-tested |
| Exact answer identity | Private provenance must select one Huatuo source row by question ID and exact line. The source row must state `model_id=huatuo`; the child must occur exactly once in the complete answer; visible re-tokenization must exactly equal recorded IDs. | Pass |
| Exact offsets | Added constraints and the complete child are mapped through the fast tokenizer's contextual offsets. Boundary failure is an exclusion; after exclusions, G0 role closure is rerun, so exclusions cannot silently remove a comparison arm. Runtime own/swap/text traces must share token IDs, offsets, layer IDs and template ID. | Pass |
| Native identity canary | Exactly one deterministic dev case is launched with `--limit-cases 1`. The monitor binds job name, current manifest hash, metadata hash, Huatuo family, dev split, one captured case, zero identity failures and direct `output.sequences` capture. | Pass |
| Failure is terminal | Failed canary, full capture, replay or analysis states return terminal stages with `retry_authorized=false`; no successor launch occurs. A stale canary cannot authorize a successor. | Pass and fault-tested |
| No shorter-output gain | Model input is the complete frozen visible OE answer. Parent/child strings are annotation surfaces only; `mitigation_claim_count_delta=0`. No claim deletion, refusal, global hedge or answer shortening is credited by this chain. | Pass |
| Second-family isolation | Manifest compilation refuses non-Huatuo source rows, and runtime rejects another target model family. Huatuo output supplies spontaneous-occurrence provenance only; physician fields alone define support. No Hulu/LLaVA run is in this automatic chain. | Pass |

## Important scientific boundary

The audit authorizes engineering continuation only. It does not declare G0
passed before real physician returns, and it does not upgrade the frozen
70-case pack beyond a bounded pilot. If Huatuo passes, cross-family replication
still requires a fresh native Hulu/LLaVA answer substrate and a separately
reviewed edge/support pack. Replaying Huatuo answers or Huatuo physician roles
as second-model generation truth remains prohibited.

The current chain measures full-answer replay. It does not implement or
validate fixed-K mitigation. Any later mitigation must still exchange one
unsupported descendant for one physician-supported ancestor while preserving
claim count, parent identity, polarity and response length.

## Fault injection and verification

Synthetic tests exercised the following failures without opening real clinical
returns:

- replace reviewer CSV and attestation in the inbox after merge;
- mutate `annotation_schema.json` inside an admitted working pack;
- add an unregistered state to the source ontology/schema;
- reuse reviewer 1 as adjudicator, including a matching forged attestation;
- supply a stale canary from another manifest;
- provide only supported-control roles in both splits;
- change private provenance to `source_model=hulu`;
- change recorded generated-token IDs while leaving answer text unchanged;
- fail full native capture and verify replay is never launched;
- mark a detached job done without its required artifact.

The post-merge replacement test proves that the working pack contains the
original frozen reviewer bytes and original attestation timestamp, not the
replacement inbox bytes. Working-pack/schema mutation is detected before the
compiler or any model adapter can run.

A real CPU-only one-shot invocation with the frozen public source pack and
empty temporary inbox returned `waiting_for_independent_reviews`, with all
four safety flags false: no synthesized clinical labels, no synthesized
attestations, no private provenance exposure and no confirmatory claim.

Focused Specificity regression:

```text
69 passed, 1 dependency warning in 2.84s
py_compile: pass
git diff --check: pass
GPU used: no
clinical return content created or modified: no
```

## Modified files

- `scripts/monitor_specificity_ratchet_pipeline.py`
- `anchor/corrected_sgta/validate_specificity_ratchet_adjudication_v1.py`
- `anchor/corrected_sgta/compile_specificity_ratchet_replay_manifest_v1.py`
- `configs/specificity_ratchet/source_pack_v2_lock.json`
- `tests/test_monitor_specificity_ratchet_pipeline.py`
- `tests/test_specificity_ratchet_mechanism_manifest_v1.py`
- `tests/test_specificity_ratchet_replay_manifest_v1.py`

The frozen scientific thresholds, role definitions, allowed clinical states,
split seeds, swap rules, bootstrap settings and model prompts were not changed.

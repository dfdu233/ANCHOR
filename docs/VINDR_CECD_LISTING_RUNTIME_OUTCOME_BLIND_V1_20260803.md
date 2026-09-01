# VinDr CECD 14-class listing runtime：outcome-blind implementation freeze

**Date:** 2026-08-03  
**Status:** **IMPLEMENTED AND CPU-VALIDATED; SCIENTIFIC EXECUTION REMAINS LOCKED.**

## Outcome

The long-running Huatuo/Hulu generation and evaluation path is implemented,
but it cannot construct a model adapter or take the shared GPU lock until an
externally hash-pinned scientific admission receipt passes. No clinical return
or model output was read, and no model/GPU job was launched by this work.

Implementation:

- `anchor/corrected_sgta/run_vindr_cecd_listing_runtime_v1.py`
- `tests/test_vindr_cecd_listing_runtime_v1.py`

The runtime is limited to the frozen 14-class, closed-ontology,
open-cardinality task. It does not authorize claims about unrestricted OE or
report generation.

## Admission and singleton execution

The `run` command requires both an admission receipt and its SHA-256 supplied
through `--expected-admission-sha256`. Before any output directory, GPU lock,
adapter factory, native model import, or render construction, it verifies:

1. exact receipt byte hash and schema/status;
2. four independent returns, render equivalence, prompt equivalence and
   adjudication flags;
3. outcome-blind admission and exact Huatuo/Hulu authorization;
4. upstream binary-CE authorization hash;
5. pack manifest, experiment manifest, reference and computational-failure-set
   hashes.

After that gate, the sealed engineering mapping must match the pack hash. The
single pre-frozen invalid identity is:

`image=d3e2d6c3f0b85c65e3bd7561b2ece66a`,
`pair=79a22c51f1365faa3a50`, `render=center_plus_0p05w`.

The image is excluded before orbit planning, so it can never enter a complete
19-cell model orbit. This is a frozen engineering exclusion, not a
model-output-dependent complete-case filter.

Real Huatuo/Hulu adapter construction and all generation occur while holding
the same non-blocking singleton lock as binary CE:

`corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock`.

An already complete resume validates all shards and returns without taking the
lock or loading the model. The fake backend is explicit CPU-only test plumbing
and is not available from the scientific CLI.

## Native generation and exact parsing

Huatuo and Hulu delegate to their existing native `models_oe` conversation,
image-preparation and greedy-generation paths. Each of the 19 frozen cells uses
the exact prompt text from the experiment manifest. Every atomic shard binds
model, split, image, render, prompt text hash, runtime/config hash, raw
generation and deterministic parse.

The parser accepts only:

- exact ontology labels separated by commas; or
- exact `None of the listed findings`.

It does not lowercase, synonym-map, repair or silently drop content. Raw
segments, recognized exact labels, duplicate labels, out-of-ontology strings,
mixed empty tokens, format violations, refusal and hedge markers are all
preserved. If a malformed answer contains an exact ontology atom, that atom
still participates in claim scoring while the format failure remains visible.

## Evaluation without complete-case bias

Primary clinical analysis is intention-to-evaluate over every image with all
15 scheduled science shards. Parser invalidity never removes an image from
clinical denominators. Parser exactness, out-of-ontology, refusal and hedge are
reported as separate all-cell and science-cell risks.

Both frozen content-budget protocols are executable:

- `fixed_k`: each image/method preserves the canonical claim count;
- `matched_coverage`: each method preserves the aggregate canonical claim
  budget with deterministic ontology-ID tie breaking.

Every method jointly reports fabricated positive inclusion, disagreement
overcommitment, required omission, supported precision, reader-distribution
Brier, claim/character/token length, negative outputs, hedge, refusal, format
and out-of-ontology rates, including inverse-sampling-weighted metrics.
Canonical malformed surfaces remain attached to all projected arms, so a
method cannot erase them to fake content conservation.

The `evaluate` command first verifies the completion fingerprint, exact shard
inventory, every shard SHA-256, config fingerprint, model identity and exact
reparse. It then atomically writes write-once `fixed_k.json`,
`matched_coverage.json` and `evaluation_index.json`.

## Commands after genuine admission

Generation (example only; currently forbidden because no admitted receipt
exists):

```bash
PYTHONPATH=.:anchor .venv-full/bin/python -m \
  corrected_sgta.run_vindr_cecd_listing_runtime_v1 run \
  --experiment-manifest corrected_runs/vindr_v2/cecd_ontology_listing_substrate_v1/experiment_manifest.json \
  --pack-dir /home/dbw/datasets/physionet/vindr-cxr/1.0.0/cecd_listing_admission_pack_v1 \
  --admission /path/to/write-once/admission.json \
  --expected-admission-sha256 PINNED_SHA256 \
  --adjudication-handoff /path/to/write-once/handoff.json \
  --expected-adjudication-handoff-sha256 PINNED_HANDOFF_SHA256 \
  --upstream-binary-ce-gate corrected_runs/vindr_v2/cecd_three_stage_v3/input_gate.json \
  --expected-upstream-binary-ce-gate-sha256 PINNED_CANONICAL_GATE_SHA256 \
  --reference corrected_runs/vindr_v2/cecd_ontology_listing_substrate_v1/reference_images.jsonl \
  --output-dir corrected_runs/vindr_v2/cecd_listing_runtime_v1/huatuo/pilot \
  --model huatuo --split pilot
```

Evaluation:

```bash
PYTHONPATH=.:anchor .venv-full/bin/python -m \
  corrected_sgta.run_vindr_cecd_listing_runtime_v1 evaluate \
  --experiment-manifest corrected_runs/vindr_v2/cecd_ontology_listing_substrate_v1/experiment_manifest.json \
  --reference corrected_runs/vindr_v2/cecd_ontology_listing_substrate_v1/reference_images.jsonl \
  --run-dir corrected_runs/vindr_v2/cecd_listing_runtime_v1/huatuo/pilot \
  --output-dir corrected_runs/vindr_v2/cecd_listing_runtime_v1/huatuo/pilot/evaluation
```

## Validation

- Focused runtime tests: `6 passed`.
- Combined runtime + listing admission + ontology substrate: `16 passed`.
- Explicit real-pack missing-admission CLI: exit code `1`, no run directory
  created, and failure occurred at the admission gate.
- Actual pack engineering-failure audit: exactly the one frozen image/pair
  above.
- `git diff --check` on the new runtime/test files: clean.

The runtime now additionally requires the exact write-once human-adjudication
handoff and canonical three-stage binary-CE GO. The receipt must hash-bind all
eight frozen returns, completed clinical/prompt adjudication, two distinct
adjudicator attestations, the canonical validator/assembler sources, the
canonical locked confirmation, and all frozen stage-selection identities.
Until those genuine artifacts exist, this runtime remains inert.

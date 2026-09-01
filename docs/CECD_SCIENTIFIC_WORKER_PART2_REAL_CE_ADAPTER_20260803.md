# CECD Scientific Worker Part 2：真实 CE adapter

**Date:** 2026-08-03  
**Verdict:** Huatuo/Hulu single-token CE adapter and seven centered-logit
controls are implementation-ready and CPU-preflighted on one local VinDr
sample. GPU smoke was not launched because the shared GPU and its frozen lock
were occupied by the authorized unified-evaluation OE run. No placeholder or
formal scientific output was written.

## Implemented boundary

`cecd_dual_semantics_ce_adapter_v1.py` constructs an exact two-render x
two-prompt orbit for one fixed clinical claim:

```text
h00 = render_0, prompt_0
h10 = render_1, prompt_0
h01 = render_0, prompt_1
h11 = render_1, prompt_1
```

The engineering defaults are the audited DICOM displays
`baseline_percentile` and `native_linear`, and the proposition/speech-act
matched `existential` and `radiograph_subject` polar prompts. Every selected
render must pass the existing finite/saturation/edge engineering guard.
Clinical equivalence remains explicitly false until the independent admission
closes.

Both real scorers reuse the Stage-1 pathways rather than inventing new model
wrappers:

- Huatuo: multimodal embedding expansion followed by its Qwen2 decoder and
  FP32 Yes/No/Maybe LM-head readout;
- Hulu: native adaptive visual-token processor followed by its Qwen3 decoder
  and the same FP32 three-token readout.

The seven controls operate only after subtracting each cell's common logit
offset. This gauge fixing is necessary because softmax is invariant to that
offset and prevents irrelevant cellwise logit shifts from becoming a fake
factorial interaction. `random_norm` is sampled inside the centered-logit
subspace, orthogonalized against the observed interaction, and rescaled to its
exact quotient-space norm. Every resulting method logit vector is re-centered
before softmax.

## Atomicity and resume

The engineering pilot freezes:

- full checkpoint/processor/template/hook/transport fingerprint;
- adapter, worker, kernel, renderer, scorer and model-runtime source hashes;
- manifest and DICOM content hashes;
- selected record, two render pixel hashes and two prompt hashes;
- method order, seed, GPU lock and explicit engineering-only authority flags.

Each real forward is written atomically as `cells/h00.json` through
`cells/h11.json`. Resume reuses only a shard with the same config fingerprint,
cell identity and finite exact three-state logits. A malformed shard is scored
again; config or source drift aborts. If all four cells exist, resume computes
the seven controls without loading a model. The summary always contains:

```text
scientific_status = engineering_only_no_scientific_authorization
formal_method_output_authorized = false
oe_adapter_implemented = false
cecd_hidden_intervention_implemented = false
treble_variants_implemented = false
paper_claim_authorized = false
```

The adapter uses a nonblocking `flock` on
`corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock`. Lock contention aborts
before model construction.

## Real CPU preflight

Both families were run against the same deterministically selected local VinDr
claim:

```text
aortic_enlargement__d925309691e7929d905eaa42f081833f__badf6249e030
```

The DICOM existed, both render guards passed, all code/model/input hashes
closed, and both returned:

```text
status = cpu_preflight_passed_no_model_or_cuda
scientific_status = engineering_only_no_scientific_authorization
model_loaded = false
cuda_initialized_by_adapter = false
```

Temporary audit hashes:

```text
b6ba5942fba66e4fbe7bf845e631a10c5ee3388d394351fa7dd37cc6ee4776d6  /tmp/cecd_huatuo_ce_adapter_preflight.json
594876b9adaf17d26ccbd8675b7f39e2f59c5ed246e504f71035ce7f1a150576  /tmp/cecd_hulu_ce_adapter_preflight.json
```

GPU use did not change during either preflight. During the first check, PID
`590308` used 19,590 MiB at 96--97% utilization and held the exact shared lock
for a Huatuo OE control run. When that process finished, PID `601956`
immediately continued the paired Hulu OE control run with 37,414 MiB and the
same lock. Per the frozen rule, no CE GPU smoke was queued, forced or run.

## Tests

```bash
cd /home/dbw/ANCHOR
PYTHONPATH=.:anchor .venv-full/bin/pytest -q \
  tests/test_cecd_dual_semantics_ce_adapter_v1.py \
  tests/test_cecd_dual_semantics_scientific_worker_v1.py \
  tests/test_run_cecd_dual_semantics_controlled_v1.py \
  tests/test_treble_collision_contract.py
```

Result:

```text
.................................                                        [100%]
33 passed in 1.32s
```

The six new adapter tests cover cellwise gauge invariance, centered random
energy matching, exact method closure, atomic shard resume/corruption recovery,
nonfinite-logit rejection, write-once config drift, real nonblocking GPU lock
contention and permanent engineering-only metadata.

## Remaining gaps

1. **GPU engineering smoke:** not run because the shared GPU was occupied; no
   model-output claim is made.
2. **Formal CE:** the current CLI is deliberately engineering-only. Admission,
   Stage-3 split and formal runner binding remain upstream work.
3. **Aligned OE:** absent. The controlled runner cannot accept CE-only arms.
4. **CECD hidden intervention:** absent; no hidden hook is installed.
5. **Treble proceedings/released:** absent; neither semantic variant is
   impersonated by centered-logit averaging.
6. **Dynamic baseline:** absent from the current v1 preflight and must be frozen
   before authorization/output.
7. **Clinical inference:** one sample and synthetic/unit tests prove plumbing
   only, not mitigation efficacy or mechanism validity.

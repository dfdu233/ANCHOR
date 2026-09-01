# CECD scientific worker：outcome-blind Part 1

**Date:** 2026-08-03  
**Decision:** CPU/runtime and mathematical-kernel layer implemented; every real
model arm remains fail-closed. No authorization, sealed outcome, model load,
CUDA initialization or scientific output was used.

## What is now real

`cecd_dual_semantics_worker_v1.py` implements the descriptor protocol consumed
by the controlled runner. For each of Huatuo and Hulu it independently binds:

- every file under the model directory, including every weight shard;
- tokenizer, processor and image-processor subset as a separate digest;
- exact polar and aligned-OE prompt/template contract;
- deterministic generation and no-exchange contract;
- architecture-specific hook paths and layer counts;
- vision-token transport semantics, including an explicit Hulu
  `not_admitted_not_implemented` state;
- the worker, factorial kernels, runner, collision contract, model runtime
  sources and input-binding sidecar;
- calibration manifest, locked evaluation manifest, ordered record keys and
  atomic-claim contract against preflight hashes;
- Python, Torch, Transformers, NumPy and Pillow package versions without
  importing Torch or initializing CUDA.

The input sidecar is fixed at
`<preflight-stem>.inputs.json` and has schema
`cecd-dual-semantics-input-bindings-v1`. It must contain exactly the current
preflight hash, two model directories, the Huatuo external source root and the
four input paths. The descriptor re-hashes every target; a path declaration is
never treated as evidence by itself.

## Real local fingerprint check

A CPU-only full-content pass over the installed checkpoints completed without
loading either model:

| Family | Time | Checkpoint | Processor | Template | Hook | Vision transport |
|---|---:|---|---|---|---|---|
| HuatuoGPT-Vision-7B | 7.839 s | `b1d83471...f48abd` | `3f9196bc...10ec58` | `8865432c...6f15bb` | `0a63bb2e...0258dd` | `5641848e...1e31b` |
| Hulu-Med-4B | 4.298 s | `dc5cd2f8...09890` | `72b302d7...81772` | `d132640b...5ff94` | `20196a92...1edfd` | `76e00f13...65f959` |

The hook descriptor recovered Huatuo's 28 decoder and 24 vision layers and
Hulu's 36 decoder and 27 vision layers. The common generation-contract digest
is `8219d2c0...0f1381` because decoding/no-exchange semantics are intentionally
shared; architecture-specific hook and transport digests remain different.

These are implementation fingerprints, not scientific results. The future
preflight must contain these exact full records and will fail if any model,
processor, template, runtime source or contract changes.

## Seven architecture-neutral kernels

For a balanced four-cell activation orbit, define:

\[
\begin{aligned}
g &= (h_{00}+h_{10}+h_{01}+h_{11})/4,\\
r &= (-h_{00}+h_{10}-h_{01}+h_{11})/4,\\
p &= (-h_{00}-h_{10}+h_{01}+h_{11})/4,\\
i &= (h_{00}-h_{10}-h_{01}+h_{11})/4.
\end{aligned}
\]

`cecd_dual_semantics_kernels_v1.py` freezes the following unambiguous pure
NumPy controls at the joint target cell:

| Method | Kernel output | Status |
|---|---|---|
| `unmitigated` | `g+r+p+i = h11` | kernel passed |
| `full_orbit` | `g` | kernel passed |
| `render_only` | `g+p = (h01+h11)/2` | kernel passed |
| `prompt_only` | `g+r = (h10+h11)/2` | kernel passed |
| `random_norm` | `g+r+p+i_random` with last-axis norm matched and interaction-orthogonal random direction | kernel passed |
| `sign_permuted` | `g+r+p+permute(i)` with exact last-axis energy preservation | kernel passed |
| `main_effect_removal` | `g+i` | kernel passed |

The synthetic conformance checks exact orbit reconstruction, closed forms,
determinism, random-direction orthogonality and interaction-energy matching.
At seed 42 its maximum numerical errors are at floating-point noise level:

```text
reconstruction                    4.440892098500626e-16
closed form                       4.440892098500626e-16
random norm                       4.440892098500626e-16
sign-permuted norm                4.440892098500626e-16
random orthogonality              5.551115123125783e-16
determinism                       0.0
gpu_used                          false
scientific_model_output           false
```

## What is deliberately not implemented

| Method group | Mathematical kernel | Real Huatuo adapter | Real Hulu adapter | Formal behavior today |
|---|---:|---:|---:|---|
| Seven controls above | yes | no | no | `kernel_implemented_real_model_adapter_not_implemented` before output/model/GPU |
| `cecd_interaction_projection` | no formal intervention | no | no | `method_not_implemented` before output/model/GPU |
| `treble_proceedings` | representation arithmetic only exists in collision contract | no | no | `method_not_implemented` before output/model/GPU |
| `treble_released` | released shift arithmetic only exists in collision contract | no | no | `method_not_implemented` before output/model/GPU |
| required dynamic activation baseline | not yet frozen | no | no | absent from current v1 preflight; must be added before authorization/output |

This separation is essential. An activation-array average is not yet a valid
autoregressive CE/OE method: a real adapter must freeze token alignment, hook
location, cache semantics, logits versus probability averaging, teacher-forced
claim spans, generation length and compute accounting. Therefore the worker
does **not** emit a placeholder `completion.json` for any formal arm.

Treble remains especially blocked. The proceedings and released-source
variants require separate counterfactual collection, PCA/direction fitting and
intervention semantics. Hulu's adaptive visual grid has no admitted token
transport. Resizing, pooling or interpolation would create a surrogate and is
explicitly forbidden from being labelled source-faithful. CECD hidden editing
also remains blocked until noising/denoising, activation-distance and
joint-cell-specific hook conformance are implemented.

## Fail-closed formal path

A formal worker request first revalidates the existing narrow authorization,
preflight, run-contract fingerprint, exact two-model closure, method membership
and output-root confinement. It then raises `MethodNotImplementedError` before:

1. importing Torch;
2. loading a tokenizer or model;
3. initializing CUDA;
4. creating the requested output directory;
5. writing a completion or metric artifact.

This means the master runner can safely adopt the scientific worker for
descriptor preparation later, while execution remains impossible until a real
adapter satisfies its own conformance gate.

## Verification

```bash
cd /home/dbw/ANCHOR

PYTHONPATH=.:anchor .venv-full/bin/python -m \
  anchor.corrected_sgta.cecd_dual_semantics_worker_v1 \
  --synthetic-conformance

PYTHONPATH=.:anchor .venv-full/bin/pytest -q \
  tests/test_cecd_dual_semantics_scientific_worker_v1.py \
  tests/test_run_cecd_dual_semantics_controlled_v1.py \
  tests/test_treble_collision_contract.py
```

Current focused result:

```text
...........................                                              [100%]
27 passed in 1.72s
```

The seven new worker tests cover exact factorial arithmetic, randomized-control
energy/orthogonality, malformed orbit rejection, CPU-only synthetic status,
real descriptor closure on two architecture-shaped fixtures, input/model/
sidecar drift, and formal failure before output creation for every current
method family.

The broader authorizer/transition tests are not claimed here because a
concurrent repository update migrated production code from the former
two-model Stage-1 path to `cecd-three-stage-v3` while its older fixtures still
construct the legacy path/state shape. At this freeze they produce nine
legacy-state/path failures before reaching the worker. This Part-1 change does
not edit those concurrent files or reinterpret those failures as worker
evidence.

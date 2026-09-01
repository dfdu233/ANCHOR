# HALP three-plane compatibility preflight

**Date:** 2026-08-03  
**Status:** CPU/source audit and fake-hook conformance complete. No probe was
trained, no hallucination outcome was read, and no scientific result was run.

## Boundary of the port

[HALP (EACL 2026)](https://aclanthology.org/2026.eacl-long.287/) defines three
pre-generation representation families: globally pooled visual features before
multimodal projection, the decoder state at the final visual token, and the
decoder state at the final query token. The official repository was inspected
at commit `bdc529453dc9ce886956f70f8c2175d75b1ea7a8`.

The local implementation preserves those conceptual positions but is called a
**HALP-style compatibility port**, not a HALP reproduction. This distinction
is material:

- the official LLaVA extraction script estimates the visual-token count from
  sequence length, while this port derives the exact architecture-specific
  visual span and fails if it is ambiguous;
- the official released probe scripts use ordinary random train/test splits,
  while this medical protocol requires dev-only group CV by global image ID;
- Huatuo and Hulu do not expose equivalent visual latents even when both are
  sampled immediately before the projector.

No placeholder vectors are emitted on capture failure. The official LLaVA
script's zero-vector fallback is intentionally not reproduced.

## Exact representation semantics

### `visual_only`

One global mean over the model-native vision encoder output immediately before
`mm_projector`.

- **Huatuo / LlavaQwen2:** CLIP selected layer `-2`, patch tokens after CLS
  removal, before the two-layer GELU projector. The hook is on
  `get_vision_tower()`.
- **Hulu / custom Qwen3:** custom vision encoder output after its final layer,
  post-layer normalization and model-native bilinear spatial interpolation,
  before the two-layer GELU projector. The hook is on
  `get_vision_encoder()`.

These are positionally analogous but not identical latent variables. Huatuo
uses a penultimate CLIP feature grid; Hulu uses a final custom-encoder feature
grid after a different spatial operation.

### `decoder_vision_token`

At every decoder block, capture the raw post-block/pre-final-norm residual
state at the **last exact visual-token position**.

- **Huatuo:** one negative image placeholder is replaced by `P` projected
  patches. If its original position is `s`, then
  `P = expanded_length - original_length + 1` and the captured position is
  `s + P - 1`.
- **Hulu:** the processor materializes a contiguous run of
  `image_token_index=151669`; the projector output replaces these positions
  in-place. The config is admitted only when `use_token_compression=false` and
  prepared length equals tokenized length. The captured position is the final
  member of that verified run.

### `query_token`

At every decoder block, capture the raw post-block/pre-final-norm state at the
last active context token used to predict the first answer token. It is
verified to lie outside the visual span. This is a chat-template boundary
token, not necessarily the lexical final word of the user's question; the two
models' query-token surface semantics therefore remain architecture/template
specific.

The capture performs one teacher-free pre-generation forward and never calls
`generate`.

## Layer contract

Layers are enumerated as one-based decoder blocks only; the embedding state is
not called “layer 0.” The hook location is fixed to block output after the
residual update and before the model's final RMSNorm.

```text
Huatuo: model.layers.0 ... model.layers.27  -> layers 1 ... 28
Hulu:   model.layers.0 ... model.layers.35  -> layers 1 ... 36
```

Each row binds:

- one-based layer and zero-based block index;
- normalized depth `layer / total_layers`;
- qualified module path and runtime class;
- parameter count and ordered parameter name/shape/dtype schema hash;
- the source-audit fingerprint and ordered layer-contract fingerprint.

Absolute layer numbers are never compared across the 28- and 36-layer models.
Candidate selection is model-family-specific; normalized depth is descriptive
and a dev tie-breaker, not an assumption of functional equivalence.

The final selected hook is checked causally against the decoder output:
applying the native final norm to the captured final-block tensor must recover
the model's final hidden state at both the final visual and query positions.

## Split discipline

The frozen policy is:

```text
selection split       = dev only
group unit            = global image_id
CV                     = 5-fold stratified group CV
family-specific choice = true
primary choice metric = group-CV AUROC
tie-breakers           = group-CV Brier, then shallower normalized depth
confirmation           = apply only
```

Confirmation forbids refitting, layer selection, plane selection, threshold
tuning and outcome access during capture. A selection-policy fingerprint binds
all of these fields. This preflight defines and tests the contract only; it
does not fit even a fake probe.

## CPU/source audit

Both audits ran in fresh CPU-only processes and verified local config,
checkpoint index, projector/encoder integration source markers, capture source
hash, all layer rows and the selection policy.

```text
Huatuo: llava_qwen2, 28 layers, hidden size 3584
Hulu:   hulumed_qwen3, 36 layers, hidden size 2560
status: cpu_source_audit_passed_no_model_or_cuda
```

Artifacts:

```text
corrected_runs/vindr_v2/halp_three_plane_preflight_v1/huatuo/source_audit.json
corrected_runs/vindr_v2/halp_three_plane_preflight_v1/hulu/source_audit.json
```

Final file hashes:

```text
a36cb87d16ca92333adf5249c12ea7e45dfc9f32ecebce0d39c380940c21c871  huatuo/source_audit.json
f2ca4bee1b856698cc19a5c9b7bbb80f8935d4e5573e089c0a04ca5707e251a0  hulu/source_audit.json
```

An initial attempt with the host system Python stopped at import because that
interpreter has no NumPy. It failed before creating either audit artifact and
before any model or CUDA access. Both final audits were then produced with the
repository's `.venv-full` interpreter.

## Fake-hook conformance

Seven tests cover:

- both real source/config audits without model loading or CUDA;
- dev group-CV and confirmation apply-only tamper rejection;
- exact global visual mean, final visual position and final context position;
- every post-block layer and final-norm location conformance;
- deterministic ordered runtime layer hashes and layer-count drift rejection;
- exact-once, non-mutating vision hook removal;
- ambiguous span/query failure;
- permanent engineering-only metadata that cannot masquerade as a trained
  probe or official HALP reproduction.

```text
7 passed
```

## Deferred work

No real GPU capture was launched in this preflight: the requested deliverable
was narrowed to compatibility artifacts and fake hooks, and GPU0 was occupied
by PID `623910` at the final check. The real
single-claim entry point exists but, when used later, remains a plumbing-only
capture with `probe_trained=false`, `outcome_read=false`, and
`paper_claim_authorized=false`.

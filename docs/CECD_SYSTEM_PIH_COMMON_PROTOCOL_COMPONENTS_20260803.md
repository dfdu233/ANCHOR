# CECD System/PIH common-protocol CPU component implementation

Date: 2026-08-03  
Scope: independent clean-room CPU/FakeTensor components and fail-closed preflight only  
Non-scope: no GPU, no target-model forward pass, no model output, no head selection, no scientific outcome

## Result

The architecture-neutral tensor semantics needed for two adversarial controls are now implemented and hash-bound.  This does **not** make either control executable on Huatuo or Hulu: model-specific runtime integration and numerical-canary artifacts remain deliberately absent and are separate hard blockers.

The admissible system-control name remains `pre_image_prefix_attention_redistribution_control`.  The source is a positional prefix; current locked prompts contain no proven system-role source span.

## Bound generic components

| Preflight binding | File | Semantics |
|---|---|---|
| `dynamic_span_builder` | `anchor/corrected_sgta/cecd_dynamic_span_builder_v1.py` | Role provenance after multimodal expansion; dynamic contiguous image run; right-padding and generated-token exclusion; system-role proof kept distinct from positional prefix. |
| `system_attention_source_patch` | `anchor/corrected_sgta/cecd_positional_prefix_attention_v1.py` | Clean-room callable/reference kernel at the post-FP32-softmax, pre-value boundary; this is not a target-model runtime patch. |
| `system_numerical_canary` | `anchor/corrected_sgta/cecd_system_numerical_canary_v1.py` | Interface for comparing native/eager first-token logits; this is not a completed target-model canary artifact. |
| `pih_mean_ablation_hook` | `anchor/corrected_sgta/cecd_pih_mean_ablation_v1.py` | Batch-one, per-sample frozen-prefix mean replacement on concatenated query heads before `o_proj`, including cached-decode state isolation. |
| `pih_selection_runner` | `anchor/corrected_sgta/cecd_pih_dev_selection_schema_v1.py` | Outcome-free dev-manifest validation and candidate enumeration only; it has no outcome, ranking, selection, or artifact-writing operation. |

The binding names are retained for preflight compatibility, but `generic_tensor_components_bound=true` is explicitly separate from per-model execution readiness.

## Tensor invariants closed

- Expanded spans are mutually exclusive and exhaustive and cannot use fixed boundaries such as 35 or 576.
- A frozen prefix cannot truncate an expanded image run.
- Current Huatuo/Hulu prefixes remain positional user delimiters, not system tokens.
- Attention input must be FP32, finite, nonnegative, row-normalized post-softmax probabilities.
- Primary redistribution uses only the last frozen-prefix query row and conserves mass to absolute tolerance `1e-6`.
- `alpha=1` is exact identity.
- `source_zero` is explicitly labeled as the non-conserving deficit control and its deficit must equal removed source mass.
- Proportional image+suffix, image-only, suffix-text-only, and seeded equal-width random-source controls are implemented.
- GQA geometry is explicit: Huatuo `28/4/128`, Hulu `32/8/128` query/KV/head width.
- PIH head width is derived from `o_proj.in_features / num_query_heads`: Huatuo `3584/28=128`, Hulu `4096/32=128`; Hulu hidden-size inference `2560/32=80` is inadmissible.
- PIH batch size is fixed to one, the mean is per sample over frozen-prefix tokens, and cached decoding can only reuse that sample's detached prefix mean.

## New runtime fail-closed boundary

`configs/cecd_system_pih_control_preflight_v1.json` and its validator now require, separately for Huatuo and Hulu:

1. a real model-runtime eager-attention source patch;
2. a hash-bound native-vs-eager first-token canary artifact produced before intervention;
3. a real model-runtime pre-`o_proj` hook integration.

All six artifacts are currently null and both runtime statuses are `not_implemented`.  PIH selected/random head artifacts also remain null and selection status remains `not_implemented`.  The current preflight therefore correctly returns `passed=false`, `control_execution_ready=false`, `generic_tensor_components_bound=true`, and `per_model_runtime_integration_ready=false`.

Current blockers are exactly:

```text
huatuo:native_eager_canary_artifact_missing
huatuo:pih_o_proj_runtime_integration_missing
huatuo:pih_selection_not_ready
huatuo:runtime_integration_not_ready
huatuo:system_attention_runtime_patch_missing
hulu:native_eager_canary_artifact_missing
hulu:pih_o_proj_runtime_integration_missing
hulu:pih_selection_not_ready
hulu:runtime_integration_not_ready
hulu:system_attention_runtime_patch_missing
```

Thus future selected-head files alone cannot accidentally authorize execution.

## Verification

Command:

```bash
PYTHONPATH=.:anchor .venv-full/bin/pytest -q \
  tests/test_cecd_system_pih_common_protocol_v1.py \
  tests/test_validate_cecd_system_pih_control_preflight_v1.py
```

Result: `25 passed in 1.23s`.

The tests cover source semantics, dynamic lengths, recipient variants, mass conservation, exact identity, explicit source-zero deficit, GQA geometry on both models, FakeTensor shape execution, PIH lifecycle/no-cross-batch behavior, numerical-canary behavior, outcome-free selection schema, hash drift, cross-model selected-head reuse, and runtime fail-closed status.

`py_compile` and scoped `git diff --check` also pass.  No control output directory was created or read.


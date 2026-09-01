# System-Mediated Attention / PIH architecture compatibility audit

Date: 2026-08-03  
Scope: CPU/source-only, outcome-blind; HuatuoGPT-Vision-7B and Hulu-Med-4B  
Non-scope: no GPU/model execution, no outcome read, no worker/listing change, no CECD three-stage threshold change

## Executive verdict

Neither control can be represented as a paper-native or official-code faithful port on the two target models.

1. The released System-Mediated Attention implementation is a modified LLaVA-1.5/LLaMA attention stack with hard-coded `img_start_pos=35` and `img_length=576`. Both target decoders are GQA Qwen variants. More importantly, the locked Huatuo and Hulu prompts contain **no true system-role content**: the pre-image prefix is a user-role delimiter. Moving that prefix's attention mass is a useful positional control, but calling it a system-attention intervention would be a category error.
2. PIH's pre-`o_proj` mean-ablation operation is mechanically portable in an independent implementation. The released hook works shape-wise for Huatuo Qwen2, but its `hidden_size / num_heads` assumption fails on Hulu Qwen3: Hulu has hidden size 2560, 32 query heads, explicit head dimension 128, and `o_proj.in_features=4096`. The official code would infer 80 and fail its own shape assertion.
3. The repositories have no LICENSE file or explicit license grant at the frozen commits. Their equations and behavior can inform a clean-room control, but their source is not copied or redistributed here.
4. PIH head sets are model-specific. Official Qwen/LLaVA/Janus head IDs are inadmissible for Huatuo or Hulu. Selection must be patient/image-disjoint dev-only, independently per model; locked test data cannot be scanned.

The admissible names are therefore:

- `pre_image_prefix_attention_redistribution_control`, explicitly a positional-prefix control rather than a system-instruction mechanism;
- `dev_selected_prompt_copy_head_mean_ablation_control`, an independent architecture-neutral PIH-inspired control.

## Frozen primary sources

### System-Mediated Attention Imbalances Make VLMs Say Yes

- Paper: [ACL Findings 2026, paper 1940](https://aclanthology.org/2026.findings-acl.1940/)
- Official author repository: [anisha0325/vlm-hallucination-yes-bias](https://github.com/anisha0325/vlm-hallucination-yes-bias)
- Frozen commit: `343ad9da36bd23e555816b7531da626234aeda22`
- Git tree-listing SHA-256: `1d4a13510262d0e2539d28eb47d39757a1f49471d13a663b4be2bcec200aeb6d`
- Paper PDF SHA-256: `dc1a8486fe0fb55d891c17af91f10a5df6be841e7af3e35594f9b160d3db52ba`
- License at frozen commit: no LICENSE file and no explicit grant found.
- Provenance warning: the repository README's clone target differs from the ACL page's linked author repository.
- Release warning: `y25_fm_sys.sh` references `all_heads_llava157b.json`, which is not tracked. The tracked Q4 file is `layers25to32_llava157b.json`.

Key source hashes:

| Frozen path | SHA-256 |
|---|---|
| `README.md` | `bf8bc832a85357136c01c03f5554f150f02ebfcf8f7bd77c7e1ce77bcc34268b` |
| `LLaVA/llava/model/language_model/modeling_llama.py` | `0fc835999482c3b026db801c0e15903ef62a929a9c34558f5485978fbad41c17` |
| `LLaVA/eval_scripts/analyze_attention_reweight_matrices_fm_sys.py` | `9fc9e1a9b01588fd1fb636605c1803df57fdbebe9adb9cc4beaf14e7d46f1717` |
| tracked Q4 head JSON | `1530d06d9a6990261292e6bfb08a31dc24fe10bf57bb88e81745857d4f74a9d0` |
| `LLaVA/bash_scripts/y25_fm_sys.sh` | `73f556e1cb6d90ad3d7cb410fce01fe6fb96a3a7fbeb3dc4fba82b4ee72dd3c0` |

### Mechanisms of Prompt-Induced Hallucination in VLMs

- Paper: [ACL 2026, long paper 1941](https://aclanthology.org/2026.acl-long.1941/)
- Official author repository: [michalg04/prompt-induced_hallucinations](https://github.com/michalg04/prompt-induced_hallucinations)
- Frozen commit: `f301d68ed3743417c29c384c632838523610c9c4`
- Git tree-listing SHA-256: `cfd1f2983c760b23f757a0edcb1919d51403c3df33e46b2fc240f9dd52c5c73f`
- Paper PDF SHA-256: `e4036d8fdfc82abd8b13a53084942562b098a82a55fbcdd67eac441b7691bac4`
- License at frozen commit: no LICENSE file and no explicit grant found.

Key source hashes:

| Frozen path | SHA-256 |
|---|---|
| `README.md` | `d5060e617506c17b5508c5216678988772222fbed26a056df37298ce2b18dc05` |
| `knockout_utils.py` | `1e4cb59172100e4826ed51d1f6725edd14e674f681a0269c5a64b56e43a1dc09` |
| `3_knockouts.py` | `9dad1bc6e635475aae108c7c61b8a9dc0483c75224520e46c80b9acc83ab0d03` |
| `4_evaluate_knockouts.py` | `c16ff61656de3baacbf7a47902616012124e1dab742c2ce41e1fb71fcbb7f145` |
| `5_attention_mass.py` | `5d1866b277f999281f93480c686ce2e033ff97feedf0c5648264f8b54cf1758e` |
| `requirements.txt` | `73bf4de4e24152da6efaa02e01f05a5b36a69c07350077151ad9095602a22b8d` |

## Exact method semantics

### System-Mediated Attention

The official intervention is inside eager attention, after softmax and before multiplication by values. For selected query heads it multiplies source-modality mass by alpha, computes removed row-wise mass, assigns recipient modalities a share proportional to their original modality mass, then spreads each share uniformly over tokens in that modality. It does not apply a second softmax. The main setting uses alpha 0, all heads in LLaVA-1.5's fourth quarter (paper layers 25–32, code indices 24–31).

The released code partitions keys positionally:

```text
system := [0, 35)
image  := [35, 35 + 576)
text   := [35 + 576, kv_length)
```

This is not a role-aware system span. It only behaves like one under the paper's exact prompt layout. Its proportional branch also has a zero-recipient edge case for causally early query rows; a mass-conserving common protocol must intervene only where recipient mass is finite and greater than epsilon, or restrict the operation to the final frozen-prefix query.

### Prompt-Induced Hallucination

The official operation registers a forward pre-hook on `layer.self_attn.o_proj`. It receives the concatenated query-head output `[B,T,H]`, slices one head, computes its mean over batch and token axes, and broadcasts the mean over that head's original positions. Official experiments are effectively batch one; a formal port must either require batch one or compute a per-sample token mean, never mix samples.

Individual heads are ranked by whether ablation changes a prompt-induced wrong count to the ground-truth count. Group sizes 1, 3, 5 and 10 are compared, with a random control matched to selected layers and head count. The released scripts do not provide a sufficiently explicit independent dev/test freeze for head discovery and reporting. Our protocol corrects this with image-disjoint dev-only selection. That correction is scientifically necessary but is another reason not to call the result paper-native.

## Architecture audit

| Property | HuatuoGPT-Vision-7B | Hulu-Med-4B |
|---|---:|---:|
| Decoder | Qwen2, Transformers 4.37.2 | custom Qwen3, Transformers 4.51.2 |
| Decoder layers | 28 | 36 |
| Query / KV heads | 28 / 4 | 32 / 8 |
| Hidden size | 3584 | 2560 |
| Pre-`o_proj` width | 3584 | 4096 |
| Query-head width | 128 | 128 |
| Locked attention backend | eager | SDPA |
| Visual tokens | fixed 576 in primary path | processor-derived variable contiguous run |
| True system-role tokens in locked prompt | none | none |

Both decoder layer paths resolve as `model.model.layers[i].self_attn`; PIH hooks target `.o_proj`. System redistribution needs an internal eager-attention source patch, not an ordinary module hook. Hulu's SDPA runtime must use a separate eager control path and pass a native-vs-eager first-token numerical canary before any intervention.

Huatuo prepends `<image>` to the user message, then renders `<|user|>\n...<|assistant|>\n`; the only pre-image text is the user delimiter. Hulu renders a Qwen user message whose first content item is image and currently calls the processor with `add_system_prompt=False`; again the pre-image text is a user delimiter. Therefore a true system-source intervention is unavailable without changing the prompt surface.

## Frozen architecture-neutral contracts

### Expanded token spans

Spans must come from provenance after multimodal preparation, excluding padding and generated tokens:

```text
system    = expanded keys originating from an explicit system-role message
image     = expanded/projected visual keys
user_text = all other unpadded textual prefix keys, including role delimiters
```

They must be mutually exclusive and exhaustive, and image keys must form the observed dynamic contiguous span. Magic boundaries 35 or 576 are rejected as formal boundaries. A separate positional partition `prefix_before_image / image / suffix_after_image` is permitted, but the prefix may be called system only if role provenance proves every prefix key is a system-role key.

### Positional-prefix redistribution control

- Hook: eager attention after FP32 softmax, before value matmul.
- Rows: final frozen-prefix query only, avoiding causally unavailable recipient rows.
- Layers: relative final quarter, fixed without outcome selection: Huatuo 21–27; Hulu 27–35, zero-indexed.
- Heads: all query heads.
- Alpha: 0; distribute to image and suffix in proportion to original row-wise modality mass, uniformly within modality.
- Required controls: alpha-one identity; zero source without redistribution; image-only and text-only recipients; equal-width random key span; native-vs-eager first-token canary.
- Required invariant: finite values and per-row mass conservation within absolute tolerance `1e-6`.
- Interpretation: positional-prefix competition only. It cannot establish a system-instruction mechanism on current prompts.

### PIH-inspired mean-ablation control

- Hook: forward pre-hook at decoder self-attention `o_proj`.
- Head width: `o_proj.in_features / num_query_heads`; exact divisibility required.
- Replacement: per-sample mean over frozen-prefix tokens; batch size one; no cross-sample averaging.
- Discovery: exhaustive/model-appropriate layer-head sweep on patient/image-disjoint dev only.
- Score: reader-grounded correction of prompt-copy error, subject to non-degradation on aligned prompts.
- Group sizes: 1, 3, 5, 10, selected on dev independently per model.
- Locked test: never scanned during sweep, ranking or group-size selection.
- Random baseline: same model, same selected-layer multiset, same head count, frozen seed.
- Head-set artifacts: selected and random sets are hash-bound and cannot be reused across Huatuo/Hulu.

## Fail-closed implementation status

The executable contract is in:

- `anchor/corrected_sgta/validate_cecd_system_pih_control_preflight_v1.py`
- `configs/cecd_system_pih_control_preflight_v1.json`
- `tests/test_validate_cecd_system_pih_control_preflight_v1.py`

Generic CPU/FakeTensor semantics are now implemented and hash-bound as documented in `docs/CECD_SYSTEM_PIH_COMMON_PROTOCOL_COMPONENTS_20260803.md`: dynamic expanded spans, a post-softmax/pre-value reference redistribution kernel, a numerical-canary interface, a per-sample PIH mean-ablation hook, and an outcome-free dev-selection schema skeleton. These bindings do not authorize target-model execution.

The preflight remains deliberately blocked because neither Huatuo nor Hulu has a hash-bound real attention-runtime patch, a completed native-vs-eager first-token canary artifact, or a verified real `o_proj` runtime integration. Dev-selected and matched-random head artifacts also remain absent. Generic kernels and per-model runtime readiness are now separate preflight fields, so future selected-head artifacts alone cannot make the control executable. The gate also remains blocked on source/hash drift, magic visual boundaries, wrong Hulu head width, test-set selection, cross-model head reuse, missing same-layer random controls, or nonempty outputs created before the contract freeze.

This status is a compatibility result, not an experimental result and not evidence for or against CECD.

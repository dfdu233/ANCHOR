# CECD System/PIH common-protocol：Huatuo/Hulu runtime integration v1

**日期：** 2026-08-03  
**结论：** Huatuo-Qwen2 与 Hulu-Qwen3 的 clean-room source-level runtime
integration 已实现，并通过 fake-module conformance；真实 native-vs-eager canary、
dev head selection 和 GPU/model run 均未执行。因此 preflight 仍正确地
`passed=false`。

## 许可与 claim 边界

ACL 2026 两篇相邻工作的官方仓库在冻结 commit 上均没有可识别的 license 文件或
明确授权。本实现没有复制其代码，只依据论文所描述的干预位置以及本地 Apache
Transformers/Huatuo 模型接口，独立实现 architecture-neutral common protocol。

允许的名字仍是：

```text
pre_image_prefix_attention_redistribution_control
dev_selected_prompt_copy_head_mean_ablation_control
```

禁止称为 paper-native、official implementation 或 faithful official port。当前 locked
prompts 的 pre-image prefix 是位置 surrogate，并没有 true system-role provenance，
所以也不能把结果称为 system-instruction mechanism。

## 精确 expanded provenance

新增源码：

```text
anchor/corrected_sgta/cecd_system_pih_runtime_integration_v1.py
```

它提供两个显式 builder：

- `build_huatuo_expanded_provenance`：要求唯一 image placeholder，把它替换为运行时
  实际 projected visual-token count；每个 patch token 记录原 placeholder 与 patch
  index。
- `build_hulu_expanded_provenance`：从 processor 已展开的 exact image token IDs 验证
  唯一非空连续 run，并要求 token roles 与这些 IDs 逐位置一致。

两者都保留：

- 每个 expanded slot 的 `system/image/user_text` role；
- source-token 或 expanded-image origin；
- right-padding mask；
- generated tokens 之外的 frozen prefix length；
- disjoint/exhaustive role partition；
- exact contiguous image span；
- provenance fingerprint。

不使用固定 `35` 或 `576` 边界，也不根据序列总长猜 image span。

## System-attention runtime

`EagerAttentionPatchContext` 会解析并验证完整 decoder layer closure：

```text
Huatuo: model.model.layers[0:28]
Hulu:   model.model.layers[0:36]
```

每层都验证 q/k/v/o projection、GQA heads 和 head width：

```text
Huatuo Q/KV/head = 28/4/128, o_proj input = 3584
Hulu   Q/KV/head = 32/8/128, o_proj input = 4096
```

### Huatuo Qwen2

Huatuo 的 native backend 已是 eager。Context 对每个 attention instance 临时绑定
clean-room Qwen2 forward，复用当前 pinned runtime 的 rotary helper 和 cache API，
但独立完成 projection、GQA repeat、mask、native-dtype score matmul、FP32 softmax、
value aggregation 与 output projection。

### Hulu Qwen3

Hulu 的 native backend 是 SDPA。Context 不修改共享 config 或全局 Transformers
registry，而是在每个 Qwen3 attention instance 上临时绑定 clean-room eager forward。
它保留 Q/K normalization、rotary、cache update、GQA 与 `4096 -> 2560` o projection，
形成明确的 `native_sdpa_to_clean_room_eager` control path。

### 精确干预边界

共同路径为：

```text
native-dtype QK score -> mask -> FP32 softmax
                      -> positional-prefix patch
                      -> cast to query dtype -> value matmul
```

Patch 只在 cache-free full prefill 的最后一个 frozen-prefix query 上执行一次；cached
decode 的 `Q=1, K>prefix` 完全不改。第二次 full prefill、decode-before-prefill 或未知
chunked shape 均 fail-closed。

Primary layer IDs 仍是预注册的 zero-indexed 范围：

```text
Huatuo 21..27
Hulu   27..35
```

Context 本身不从 outcomes 选择 layer/head。

## PIH pre-`o_proj` runtime

`PIHPreOProjPatchContext` 在选定 layer 的 `attention.o_proj` 上注册 instance-local
forward pre-hook：

- head width 只从 `o_proj.in_features / num_query_heads` 推导，两模型均为 128；
- batch 固定为 1；
- prefill 时只对当前样本 frozen prefix token axis 求 mean；
- 每层/每 head 独立保存 detached current-sample mean；
- cached decode 只使用同一个 context 生命周期内的 mean；
- empty、duplicate、越界 layer/head 在注册前 fail；
- 不允许 cross-sample reduction。

`SystemPIHRuntimeContext` 用 transactional `ExitStack` 组合两类 patch。普通退出、body
exception 或第二个 context 在 enter 中失败，都会倒序删除所有 hooks，并恢复原始
instance forward；模型 config 始终不被改写。

## Native-vs-eager canary 入口

源码包含真实 command entry，但本次没有调用：

```bash
PYTHONPATH=.:anchor .venv-full/bin/python -m \
  anchor.corrected_sgta.cecd_system_pih_runtime_integration_v1 canary \
  --family hulu \
  --model-factory package.module:build_model \
  --input-factory package.module:build_expanded_canary_input \
  --output /path/to/write-once/hulu.native_eager_canary.json
```

未来 input factory 必须返回 exact `ExpandedRoleTokenProvenance`、与其等长的 batch-one
already-expanded `inputs_embeds`、可选一致 attention mask、forward kwargs 与匿名 input
identity。Canary 强制 `model.eval()`、cache-free prefill、`use_cache=false`，先运行
unmodified native backend，再在无干预的 clean-room eager context 中运行同一输入，
比较 first-token logits、argmax、absolute/relative error。Artifact 同时绑定 integration
source 和两个 factory source hashes。

仅检查 CPU/source metadata 的安全入口为：

```bash
PYTHONPATH=.:anchor .venv-full/bin/python -m \
  anchor.corrected_sgta.cecd_system_pih_runtime_integration_v1 describe \
  --family huatuo
```

本次只运行过 `describe`，其 artifact 在 `/tmp`，未进入 scientific output root。

## Fake conformance

新增：

```text
tests/test_cecd_system_pih_runtime_integration_v1.py
```

覆盖：

1. Huatuo placeholder expansion 与 Hulu exact image-token run；
2. dynamic visual length、origin identity 与 role mismatch rejection；
3. Qwen2/Qwen3 exact forward tuple contracts；
4. Qwen2 `28/4/128` 与 Qwen3 `32/8/128` GQA；
5. additive mask、rotary helper 调用与 cache-free prefill；
6. system patch 只改变最后 frozen-prefix query，attention mass 保持 1；
7. generated decode query 不被 patch；
8. Hulu SDPA config 在 context 前后保持原值；
9. PIH `o_proj` pre-hook width 128、prefill/decode cache lifecycle；
10. body exception 与 enter-time invalid-head exception 后完整 forward/hook restoration；
11. fake Huatuo/Hulu native-vs-eager first-token canary；
12. real preflight source binding与 canary/head artifacts fail-closed 状态。

联合原 tensor/preflight 回归：

```text
37 passed in 1.32s
```

## `python -m` canonical identity incident and repair

The first explicit Huatuo adapter-backed canary failed closed before a model
forward with `input factory provenance has wrong type`.  It wrote no artifact,
and Hulu was not loaded.  The failure was not numerical: `python -m` executed
the runtime as `__main__`, while the factory imported its canonical package
name, creating two module objects and therefore two incompatible identities for
the same dataclass source.

The runtime now aliases `__spec__.name` to the executing `__main__` module in
`sys.modules` before local imports and before any runtime class definition.  It
also fails on a pre-existing conflicting canonical module rather than silently
accepting ambiguous identity.  This keeps the CLI and later factory imports on
one exact module object.

The regression test uses the real `python -m ... canary` entrypoint and a
temporary source-backed factory that imports the canonical runtime name and
returns its `ExpandedRoleTokenProvenance`.  It exercises both frozen
interpreters, not merely ordinary import semantics.  Current CPU-only results:

```text
Huatuo environment, runtime + adapter-factory suites: 22 passed
Hulu environment, isolated dash-m regressions:        2 passed
Hulu environment, adapter-factory suite:              8 passed
```

No checkpoint or GPU was used for the repair validation.  The failed output
path remained absent and no canary result was promoted.

## Preflight 更新与仍未满足的门

`configs/cecd_system_pih_control_preflight_v1.json` 现在为两个模型分别绑定同一份
双架构 clean-room integration source，两个 runtime status 均为：

```text
source_integration_ready_canary_pending
```

以下字段仍保持 `null`：

```text
huatuo/hulu native_eager_canary_artifact
huatuo/hulu selected_heads_artifact
huatuo/hulu random_heads_artifact
```

因此当前 validator 的六个 blocker 精确为：

```text
huatuo:native_eager_canary_artifact_missing
huatuo:pih_selection_not_ready
huatuo:runtime_integration_not_ready
hulu:native_eager_canary_artifact_missing
hulu:pih_selection_not_ready
hulu:runtime_integration_not_ready
```

当前状态必须保持：

```text
passed = false
control_execution_ready = false
paper_native_reproduction_authorized = false
official_code_port_authorized = false
```

这表示 source-level 机械集成已闭合，但真实 backend numerical equivalence 与 dev-only
head selection 尚未建立；不能运行 scientific intervention 或形成 paper claim。

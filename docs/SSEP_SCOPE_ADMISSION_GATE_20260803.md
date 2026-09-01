# SSEP 一日 outcome-blind scope admission gate

## 决策

**NO-GO。** Shared-Scope Evidence Pooling（SSEP）在本地同任务、原生报告生成文本中没有第二个可用模型。Hulu 有足量候选；LLaVA-Med 没有一个通过冻结 parser 的候选。reference report 不是模型，Huatuo 的 binary-VQA 自由解释也不是同一 report-generation contract，均不得在看到结果后用于补齐第二模型。

本 gate 没有读取图像、临床 label、模型 score 或 shared evaluator，没有运行 GPU，也没有把 regex candidate 称为 scope truth。由于机械 gate 已失败，没有启动人工复核或 minimal-pair 模型实验。

## 先冻结、后统计

冻结规则位于 `anchor/corrected_sgta/ssep_scope_admission.py`：

- 正式来源仅为同一 MIMIC single-image report-generation contract 下的 Hulu 与 LLaVA-Med greedy outputs；
- 总候选数至少 100；
- 至少两个模型各有 30 个候选；
- 至少三个 finding 各有 20 个候选；
- parser–human agreement 至少 0.90，且自然性与语义等价必须由独立人工确认；
- parser 只允许输出 `candidate_only_not_human_truth`；
- reference 只提供自然语言 substrate 上界，不计为模型；
- minimal pair 仅限两个 sibling 的 negated coordination，并机械保持 ordered atoms、claim count、polarity 与 whitespace word count。

正式输入与 SHA256、脚本 SHA256、命令和完整 fingerprint 均写入 `scope_admission.json`。当前 fingerprint 为 `9695e97af975c7b8229452fe8ddde33f21583d6c0a99b236da9f48fa1dccb57d`。

## 结果

| source | kind | rows | unique texts | parser candidates | qualification |
|---|---|---:|---:|---:|---|
| Hulu MIMIC report greedy | model | 694 | 424 | 254 | count pass |
| LLaVA-Med MIMIC report greedy | model | 694 | 13 | 0 | fatal fail |
| MIMIC reference report | reference | 694 | 647 | 123 | language-only; not a model |

Hulu 的候选中，negated coordination 为 251，hedged alternative 为 3。三个达到 finding 门槛的原子是：

- pleural effusion：236 cases；
- pneumothorax：247 cases；
- consolidation：71 cases。

形式上可构造 193 个 case-level、10 个 unique-template 的两 sibling negative minimal pairs。例如：

```text
shared:       No pleural effusion or pneumothorax.
distributive: No pleural effusion; no pneumothorax.
```

两者 ordered atoms、两个 claim、negative polarity 和 whitespace word count 相同。这里仅证明**机械可构造**；自然性、scope 正确性和语义等价字段仍为空，未被代码或本 session 伪造成人工判断。

## 额外风险

Hulu 的 254 个候选只有 29 个 unique candidate sentences；最大完全相同句群为 79/254（31.1%）。最常见 sibling set 是 `pleural_effusion | pneumothorax`（157 cases），其次是 `consolidation | pleural_effusion | pneumothorax`（51 cases）。因此即使强行忽略第二模型失败，当前 substrate 也更像少数报告模板的高频重复，而不是足以支撑跨 finding、跨 realization 的语言机制。

LLaVA-Med 的 694 个输出仅有 13 个 unique texts，与已有“输出过短/模板化”的结构审计一致。把宽松字符串中的一般 `and/or` 计数、reference 句子或不同任务的 Huatuo explanations 填进模型门槛，会把 response geometry 当成跨模型机制证据；本 gate 明确禁止这种救援。

## 为什么停止

SSEP 的论文新意要求 shared scope 使一个 sibling 的视觉真值传播到另一个 sibling，并且该交互跨至少两个模型成立。当前数据只支持“一个模型经常生成协调否定”。这既不能与 NegBench/negation-scope prior work 拉开距离，也不能做跨模型 `scope × sibling truth` 交互。

因此：

- `scope_claim_authorized = false`；
- `minimal_pair_model_run_authorized = false`；
- 不做人审、不跑 GPU、不构造临床 outcome；
- 不降低每模型门槛、不把 reference 当第二模型、不追加异任务输出 post-hoc rescue。

若未来有另一个**同任务、原生、非坍缩**模型报告语料，可把它作为全新的预注册 gate 重开；不能修改本次冻结 census。

## 可复现产物

- `corrected_runs/ssep_scope_gate_v1/scope_candidate_spans.jsonl`：377 个 parser candidates，含 operator、siblings、来源和候选 minimal pair；
- `corrected_runs/ssep_scope_gate_v1/scope_admission.json`：冻结规则、来源/算子/finding/模板统计、gate 与 provenance；
- `corrected_runs/ssep_scope_gate_v1/scope_human_review_template.csv`：空白人工复核模板，不含伪造判断；
- `tests/test_ssep_scope_admission.py`：共享/分配式否定、contrastive scope、reference 不计模型、human fail-closed 等五项测试。

验证：`5 passed`；目标脚本与测试 `compileall` 通过。

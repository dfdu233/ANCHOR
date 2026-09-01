# Huatuo Clinical Presupposition：formal-600 模板吸引诊断

## 结论边界

这是一个完整、可追溯的 **exploratory surface diagnostic**，不是临床评测，
也不是新机制成立的证据。200 张 VinDr pilot 图像各自完成三种 pragmatic task，
共 600 个输出；config、direct-vs-standard conformance、source、manifest、600 个
shard、合并 JSONL 和 summary 的身份与哈希全部通过。

Template Collapse（arXiv:2605.30984）和 Pensieve 已使“重复模板/跨图减法”成为
直接碰撞。因此本结果的决策是 **独立模板主线 NO-GO**，不启动 dev 或第二模型的
模板消除实验。唯一保留的窄假设是 clinical autoregressive lock-in；它必须通过
文末四级因果门槛，不能由以下统计直接推出。

## Provenance

- generation fingerprint: `137837958c559948535428eafa787d6af64923b579eab9456f69394598691e21`
- generations SHA256: `1335d7c571d6c7759bac4aa7abc5ec985a44c893166c9151ef86113289cd4d81`
- analyzer source SHA256: `0c2a8cf2726f4c6bebaf0d673d61f0f74a536ff79ce74cbba394a00df02a5f26`
- analysis fingerprint: `a5c67bf13db02ecf51f3d497a46837122eae92f693fefdf2e323d76e661919fe`
- bootstrap: 2,000 次，单位为 image，seed `42081`（各 family/condition 使用源码中冻结的确定性 offset）
- exact command 记录在 `summary.json`；正式执行输出到空 staging directory，完整性验证后替换 canonical directory。
- 4 个聚焦测试通过；source compilation 与 diff check 通过。

## 模板与长度诊断

| condition | exact unique | exact top-1 / top-3 | 跨图重复输出 | prefix-8 top-1 / entropy | prefix-12 top-1 / entropy | >30 words |
|---|---:|---:|---:|---:|---:|---:|
| neutral | 91 | 29.0% / 41.5% | 64.0% | 80.5% / 1.186 bits | 80.0% / 1.368 bits | 67.5% |
| existential | 62 | 29.0% / 50.0% | 76.0% | 69.0% / 1.398 bits | 62.0% / 1.957 bits | 98.5% |
| negative obligation | 45 | 36.0% / 61.5% | 87.0% | 99.5% / 0.045 bits | 86.5% / 0.602 bits | 100.0% |

最明显的三个 prompt-conditioned prefix 是：

- neutral：80.0% 以 `the chest x-ray shows clear lungs, normal heart size` 开始；
- existential：62.0% 以 `the chest x-ray shows a right-sided pleural effusion` 开始；
- negative obligation：86.5% 以 `this chest x-ray shows no common abnormalities such as consolidation` 开始。

输出明显不遵守共同的“至多 30 words”要求：三组平均分别为 51.1、50.0、
45.6 words。visible-answer token 没有超过 256；existential 有一次到达生成 cap。
Huatuo raw generation-only sequence 比 visible tokens 多一个 template boundary token，
已单列审计，未误记成 visible-token 超限。

## 预声明 embedded claim families

下表只使用源码中预先声明的四个窄 literal mappings，并联接完整 R8/R9/R10 votes。
它不覆盖同义词、长距离否定、定义性提及或混合极性，不能取代 shared evaluator 或
医生真值。

| literal family | neutral | existential | negative obligation | 与 reader support 的主要关系 |
|---|---:|---:|---:|---|
| positive pleural effusion | 7.0% | 70.0% | 0.0% | neutral AUROC 0.695 [0.545, 0.838]；existential 0.475 [0.411, 0.536] |
| negative pleural effusion | 62.0% | 5.5% | 100.0% | negative obligation 在 0/3、1/3、2/3、3/3 均为 100%，故 AUROC 不可定义 |
| positive lung opacity | 3.5% | 5.0% | 0.5% | neutral AUROC 0.782 [0.565, 0.965]；existential 0.692 [0.505, 0.854]，但分别仅 7/10 个命中 |
| uncertain lung opacity | 3.0% | 0.5% | 88.0% | negative obligation disagreement-AUROC 0.456 [0.352, 0.551] |

最关键的 pattern 不是“模型完全不看图”，而是 task prompt 可以压过同一 lexical
family 内原本存在的 reader association：例如 neutral positive-effusion 随 votes
上升（support slope 95% CI `[0.073, 0.514]`），existential 则在四个 vote bins 中
分别为 71.3%、57.1%、77.8%、61.5%，没有单调证据；negative-obligation 的
negative-effusion 与 uncertain-opacity 则近乎固定出现。它支持“值得做因果定位”，
不支持“幻觉已经由 regex 证明”。

## 停止规则与唯一可继续的因果问题

只有以下四关依次通过，才允许把方向改称 clinical autoregressive lock-in：

1. 回答首 token 前，reader polarity 在 image-conditioned state 可解码，且显著优于
   same-support shuffled image 和 text-only controls。
2. 对完全相同的 prefix/continuation 做 correct-vs-view/vote-matched-swapped image
   teacher forcing；image causal NLL effect 必须随 prefix 增长在明确 token/layer
   坍缩，而 length/condition/vote-matched non-attractor 不坍缩。
3. 只在冻结的 crossing token/layer patch correct-image component，必须选择性恢复
   clinical next-token polarity margin；random/norm/text-only/pre-prefix 控制为空，
   clear supported claims 变化不超过 1pp。
4. 用独立 shared audited evaluator/医生标签证明 lock-in 与临床错误相关，patch 降低
   错误且不增加 omission、缩短回答或 blanket hedge。

任一关失败即停止；禁止在 pilot 上调 template subtraction 或生成 confirmation
manifest。完整 alternative explanations 与精确 gate 已冻结在
`causal_lock_in_gate.json`。

## 产物

- `corrected_runs/vindr_v2/clinical_template_attractor_diagnostic_v1/summary.json`
- `corrected_runs/vindr_v2/clinical_template_attractor_diagnostic_v1/text_diagnostics.jsonl`
- `corrected_runs/vindr_v2/clinical_template_attractor_diagnostic_v1/frozen_pilot_template_spec.json`
- `corrected_runs/vindr_v2/clinical_template_attractor_diagnostic_v1/causal_lock_in_gate.json`
- `corrected_runs/vindr_v2/clinical_template_attractor_diagnostic_v1/COMPLETE.json`


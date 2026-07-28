# LET 研究周报（截至 2026-07-27）

## 1. 本周目标

本周围绕 Layer Evidence Transport（LET）完成三项工作：验证完整自然语言解码是否能利用中间层医学证据；在 RULE/MIMIC-CXR 上扩大到全量样本；统一现有 training-free decoding baselines 的数据范围和指标口径。

## 2. 方法进展

LET 在每个自回归步骤读取中间层与最终层的完整词表分布：

\[
p_{\alpha,t}(v)\propto p_{L,t}(v)^{1-\alpha}p_{\ell,t}(v)^\alpha.
\]

它等价于两分布的 reverse-KL barycenter，也等价于自然参数空间的一行更新：

\[
z_{\alpha,t}=(1-\alpha)z_{L,t}+\alpha z_{\ell,t}.
\]

当前固定配置为 `layer=L-12, alpha=0.30`。方法冻结模型参数，只增加一次中间层 unembedding；不使用 Yes/No 阈值、候选重排、额外图像视图或目标域标签，最终仍生成完整句子。

## 3. 数据与评测协议

### LET native full-sentence 协议

- 模型：LLaVA-Med-7B；
- 数据：RULE/MIMIC-CXR，3,470 questions，218 patients；
- prompt：原始问题，`mistral_instruct`，不附加 `[yes, no]`；
- 解码：greedy，`max_new_tokens=32`；
- 评测：完整句子生成后由修正后的 RULE parser 判定；
- 统计：patient-cluster bootstrap 与 exact McNemar。

### 官方 mitigation baseline 协议

- 数据：共同有效子集 3,466 questions；
- prompt：RULE-style `vicuna_v1`，含二元答案约束；
- 预算：64 tokens；
- 所有表中数值均由 TP/TN/FP/FN 独立复算。

两个协议的 prompt、token budget 和样本数不同，因此目前只能分块报告，不能将 LET 82.54% 与官方 greedy 75.56% 直接相减。

## 4. LET 全量结果

| Method | Acc | BAcc | TP | TN | FP | FN | Precision | Recall | F1 | Parse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native greedy | 80.12% | 77.96% | 1863 | 917 | 420 | 270 | 81.60% | 87.34% | 84.38% | 100% |
| LET | **82.54%** | **78.23%** | 2069 | 795 | 542 | 64 | 79.24% | **97.00%** | **87.23%** | 100% |

- Accuracy：+2.42pp；patient-clustered 95% CI `[+1.32,+3.56]pp`；
- Rescue/Harm：223/139；exact McNemar `p=1.18e-5`；
- F1：+2.85pp；
- 输出平均长度：14.59 → 17.18 words；
- Sensitivity：+9.66pp，但 Specificity：−9.12pp；
- Balanced Accuracy 仅 +0.27pp。

结论：LET 确实恢复了大量阳性发现，但也形成明显 affirmative bias。它支持“中间层存在可操作医学证据”，尚不能支持“临床可靠性全面提高”。

## 5. 统一后的官方 Baselines

所有方法均在同一 3,466 样本范围内：

| Method | Acc | BAcc | TP | TN | FP | FN | Prec | Rec | F1 | vs greedy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 75.56% | 74.52% | 1685 | 934 | 401 | 446 | 80.78% | 79.07% | 79.91% | — |
| beam | **79.11%** | **77.82%** | 1778 | 964 | 371 | 353 | **82.74%** | **83.44%** | **83.08%** | **+3.55pp** |
| DoLa | 75.48% | 74.45% | 1682 | 934 | 401 | 449 | 80.75% | 78.93% | 79.83% | −0.09pp |
| VCD | 69.53% | 68.79% | 1535 | 875 | 460 | 596 | 76.94% | 72.03% | 74.41% | −6.03pp |
| M3ID | 68.52% | 67.71% | 1518 | 857 | 478 | 613 | 76.05% | 71.23% | 73.56% | −7.04pp |
| OPERA | 77.61% | 76.31% | 1747 | 943 | 392 | 384 | 81.67% | 81.98% | 81.83% | +2.05pp |
| PAI | 75.68% | 74.69% | 1683 | 940 | 395 | 448 | 80.99% | 78.98% | 79.97% | +0.12pp |

观察：Beam 是官方协议下最强方法，OPERA 次之；DoLa 与 PAI 近似 greedy；VCD 与 M3ID 在医学二元 VQA 上显著退化。该结果提示，通用视觉对比解码并不会自动迁移到医学场景。

## 6. 本周非平凡结论

1. **LET 的收益真实但不均衡。** 配对统计显著，且不是 parser 可解析率造成；收益主要来自 FN→TP，而不是同时改善两类。
2. **层证据传输优于简单“早层减晚层”的叙事。** DoLa 在官方协议中无收益，而 LET 的正 KL-barycentric interpolation 在 native 协议中有效；不过需要同协议复验才能归因。
3. **通用 mitigation 存在明显医学迁移失败。** VCD、M3ID 分别下降 6.03pp、7.04pp，表明视觉先验校正方向和强度必须适配医学生成。
4. **协议本身是强变量。** 官方 greedy 75.56% 与 native greedy 80.12% 相差 4.56pp；prompt/template 的影响大于多数方法增益，任何跨协议 SOTA 比较都不可信。

## 7. 风险与完整性状态

- LET raw 包含两个由续跑范围变化产生的 fingerprint；算法配置相同，但最终论文需单 fingerprint 重跑或完成独立 integrity audit。
- 官方 baseline 为 3,466 条，LET 为 3,470 条；尚非严格共同样本集。
- LET 尚未完成有效 OE/report 全量实验；“支持开放式生成”目前是接口性质，不是实验结论。
- 当前不能声称 LET 超过 Beam、OPERA 或其它官方 baseline。

## 8. 下一周最小计划

1. 冻结共同 3,466-ID manifest，在 RULE 官方 prompt、parser、64-token 预算下运行 native greedy 与 LET；
2. 同时保留 full-sentence 文本，并报告 Acc/BAcc/Sensitivity/Specificity、患者聚类 CI 和 McNemar；
3. 若 LET 在统一协议下仍超过 greedy 且 BAcc 不退化，再与 Beam/OPERA 做正式主表；
4. 使用已修复的 OE prompt 做 paired greedy-vs-LET report pilot，primary metric 固定为 RadGraph F1；
5. 只在 source validation 上选择 class-balanced `layer/alpha`，避免针对 MIMIC test 调参。

本周状态：**LET accuracy claim 有统计证据；统一 SOTA claim 尚未成立。**

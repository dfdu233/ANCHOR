# Reviewer gate：Clinical Source-Ownership Binding Failure

**Verdict: ACCEPT WITH REVISIONS, pending the frozen discovery experiment.** 这不是 ICLR-ready
idea；当前只值得做一次决定性的验证，尚不值得扩成完整方法矩阵。

## 1. 第一印象

- **论文类型：** New Problem/Setting，而不是又一个 head suppression 方法。
- **一句话故事：** 医学 VLM 可能保留了正确的临床谓词，却在多来源生成中丢失“该事实属于哪个
  患者”的变量绑定，从而把真实但属于别人的病情绑定给当前图像。

## 2. Fatal-flaw audit

| Flaw | Severity | 当前处理 |
|---|---|---|
| Ghost Context 已覆盖 wrong-context misattributed grounding、source-blind metric 不完备和 mask-and-rerun；医学化或细粒度 mask 不新 | MAJOR | 只保留 patient-owner × clinical-predicate 的内部绑定、衰减、双向 patch 和 multimodal truth delta；任一缺失即杀 |
| 最强对照不只是 no-RAG/BM25，而是 RULE、MMed-RAG、Ghost Context、SPIN/TAF 及整段删除 | MAJOR | 已进入 baseline/gap matrix；不能完整复现的必须 N/A，不能用近似实现冒充 |

## 3. 五维评估

| 维度 | 分数 | 依据 |
|---|---:|---|
| Higher | 5/10 | 核心机制尚无 GPU 结果；现有 +1.44pp OOF stack 不能归因给 ownership mechanism |
| Faster | 6/10 | training-free、单次 trinary readout 和条件 edge intervention 有低开销路径，但未实测完整 OE 成本 |
| Stronger | 8/10 | **机制型、未被数据确认：** 若成立，它针对正确但错患者的上下文污染，并自然预测 reader ambiguity、retriever/domain shift 下的失效边界 |
| Cheaper | 7/10 | 无训练、无新医生标注即可完成机制 gate；但无医生 review 限制最终临床 hallucination claim |
| Broader | 7/10 | 把 source monitoring 与 transformer variable binding 变成 multimodal patient-specific evidence 问题；Ghost Context 使该增量必须更窄 |

## 4. 范式潜力

- **First principles：Yes。** RAG 研究通常把“context 是否相关”当作 span 属性；这里检验相关性
  是否还必须是 `fact × owner × query target` 的关系。
- **Elephant in the room：Yes。** 检索到的病例完全真实、病种也相关，却可能不属于当前患者；
  source-blind factuality 无法发现这一安全问题。
- **Technology cycle：Partial。** 医学多模态 RAG 与可 hook 的开放 VLM 使路径级实验现在可行。
- **Hamming：Partial。** 若只在 CXR CE 上成立，不改变领域；若跨 OE/report、模型和证据源成立，
  才可能改变 medical RAG 的评测单位。

## 5. 决定性实验与停止规则

1. **Behavioral discovery：** Huatuo/Hulu 均需出现 OTHER polarity transport，且至少三类 finding
   为同方向；失败任一模型即停止“通用机制”主张。
2. **Representation/cause：** 必须观察到 owner-predicate binding 曾存在后衰减，并用双向
   activation/path patch 改变 transport；若 source 从一开始就不可解码，只能写 source blindness，
   不能写 erasure。
3. **Mitigation：** fixed-coverage OE/report 中相对 raw RAG、RULE/MMed-RAG、整段删除和
   Ghost-Context-style rerun 降低 fabrication，且 omission/长度/拒答不恶化；否则方法失败。

## 6. 资源判断

- **Compute：高风险但可执行。** 单 RTX 4090，必须使用短 margin gate 抢先杀方向，完整生成继续
  由正式 baseline 长队列承担。
- **Data：可执行。** VinDr 与 MIMIC 受控来源已具备，discovery/confirmation 和 donor/query
  均隔离。
- **Clinical review：受限。** 当前无医生，论文只可用 reader votes 与数据集真值做 CE/机制；
  OE/report 的真实临床 hallucination 率不得由自动 judge 单独定义。
- **Timeline：高风险。** 2026 相邻工作增长很快；若 discovery 或 causal gate 失败，立即切换问题，
  不用更多 benchmark 延长弱方向生命周期。

## 7. 当前授权

只授权运行 512-row discovery、预冻结分析、随后视结果决定 layer/path probe。**不授权**现在实现
完整 STAF、打开 confirmation、扩大数据集或撰写 ICLR 主张。

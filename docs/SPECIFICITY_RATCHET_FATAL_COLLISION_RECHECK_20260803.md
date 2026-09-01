# Specificity Ratchet：2024–2026 Fatal Collision Recheck

日期：2026-08-03  
范围：顶会 / 顶刊 / arXiv 与官方代码的独立复核；不运行 GPU。

## 0. 最终裁决

**裁决：`METHOD DOWNGRADE / MECHANISM-ONLY CONDITIONAL GO / CURRENT GPU NO-GO`。**

这次复核发现了一个此前边界审计没有充分提升到“致命直接邻居”级别的工作：

- [Hierarchical Selective Classification](https://openreview.net/forum?id=wzof7Y66xs)（NeurIPS 2024；[官方代码](https://github.com/shanigoren/Hierarchical-Selective-Classification)）已经正式提出：当细粒度预测不够可靠时，不完全拒答，而是沿 hierarchy 退回一个较不具体但更可靠的祖先节点，并定义 hierarchical risk/coverage 与高概率准确率约束。

因此，以下方法描述已经被占据：

> “当模型对细粒度 child 不确定时，退到 supported ancestor，从而保留有用的 parent 信息。”

Specificity Ratchet 不得再把 `nearest-supported-ancestor backoff` 本身列为算法创新。ACL 2026 的 [CEBC](https://aclanthology.org/2026.acl-long.2142/) 又占据了 evidence-bounded、training-free、minimal editing 的生成修正范式；CVPR 2026 的 [ZINA](https://arxiv.org/abs/2506.13130) 占据 hallucinated-span detection and editing；MICCAI 2026 的 [CoEV](https://arxiv.org/abs/2606.18609) 已在医学 VLM 中做 statement-level visual evidence verification 与 post-hoc correction。

复核后仍未发现被直接占据的窄机制问题是：

> 在未经 fine-grained query 诱导的 native open-ended 医学生成中，模型是否先生成或保留一个医生确认有视觉支持的 parent claim，随后在语言展开中自发跨越到一个医生确认无支持的 descendant constraint；该 crossing 是否在 decoder 深度上具有方向性，并能由 crossing-stage 的选择性因果干预消除，同时保持 claim identity、polarity、claim count 和长度？

所以论文只能定位为 **New Problem + Mechanism**。ancestor backoff 只能是机制成立后的内容保持型 readout / causal test，不再是独立贡献。若没有 native trajectory 和 selective causal rescue，整个方向应降级为 HSC + CEBC/ZINA 的医学实例化，而不是继续包装。

当前仍不允许 GPU：医生 construct admission 尚未完成；当前 70-case pack 又只有 dev/test 最多 8/5 个 repeated exact-constraint lexical blocks，低于冻结的每 split 10-block confirmatory floor。这个 substrate 即使出现有利 trace，也不能支撑广泛机制结论。

### 0.1 追加的机制级碰撞（同日、模型结果出现前）

进一步检查 2025--2026 的 layerwise 工作后，`hallucinated detail gains
late and can be steered` 也不能作为剩余新颖性：

- [The Hidden Life of Tokens / VISTA](https://arxiv.org/abs/2502.03628)
  （ICML 2025）研究生成过程中视觉信息的层间变化，并以视觉信息
  steering 减轻幻觉；
- [Inject to Heal / CEI](https://aclanthology.org/2026.findings-acl.2048/)
  （Findings ACL 2026）用 Logit Lens 报告 truthful 与 hallucinatory tokens
  的 commitment-depth gap，并做 context-embedding intervention；
- [Overthinking Causes Hallucination](https://openaccess.thecvf.com/content/CVPR2026F/html/Shoby_Overthinking_Causes_Hallucination_Tracing_Confounder_Propagation_in_Vision_Language_Models_CVPRF_2026_paper.html)
  （CVPR 2026 Findings）已经追踪 decoder layer 中反复修订 hypothesis 后
  锁定错误答案；
- [Vision-Language Introspection](https://aclanthology.org/2026.acl-long.1784/)
  （ACL 2026）已做 instance-specific causal visual-anchor steering；
- [Perceptual Hallucination](https://aclanthology.org/2026.findings-acl.1237/)
  （Findings ACL 2026）用 activation patching 说明 vision-encoder error 可在
  text decoding 中传播和放大；
- [DiVE](https://aclanthology.org/2026.acl-long.1742/)
  （ACL 2026）已动态选择 visual-rich layer、分离视觉证据并进行
  contrastive decoding。

因此当前的 `constraint token vs matched token` 曲线至多识别 **late
constraint amplification**，不能单独证明 parent-to-child crossing。唯一仍
可能开放的是：native medical OE 中，医生确认的 supported parent 已被模型
直接表达/表示且在后续保持，只有一个 image-observable added constraint 从
未承诺反转为承诺，并且 claim-lattice-selective causal rescue 在 identity、
polarity、K 和 length 固定时优于上述 generic layer steering。这个交集必须
同时观测 parent-state preservation 与 constraint-state reversal；缺任一项就
降级为既有 layerwise hallucination dynamics 的医学实例化。

---

## 1. 冻结后的唯一可保留主张

### 1.1 已不能主张的内容

以下主张全部删除：

1. 首次研究 fine-grained / attribute hallucination；
2. 首次把视觉预测组织到 ontology hierarchy；
3. 首次按 uncertainty 退回 superclass / ancestor；
4. 首次在 OE 输出中定位 hallucinated span；
5. 首次用视觉证据最小编辑 unsupported content；
6. 首次在医学报告中检测或修正 finding、location、measurement 错误；
7. fixed-K 或不删除整句本身构成方法新颖性。

### 1.2 可保留的科学对象

单位必须是一个 physician-admitted edge：

\[
c_{child}\Rightarrow c_{parent},\qquad
S(c_{parent}\mid x)\ge t_p,\qquad
S(c_{child}\mid x)<t_c.
\]

但层级关系不是充分条件，还必须有时间顺序：

```text
native OE generation
    -> supported parent is available
    -> descendant continuation gains probability late
    -> unsupported child crosses commitment boundary
```

最终答案包含错误 detail 不能证明发生 ratchet。静态 fine-grained perception error、模板共现、答案更长、prompt 请求具体属性、parser 粒度差异都可产生同样的最终文本。

### 1.3 回退操作的新定位

nearest-supported-ancestor projection 现在只承担三个角色：

1. 证明干预只移除 added constraint，而不删除 parent；
2. 排除“少说换低幻觉”；
3. 把 layerwise crossing 与可观察的临床输出变化连接起来。

它不能作为论文标题中的独立新方法，除非其触发变量是已证实的 ratchet-specific causal state，并且显著优于 HSC-style hierarchical selection、CEBC minimal edit 与 ZINA/CoEV post-hoc correction。

---

## 2. 最近十二篇与精确碰撞

| # | 工作 | 年份 / venue | 已覆盖的核心 | 与 Ratchet 的精确重叠 | 官方代码状态 | 剩余 delta |
|---:|---|---|---|---|---|---|
| 1 | [Hierarchical Selective Classification](https://openreview.net/forum?id=wzof7Y66xs) | 2024, NeurIPS | uncertainty 下沿 taxonomy 退到较不具体的节点，而非完全 reject；定义 hierarchical risk-coverage 与 calibration | **直接占据 ancestor retreat、保留 coarse information、specificity-risk trade-off 的方法原语** | [官方代码](https://github.com/shanigoren/Hierarchical-Selective-Classification) | 封闭分类，不是 native OE；没有 physician support hierarchy、生成轨迹或医学 claim identity/polarity controls |
| 2 | [Taxonomy-Aware Evaluation of Vision-Language Models](https://arxiv.org/abs/2504.05457) | 2025, CVPR | 将 unconstrained VLM text 映射到 taxonomy，用 hierarchical precision/recall 给较不具体但正确的回答部分信用 | 直接占据“开放文本应按层级而非 flat exact match 评价”和 specificity-sensitive metric | [官方代码](https://github.com/vesteinn/vlm-eval) | 只评价最终实体分类；不判视觉支持边界，也不研究 spontaneous descendant crossing |
| 3 | [Pelican](https://aclanthology.org/2024.emnlp-main.470/) | 2024, EMNLP | 将视觉 claim 分解为 first-order subclaims，以工具和 Program-of-Thought 验证并自适应修正 | claim decomposition、subclaim verification、adaptive correction 已占据 | 官方论文页未给出可核验代码仓库 | 不使用临床 entailment edge；修正不是 fixed-content ancestor backoff，也无 native layerwise transition |
| 4 | [HalLoc](https://openaccess.thecvf.com/content/CVPR2025/html/Park_HalLoc_Token-level_Localization_of_Hallucinations_for_Vision_Language_Models_CVPR_2025_paper.html) | 2025, CVPR | 150K token-level hallucination annotations，生成时并发概率检测 | token/span-level localization 已占据；不能把“定位 modifier token”当贡献 | [官方代码与数据](https://github.com/dbsltm/cvpr25_halloc) | 不表达 parent/child entailment 和支持边界；检测但不做临床内容保持回退 |
| 5 | [FactCheXcker](https://openaccess.thecvf.com/content/CVPR2025/html/Heiman_FactCheXcker_Mitigating_Measurement_Hallucinations_in_Chest_X-ray_Report_Generation_Models_CVPR_2025_paper.html) | 2025, CVPR | query-code-update 修复 CXR 报告 measurement hallucination，并尽量保持报告质量 | 医学报告的局部、模块化、内容更新式 de-hallucination 已占据 | [官方代码](https://github.com/rajpurkarlab/FactCheXcker) | 只验证 ETT measurement；无一般临床 hierarchy 或 parent→descendant 动态 |
| 6 | [Phrase-grounded Fact-checking for Automatically Generated CXR Reports](https://arxiv.org/abs/2509.21356) | 2025, MICCAI | 用 finding-location 真假扰动训练 phrase-grounded verifier，检测自动报告中的 finding 与 location 错误 | finding/attribute-location 的细粒度视觉 fact checking 已占据 | 未在论文官方入口发现训练代码；存在作者发布的模型页面，但不等同完整代码 | synthetic perturbation + final report verification；不证明 native parent 先正确后升级 |
| 7 | [ZINA](https://arxiv.org/abs/2506.13130) | 2026, CVPR | 细粒度识别 hallucinated spans、分六类错误并给出 refinement；VisionHall 有人工 span 标注 | **span detection + minimal refinement** 与 Ratchet 后处理高度重叠 | [官方项目页](https://yuiga.dev/zina/) | flat error taxonomy；没有 nearest-supported clinical ancestor、医生视觉支持或 layerwise crossing |
| 8 | [FINER](https://arxiv.org/abs/2603.17662) | 2026, CVPR | 多对象/属性/关系中，真实元素包围的错误细节易诱发幻觉；用 DPO 缓解 | “越细粒度越容易错”、attribute/relation mismatch 和 fine-grained mitigation 已占据 | [官方代码/数据/模型页](https://explainableml.github.io/finer-project/) | 错误 detail 被写入 negative query；不是 OE 模型自发从 parent 扩展到 child |
| 9 | [MedVIGIL](https://arxiv.org/abs/2605.07919) | 2026, arXiv | 四名放射科医生监督的医学 VLM broken-evidence audit，含 specificity-drop、ROI corruption 与 paired OE | 医学 specificity 操作、医生 adjudication、OE evidence audit 已占据 | [官方 harness](https://github.com/hq0709/MedVIGIL)，[数据](https://huggingface.co/datasets/jhq0709/MedVIGIL) | specificity-drop 在输入/问题侧；没有输出 ontology crossing 或 layerwise causal intervention |
| 10 | [CEBC](https://aclanthology.org/2026.acl-long.2142/) | 2026, ACL | 用 external detector 与 conformal threshold 对 unsupported object mentions 做 evidence-bounded minimal revise/suppress，并兼顾 informativeness | **evidence-bounded minimal editing、risk-quality frontier 和保持生成质量已直接占据** | ACL 官方页未列可核验公开代码 | 自然图像对象；没有医学 parent support、descendant constraint 或 native causal trigger |
| 11 | [CoEV](https://arxiv.org/abs/2606.18609) | 2026, MICCAI accepted | 对医学 VLM statement 做双向 textual factuality × visual grounding verification，并 post-hoc correction | **医学 statement-level evidence verification + correction 已直接进入同一应用区** | 论文写明接收后释放，当前无可用官方代码 | 四象限 flat statement diagnosis；摘要未定义 clinical ancestor backoff 或 decoder trajectory |
| 12 | [CounterVHD](https://arxiv.org/abs/2606.28520) | 2026, arXiv | 抽取医学实体，用 factual/counterfactual grounding confidence 与 overlap 估计 entity hallucination | 医学 finding/attribute 的 entity-level counterfactual visual support 已占据 | [官方代码与数据](https://github.com/Agentic-CliniAI/CounterVHD) | 黑盒 detector；不修正输出，不验证 supported parent→unsupported child 的时间方向 |

### 2.1 两个补充边界

- [Perceptual Taxonomy](https://arxiv.org/abs/2511.19526) 已系统评估 object、property、spatial relation 与 taxonomy reasoning，发现 property-driven structured reasoning 明显更难。故“VLM 在属性层级组合上弱”也不能作为新发现。
- [Fine-Grained AI Feedback](https://arxiv.org/abs/2404.14233) 已把 object、attribute、relationship hallucination 做细粒度检测和训练反馈。故如果 Ratchet 最后只报告 unsupported attribute 比例，它会直接退化为已有 taxonomy。

---

## 3. 最危险的三重碰撞

### 3.1 HSC：算法原语碰撞

HSC 的核心措辞就是：

```text
uncertain fine prediction
    -> retreat to a less specific node
    -> preserve useful coarse information
```

这与“small left pleural effusion → pleural effusion”在算法抽象层完全同构。医学 ontology 和 OE 文本增加应用难度，但不会自动创造方法新颖性。

因此 HSC 必须成为 formal baseline：

- 为每条 edge 构造 child confidence；
- 沿 ancestor path 应用 HSC inference rule；
- 在 dev 上校准相同风险/coverage；
- 与 fixed-K Ratchet intervention 在相同 clinical-support truth 下比较。

如果 Ratchet projection 仅仅因为使用更好的 medical verifier 而超过 HSC，贡献属于 verifier / substrate，而不是新的 decoding principle。

### 3.2 CEBC + ZINA：开放生成最小编辑碰撞

CEBC 已经明确追求：

- 先生成 base output；
- 识别无视觉证据的 mention；
- 做 minimal revise 或 suppress；
- 控制 risk，同时避免不必要的 length / lexical drift。

ZINA 又直接学习 hallucinated span 的 refinement。于是“对 OE 草稿做局部支持检查，再最小修改”已经是拥挤区域。

Ratchet 的差异不能写成编辑幅度更小，而必须写成：

1. only added constraint is targeted；
2. parent is independently supported；
3. child logically entails parent plus one admitted constraint；
4. trigger is a causally localized native crossing state，而不是 external-detector threshold；
5. supported descendants 是必须不受伤害的 matched control。

### 3.3 CoEV + CounterVHD：医学视觉支持碰撞

CoEV 已把医学 statements 放进 textual factuality × visual grounding 四象限并做 correction；CounterVHD 已对医学实体做 counterfactual grounding uncertainty。这意味着：

- “每个医学 claim 独立验证视觉支持”不新；
- “利用 counter-evidence / counterfactual entity”不新；
- “post-hoc 修正医学 VLM 输出”不新。

剩余差异仅在 **hierarchical support law + native temporal transition + causal selectivity** 三者的交集。

---

## 4. 真正剩余的 mechanism delta

### 4.1 四个相邻概念必须拆开

| 概念 | 问题 | 既有工作是否覆盖 |
|---|---|---|
| Fine-grained error | child/detail 是否错 | 大量覆盖 |
| Hierarchical abstention | 不确定时是否退到 ancestor | HSC 已覆盖 |
| Evidence-bounded editing | unsupported mention 是否应最小修改 | CEBC/ZINA/CoEV 已覆盖 |
| Spontaneous specificity ratchet | supported parent 如何在 native generation 中变成 unsupported child | **本次未发现直接覆盖** |

### 4.2 机制主张的最小充分形式

对完整自然回答 \(Y\)、added-constraint token span \(T_\Delta\) 和相对位置匹配 token \(T_M\)，沿层计算：

\[
d_l(I)=\operatorname{mean}_{t\in T_\Delta}\log p_l(Y_t\mid I,q,Y_{<t})
-\operatorname{mean}_{t\in T_M}\log p_l(Y_t\mid I,q,Y_{<t}).
\]

再用至少两个同 modality/anatomy、不同病例且 visual-token length 完全相同的 swap images 得到：

\[
g_l=d_l(I_{own})-\frac{1}{K}\sum_{k=1}^{K}d_l(I_{swap,k}),\quad K\ge2.
\]

Ratchet 成立必须同时出现：

1. supported-child controls 在早期有更强 image-specific descendant evidence；
2. parent-only errors 具有更大的 late-minus-early constraint gain；
3. 该 late gain 的多数在 swap image 下仍存活，表明它不是 own-image support；
4. text-only NLL、answer length、token position、prompt request 与 lexical block 不能解释；
5. crossing-stage patch 选择性压低 unsupported child，但不压低 matched supported child。

这比 HSC 多出的不是 backoff，而是对“为什么需要 backoff”的生成动力学和因果证据。

### 4.3 不成立时的正确命名

- 若 child 从最早层就被错误偏好：`static fine-grained perception/representation error`；
- 若 own/swap/text-only 都有相同 late gain：`lexical continuation prior`；
- 若 physicians 不同意 edge：`parser granularity artifact`；
- 若只有 final unsupported detail：`fine-grained hallucination`；
- 若 HSC threshold 可完全修复：`hierarchical selective calibration`。

以上任一情况都不能称 Specificity Ratchet。

---

## 5. 三条最强 falsifier

### Falsifier 1 — 临床层级构念不成立

**检验：** 两名独立医生和 blinded adjudicator 分别判断：child 是否严格蕴含 parent + 一个约束、parent 是否受图像支持、child 是否 supported/refuted/undetermined、证据源是否可由当前图像观察。

**杀死条件：**

- edge validity 或 parent/child support agreement 不能达到预注册一致性；
- 大多数候选是风格性修饰、非邻接 edge 或需要 history/pathology 的 inference；
- 错误主要是 parent 本身 fabricated；
- 当前或新 substrate 无法在 dev/test 各达到 ≥10 个 repeated semantic-constraint blocks，并覆盖 supported-child 与 parent-only 两种角色。

**为什么最致命：** 没有稳定 clinical partial order，就不存在“祖先”“后代”或“越界”，后续所有 hidden-state 结果只是 parser 自证。

### Falsifier 2 — 没有 native late crossing

**检验：** 完整可见 native OE answer replay；约束 span 与 matched token；own image、两张等 visual-token-length swaps、text-only；控制 NLL、position、length、prompt-request 与 edge type。

**杀死条件：**

- error 与 supported-control 在早层已经分开，且没有 error-selective late shift；
- late shift 的 case-cluster CI 包含 0；
- text-only、random swaps、length-matched additions 或 prompt-request controls 复现；
- effect 只存在于人工 teacher-forced child，而不在模型自己的 OE answer；
- 少于两个模型家族或少于三个 edge types 复现。

**为什么最致命：** 它把 Ratchet 与普通细粒度误识别、语言频率和长回答错误区分开。没有时间方向，论文问题本身消失。

### Falsifier 3 — HSC/CEBC 已解释全部收益

**检验：** 在同一 physician truth、同一 K、相同长度与 coverage 下比较：

- HSC calibrated ancestor retreat；
- CEBC-style evidence-bounded minimal edit；
- ZINA/CoEV-style flat span correction；
- random ancestor、frequency-matched ancestor、blanket hedge、delete span；
- proposed crossing-stage causal projection。

**杀死条件：**

- HSC 或 CEBC 在 unsupported-descendant rate、parent retention、Brier 与 clinical usefulness 上等效；
- causal patch 不能额外选择性保护 supported descendants；
- 收益来自删词、统一 hedge、改变 claim identity/polarity、降低 K 或缩短回答；
- supported-child 或 parent retention 损失 >1pp。

**为什么最致命：** 即使行为问题存在，若简单 hierarchical calibration 已完整解释修复，论文最多是临床应用，不是新机制方法。

---

## 6. 审计后的贡献结构

### 6.1 可保留的贡献槽

1. **New problem:** native OE 中 physician-supported parent 到 unsupported descendant 的 spontaneous crossing。
2. **Construct:** 医生 admitted support hierarchy，把 apparent detail、intrinsic finding、外部知识边界分开。
3. **Mechanism:** full-answer own/swap trajectory 显示 error-specific late crossing，而非 static perception。
4. **Causal evidence:** crossing-stage intervention 只移除 added constraint，保留 supported descendants。

### 6.2 必须降级的贡献槽

`Fixed-K nearest-supported-ancestor projection` 应写为：

> a content-preserving causal readout and mitigation protocol instantiated for the discovered mechanism

而不是：

> a novel hierarchical decoding algorithm

HSC 是方法先例，CEBC 是开放生成编辑先例。除非 Ratchet 使用一个现有方法无法获得的、经因果验证的 internal crossing trigger，并在严格匹配下显著超过它们，否则方法贡献为零。

### 6.3 最合理的标题风格

可用：

> **When Supported Findings Become Unsupported Details: A Specificity Ratchet in Medical VLM Generation**

避免：

> Hierarchical Ancestor Decoding for Medical VLM Hallucination

后者会被 HSC 与 CEBC 直接击穿。

---

## 7. 更新后的 GO / NO-GO

### 硬 GO

只有同时满足以下条件才继续：

1. physician edge/support/source admission 通过；
2. 新 substrate 满足每个 frozen split ≥10 repeated semantic-constraint blocks；
3. 自然 OE 中有足够 supported-parent/unsupported-child events，而非 prompt-injected child；
4. error-specific late crossing 在两个模型家族、三个 edge types 上复现；
5. case-cluster bootstrap CI 排除 0，且 majority late gain survives image swap；
6. crossing-stage intervention 对 unsupported child 有选择性，对 supported child 和 parent retention 损伤 ≤1pp；
7. fixed K、identity、polarity、length、coverage、hedge/refusal 均匹配；
8. 明显优于 HSC、CEBC-style minimal editing 与 flat span correction。

### 硬 NO-GO

任一项成立即停止机制论文路线：

1. 医生不能稳定承认 hierarchy/support；
2. 主要错误是 fabricated parent；
3. unsupported child 只在 CE fine-grained query 出现；
4. 没有 native late crossing；
5. text-only / length / prompt / lexical frequency 解释现象；
6. 只在一个模型或一种 laterality edge 成立；
7. causal patch 伤害 supported descendants 或 parent；
8. HSC/CEBC baseline 等效；
9. mitigation 靠删 claim、缩短、hedge 或拒答；
10. 当前 pilot pack 被误当 confirmatory substrate。

---

## 8. 当前执行决策

| 项目 | 状态 | 说明 |
|---|---|---|
| Ancestor backoff 的算法新颖性 | **被占据** | HSC NeurIPS 2024 是直接先例 |
| OE minimal evidence edit | **被占据** | CEBC、ZINA；医学中还有 CoEV |
| Fine-grained medical verification | **被占据** | CounterVHD、phrase-grounded fact checking、FactCheXcker |
| Native spontaneous parent→child trajectory | **条件开放** | 本次 12 篇中未发现同一估计对象 |
| Physician-grounded clinical support hierarchy | **条件开放** | 不是 taxonomy label hierarchy；仍待真实 admission |
| Crossing-specific causal rescue | **条件开放且未测量** | 是决定 oral 潜力的唯一负载机制 |
| 当前 GPU | **NO-GO** | human gate 未过，70-case pack confirmatory block 数不足 |

### 最终建议

Specificity Ratchet 仍可保留为当前 priority-1 的**问题/机制候选**，但应立即修改项目叙事：

1. 删除 nearest-ancestor decoding 的独立 novelty claim；
2. 把 HSC 设为必须击败的第一 baseline；
3. 把 CEBC/ZINA/CoEV 设为 minimal-edit 与 medical-correction baselines；
4. 在 physician admission 与更强 repeated-edge substrate 完成前不跑 GPU；
5. 第一个决定性实验只回答“是否存在 native late crossing”，不开发方法 zoo。

若第一个决定性实验失败，直接降级，不再通过换阈值、换 parser 或加入 ontology engineering 挽救。若它成功并通过 selective causal rescue，论文的价值来自发现一个新的 support-to-language state transition，而不是重新发明 ancestor backoff。

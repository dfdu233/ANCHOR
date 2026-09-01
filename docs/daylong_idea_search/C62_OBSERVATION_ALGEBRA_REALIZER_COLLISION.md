# C62 — Observation Algebra / Structured-State Realizer 审计

日期：2026-08-13  
资源边界：公式级审计、文献碰撞与本地证据检查；未使用 GPU，未修改 baseline 队列。

## 裁决

> **作为 training-free、一次小专家 + 一次大 VLM、开放 claim 的新缓解原语严格 NO-GO。**

最强版本是让小医学专家把图像转换成一个带解剖、finding、极性、不确定性和来源的
结构化 observation state，再让大 VLM 只做 surface realization。这个分工很干净，但存在
一个不可消除的充分性二分：

1. 若输出只能陈述结构化状态逻辑蕴含的 claim，则遗漏只能由专家状态的覆盖率决定；
2. 若允许大 VLM 超出该状态自由补全，它重新获得生成 unsupported claim 的自由；
3. 若状态对开放临床 claim 已经既 sound 又 complete，则小专家 + observation algebra
   已经解决临床识别，大 VLM 仅是 graph-to-text 表述器。

集合、measure、transport 或 provenance semiring 可以保存 atom 的组合和来源，却不能把错误
或缺失的视觉 atom 变成正确临床证据。工程接口也分别落回 structured prompting、concept
bottleneck、graph-to-text、feature adapter 或 constrained decoding。没有剩余的全新计算对象。

## 1. 冻结最强候选

令图像为 `X`，真实 image-grounded clinical claim 集为 `Y(X)`。小专家只运行一次，输出

```text
S(X) = {anatomy, observation, polarity, uncertainty, provenance}_i.
```

在 algebra `A` 中，`oplus` 合并 observation，`otimes` 组合关系或属性，`Cl(S)` 表示由
这些 atom 和合法规则能推出的 claim closure。大 VLM `R` 只负责把状态写成自然语言：

```text
report = R(S(X), question),
C(report) = report 中的临床 claim 集。
```

为了避免普通的“专家提示”，候选希望直接把 `S` 写入/替换 VLM 的计算状态，并且不使用
输出融合、post-hoc filter、mask、crop、RAG 或额外训练。

## 2. 充分性边界

### 2.1 只允许有证据的陈述

若要从结构上保证不 fabricated，realizer 必须满足 evidential closure：

```text
C(R(S,q)) subseteq Cl(S).                         (1)
```

若还要求目标 claim 不遗漏，则必须有

```text
Y(X) subseteq C(R(S,q)).                           (2)
```

由 (1)(2) 得到必要条件

```text
Y(X) subseteq Cl(S).                               (3)
```

若 `Cl(S)` 自身还必须 sound，即 `Cl(S) subseteq Y(X)`，则

```text
Cl(S(X)) = Y(X).                                   (4)
```

直白地说：**同时保证不虚构和不遗漏，要求 specialist state 对目标 claim universe 已经是
充分且正确的临床解释。** 此时临床困难已由小专家解决；大 VLM 只负责措辞。

### 2.2 数据处理不等式给出相同边界

当 realizer 不再看原图时，信息链为

```text
Y <- X -> S(X) -> report.
```

因此

```text
I(Y; report) <= I(Y; S).
```

语言实现不能恢复 `S` 已丢失的病灶、部位或属性。若重新让 VLM 同时看 `X` 来补漏，则
`report=R(X,S,q)`；但只要允许它输出 `Cl(S)` 之外的 claim，式 (1) 就不再成立。重新加一个
validator/gate 又回到 verifier-guided generation 或 constrained decoding。

### 2.3 开放 claim 使有限 ontology 更困难

固定有限 ontology 的 `S` 必然存在不可表示的部位、程度、装置、关系或新 finding。若要求
对任意开放 claim 保持完整，`S` 必须近似保留图像的全部任务相关信息：

```text
Y -- X -- S  and  I(Y;X | S)=0.
```

这样 `S` 就是一个充分视觉表征，而不再是小而可验证的 observation algebra。若 `S` 只是
稠密视觉 token，方案退回普通视觉 encoder / projector；若 `S` 是病种概率表，退回分类器
输出与 concept bottleneck。

## 3. Semiring / measure / transport 没有越过边界

### 3.1 Provenance semiring

可以令每个 atom 带一个 provenance variable，并让一个 claim 的 proof polynomial 记录它由
哪些观察组合而来。这样能保证“每个输出 claim 有一条形式证明”，但只能证明：

```text
claim is derivable from S,
```

不能证明：

```text
claim is true in X.
```

错误 specialist atom 会被 algebra 忠实传播；遗漏 atom 则永远无法被 proof system 创造。
这解决的是 provenance bookkeeping，不是视觉 hallucination。

### 3.2 Set / measure

把 specialist 输出写成有限集合、signed measure 或 evidence measure，只改变 representation。
若最终 claim 由阈值、最大化、matching 或 posterior 产生，分别退回 calibration/classification、
assignment/transport 或 expert fusion。measure 不提供 target-VLM 可理解的新共享语义。

### 3.3 Transport

把 observation mass 运输到语言 claim 上需要一个跨空间 cost/kernel。固定人工 cost 是 ontology
映射；学习 cost 是 adapter；由 text embedding 构造 cost 是 CLIP/semantic matching；最优传输
本身只是对已有 atom 做匹配，不能新增病例信息。

## 4. 任何计算状态接口都落回已有范式

| 将 `S` 交给 VLM 的方式 | 实际对象 | 裁决 |
|---|---|---|
| 序列化成文字 | structured / visual-evidence prompt | 已有 |
| 变成 learned/fixed embedding prefix | soft prompt、prefix、adapter | 无 training-free 共享 codebook |
| scene graph / RadGraph 输入 | graph-to-text / structured report generation | 已有 |
| 限制可生成 claim 集 | constrained decoding / evidential closure | 排除项 |
| 替换 visual/projector tokens | concept bottleneck / feature adapter | 需要对齐训练，否则 OOD state |
| 生成后检查 closure | verifier / prune / rerank | 输出后处理 |

“替换计算状态而不是堆模块”本身不能消除这张表。状态必须与冻结 VLM 的内部坐标对齐；如果
没有训练或已有共享 codebook，直接替换只是 OOD activation。若使用目标 VLM 原本理解的文字、
对象名或视觉 token，分别回到 prompt 或原生 feature fusion。

## 5. 直接文献碰撞

1. **Concept Bottleneck Models, ICML 2020**：先预测高层概念，再由概念生成任务输出，已经
   定义 `X -> S -> Y` 的基本结构和可干预中间状态。
   <https://proceedings.mlr.press/v119/koh20a.html>
2. **Style-Aware Radiology Report Generation, Findings EMNLP 2023**：明确把报告拆成
   image-to-content 和 RadGraph-to-language realization；“内容与措辞分离”是其核心。
   <https://aclanthology.org/2023.findings-emnlp.977/>
3. **Why LLMs Hallucinate / Evidential Closure, EMNLP 2023**：已经要求输出与已有证据闭包，
   并指出感知、字符串到世界的映射和同义实现三部分缺一不可。
   <https://aclanthology.org/2023.emnlp-main.192/>
4. **Visual Evidence Prompting, ACL 2025**：小视觉专家把对象、属性与关系符号化，交给冻结
   LVLM 生成答案，直接覆盖“specialist observes, generalist verbalizes”的 prompt 分支。
   <https://aclanthology.org/2025.acl-long.205/>
5. **Automated Structured Radiology Report Generation, ACL 2025**：系统化定义 structured
   radiology reporting，进一步削弱“先结构、后表述”作为新 setting 的空间。
   <https://aclanthology.org/2025.acl-long.1301/>
6. **DDGIP, Findings NAACL 2025**：disease description graph + informed prompting，已经把
   graph content planning 与 report realization 结合。
   <https://aclanthology.org/2025.findings-naacl.215/>

这些工作未必给出上述同一个充分性边界，但已经占据所有可实现接口；边界只是解释为什么
training-free realizer 不能修复 specialist perception error，不构成新的缓解算法。

## 6. 本地证据为何不支持例外

- C46：冻结 XRV specialist 对 Huatuo 有条件增量 `+.0598`，但 Hulu 仅 `+.0102`，不通用；
- C47：one-bit veto 两模型只去除约 `17.4%` FP，同时误伤 `1.52%/2.33%` TP；
- C44：固定 K 的 specialist claim exchange 在 Huatuo/Hulu 反而使总错误增加 `3.35%/19.43%`；
- C54：把 specialist map 编入输入需要 target VLM 共享 codebook；可解释 carrier 退回视觉提示，
  隐藏 carrier 无 training-free 语义；
- C57：specialist commit/rollback 代数上是 verifier-guided exponential tilt，缓存 L0 为零触发。

所以当前没有一个“expert state 已足够正确，只差语言层写坏”的跨模型正信号。即使结构化
realizer 在 Huatuo 上改善，也更可能是 expert 直接接管一个较弱模型，而非通用计算原语。

## 7. 最终 fail-closed 判定

| 门 | 结果 |
|---|---|
| observation algebra 能否创造 specialist 缺失的病例信息 | **不能** |
| 是否同时允许开放覆盖与 evidence closure | **仅当 state 已对目标 claim 充分** |
| state 替换是否天然被 frozen VLM 理解 | **不能，需训练或已有 codebook** |
| semiring/measure/transport 是否改变信息边界 | **不能，只组合/匹配已有 atom** |
| 是否有未被 prompt/CBM/graph-to-text/constrained decoding 覆盖的实现 | **没有** |
| 是否值得占用 baseline GPU | **否** |

最终结论：

```text
NO-GO AS A NOVEL MITIGATION PRIMITIVE.
The structured state is useful only if its clinical sufficiency is already solved;
otherwise language realization cannot recover missing evidence without reopening hallucination.
```

这条分支关闭后，后续候选必须改变的不只是信息载体，而应改变冻结 VLM 如何利用**已有原图
病例信息**的计算动力学，同时不能退回 layer/logit/attention reweighting。

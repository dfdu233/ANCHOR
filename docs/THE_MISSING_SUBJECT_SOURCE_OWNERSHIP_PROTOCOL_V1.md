# The Missing Subject：医学 VLM 是否忘记“这个诊断属于谁”

## 核心问题

普通幻觉研究问：模型是否看见了正确病灶？本项目的新问题是：

> 模型可能仍记得“胸腔积液 = present”，却在生成过程中丢失主语“这是另一个患者”，
> 从而把一个内容正确、检索也相关的事实绑定到当前患者。

称此为 **Clinical Source-Ownership Binding Failure**。它不同于无关检索、错误知识、低视觉注意
或普通文本先验：危险信息可以完全真实、与问题同病种，只是 evidence owner 错了。

### 2026-08-10 查重后的必要收窄

TrustNLP 2026 的 **Ghost Context** 已经把“由错误 context span 支持的真实 claim”定义为
misattributed grounding，并用 mask-and-rerun 做因果归因。因此本项目**不能**把“错误来源的信息
影响回答”、source-blind metric 看不出来、或 span masking 本身声称为创新。

剩余可检验的新增量只有一个更强的机制合取：当前医学图像提供独立的 patient-specific truth；
模型内部曾分别编码 clinical predicate/polarity 与 patient owner；owner-predicate binding 在真实
生成路径上特异性衰减，而 predicate 仍保留；双向 activation/path patch 能复现和阻止跨患者状态
搬运，并最终在 fixed-coverage OE/report 中优于 Ghost Context 的 mask-and-rerun、整段删除和现有
medical RAG filtering。缺少其中任一项，本方向降为 Ghost Context 的医学应用扩展，不足以作为
ICLR 主线。

这一表述借鉴认知科学的 Source Monitoring Framework：记忆内容与其来源归属可以分离，错误来源
归因与 false memory/hallucination 相关。它不是可以声称的新理论。2026 年 Computational Reality
Monitoring 已把 source monitoring 引入 LLM，区分 parametric memory 与 retrieved context；本项目
若要新，必须进一步证明**同一 external context 内 clinical content 与 patient owner 的绑定失败**，
而不是再次说明模型分不清内部记忆和外部检索。

从表示学习看，这是 predicate-variable binding：模型需要区分
`effusion(current_patient)` 与 `effusion(other_patient)`，而不是只保留 predicate `effusion`。
ICML 2025 已研究 Transformer 在符号程序中如何学习变量绑定，因此“binding”本身也不新；医学
增量必须是 multimodal evidence owner 与临床极性在真实生成路径上的可分、衰减与幻觉因果关系。

## 最小可识别实验：Source × Polarity 四格

对同一图像、同一 finding、同一词汇和位置构造：

| source tag | positive state | negative state |
|---|---|---|
| CURRENT | current-patient + present | current-patient + absent |
| OTHER | other-patient + present | other-patient + absent |

另设 plain、unrelated-state、长度/范数匹配 control。CURRENT 是受控 source tag，不被当作
真值；target image 选择不读取 reader vote，冻结选择后才用 VinDr 0/3--3/3 votes 分层分析。
MIMIC train-only donor 与 VinDr target 跨医院、跨数据集，保证 donor/query patient-disjoint。

对每层/每 head 分解：

\[
P_h=\tfrac14[(a_{C,+}-a_{C,-})+(a_{O,+}-a_{O,-})],
\]

\[
S_h=\tfrac14[(a_{C,+}+a_{C,-})-(a_{O,+}+a_{O,-})],
\]

\[
B_h=\tfrac14[(a_{C,+}-a_{C,-})-(a_{O,+}-a_{O,-})].
\]

`P` 是一般临床极性，`S` 是来源身份，`B` 是来源与极性的绑定。论文要验证的不是“某些
head 对 positive/negative 敏感”，而是：在 donor-induced 错误中，`P` 仍存在，`S/B` 沿层或
沿路径先衰减，最终使 OTHER state 像 CURRENT evidence 一样影响答案。

## 因果验证

1. **Layerwise decay：** source 可解码性下降时 polarity 保留；必须控制 token identity、长度、
   位置、prompt paraphrase、范数和随机方向。
2. **双向 patch：** 把 CURRENT ownership component patch 到 OTHER state 应增加其可信度；把
   OTHER ownership component patch 回 CURRENT-like binding 应阻止搬运，同时 polarity 内容不变。
3. **Path specificity：** 只 patch donor-state token 到 answer query 的路径；同层 random path、
   unrelated finding、same-energy 和整 head patch 均不能复现。
4. **Reader interaction：** 真实图像越模糊，OTHER state 越容易越权；clear 3/3 与 0/3 病例不应
   被统一 prompt bias 完全解释。

## 条件方法：Source-Typed Attention Firewall（STAF）

只有四格 gate 和 causal patch 成立才实现方法。

RadGraph/小型结构抽取器只负责标记检索报告中的 `OTHER_PATIENT_STATE` spans，不判断当前图像
真值。对当前 image-grounded claim 的生成 query，只处理 donor span 的 edge contribution：

\[
c^{D\rightarrow q}_{h,t}=\sum_{j\in D}A^h_{t,j}V^h_jW^O_h,
\]

\[
\tilde c^{D\rightarrow q}_{h,t}=c^{D\rightarrow q}_{h,t}
-G_{h,t}U_hU_h^\top c^{D\rightarrow q}_{h,t}.
\]

`U_h` 是 dev-only 四格实验得到的 source-bound state subspace；`G` 只在 source-ownership
noninterference 被违反时开启。同一 head 的图像 token、知识 token、语法 token 和其他路径完全
保留。直观上不是“关掉坏 head”，而是只切断“别人的病情 → 当前患者结论”这一根连线。

CE 中 current claim 由问题给出；OE/report 采用两阶段固定协议：先生成固定预算草稿并解析 claims，
再在相同 claim 数/长度预算内执行 source-typed regenerate。禁止事后找出错误 token 再回放冒充
在线方法。

### 已确认的架构边界

Huatuo 的 eager Qwen2Attention 与 Hulu 的 SDPA Qwen3Attention 都可通过 4D additive mask
忠实干预指定 source-key columns；关闭态在微型同构模型上 logits `torch.equal`，打开目标边后
logits 会变化。正式 T1 必须在真实模型同实例验证关闭态逐步全词表/生成 token exact；Hulu 还需
显式 4D causal-mask plumbing control，避免把 SDPA kernel 切换误当方法效果。

但是“source span → 当前 answer query”硬 mask 只切断直接边，source 信息可能已先流入后续
Question tokens，再间接到达答案。因此 direct mask 目前只授权 causal probe，不授权“完整
firewall”主张。完整方法必须在所有下游 query 位置追踪并只移除 source-bound state subspace；若
做不到，方法 gate 为 NO-GO，不以直接 mask 冒充 noninterference。

## 与已有工作的边界

- SPIN（EMNLP 2025）按图像注意力动态压制整个低视觉 head；本方法必须证明 edge-level source
  binding 增量，并优于 SPIN top-k。
- ICLR 2025 Modular Attribution、V-ITI、ITI、CAA、LEACE 已覆盖坏 head 定位、正负方向和概念
  投影；这些均不能作为贡献。
- TAF（AAAI 2026）已做 token-level asymmetric attention filtering；本方法不能声称“首次精细
  attention filtering”，只剩显式患者类型、source×polarity binding、ownership decay 与
  donor→current edge causality。
- RULE/MMed-RAG、RADAR、FactMM-RAG 已研究医学 RAG 过度依赖/过滤；必须证明 relevance 完全
  匹配时仍存在 ownership failure，并保留非状态知识。
- TSD/反事实等权平均与 CAFP/Reynolds averaging 同构，已 KILL 为算法主贡献，只能作为 baseline。
- Computational Reality Monitoring / Attribution Blind Spot（2026）已研究模型内部能否区分
  parametric memory 与 retrieved context；必须以 within-context patient ownership 四格和绑定路径
  与之区分。
- Ghost Context（TrustNLP 2026）已覆盖 wrong-context misattributed grounding、source-blind metric
  不完备性、mask-and-rerun 归因和 post-hoc remediation；本文只可能贡献 multimodal patient-owner
  × clinical-predicate 的内部绑定/衰减机制及其细粒度因果路径，不能重复其问题定义。

## 竞赛与论文验收

### 竞赛提分

- CXR CE：相对 raw RAG 的 FP 至少下降 20%，FN 增加不超过 1pp；相对 plain BAcc 不劣且最好提升；
- 与 Beam、OPERA、RULE、MMed-RAG、SPIN、TAF 和普通删句/删 context 同表；
- OOF 只选择层/阈值，private patient/study holdout 只打开一次；
- 所有输出报告长度、invalid/uncertain、调用次数，不能靠统一 No 获益。

### ICLR 机制

- 四格 source×polarity interaction 在两模型、至少三 findings、两个数据域成立；
- ownership 先于 polarity 衰减的 layer/path 规律，bootstrap CI 排除 0；
- 双向 causal patch 与 edge-selective intervention 成立；
- OE/report 固定 claim 数和长度，fabrication 相对 raw RAG 下降至少 20%，omission 不增加；
- source-typed edge 方法显著优于整 head suppression、随机 edge、TAF/SPIN 和 no-RAG；
- 自然报告、controlled canonical context、held-out style/retriever 三种 substrate 一致。

## 立即淘汰条件

- CURRENT/OTHER × polarity 没有 interaction，只存在通用 polarity prompt effect；
- source identity 从未独立编码，或不随错误出现特异衰减；
- patch source component 不能改变 donor transport，或同时破坏 CURRENT clinical polarity；
- edge intervention不优于整句删除、no-RAG、SPIN/TAF或 random edge；
- natural report 的安全结构抽取覆盖太低，无法迁移 controlled 发现；
- OE/report 增益来自少生成、拒答、hedge 或 omission；
- 只能在 Huatuo/CXR binary CE 成立。

在四格 Gate 通过前，这是一条“高潜力、未确认”的新问题主线，不是完成的 ICLR 方法。

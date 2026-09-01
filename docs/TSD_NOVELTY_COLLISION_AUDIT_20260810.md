# Transportability Symmetrization Decoding：公式级新颖性碰撞审计

**日期：** 2026-08-10  
**对象：** 对同一 image/question，在原始 donor report 与 polarity-reversed twin 下分别前向，并逐 token 平均完整词表 logits：

\[
z_t^{\mathrm{TSD}}
=\frac12\left[z_t(X,R,y_{<t})+z_t(X,gR,y_{<t})\right],
\qquad g^2=\mathrm{id}.
\]

早期版本使用“匹配的阳性/阴性两份自然报告”；当前
[`TRANSPORTABILITY_SYMMETRIZATION_DECODING_V1.md`](./TRANSPORTABILITY_SYMMETRIZATION_DECODING_V1.md)
已改为**同一报告的 polarity twin**，自然 matched donors 只作生态验证。以下裁决以这个更强、
也更合理的版本为准。

## 结论先行

> **TSD 作为新 decoding algorithm 或“首次用群平均抵消 nuisance”的主贡献：KILL。**

原因不是只有“相似工作”，而是核心操作已有公式同构：CAFP 已对 binary counterfactual 输入做
两次模型查询并平均输出；Frame Averaging、Equi-Tuning 和 Reynolds Networks 已把同一操作
形式化为 group/Reynolds averaging；逐 token 平均 logits 又正是标准 logarithmic opinion pool。
把 nuisance 从 protected attribute 换成 other-patient clinical state、把 classifier score 换成
autoregressive token logits，是重要应用变化，但不是新的数学或 decoding 原理。

本轮未检索到一篇论文已经完整实现“医学图像 + 同报告临床状态反转 + 每步 logits 平均 + 固定
coverage 的 hallucination evaluation”。因此还可**条件保留一个机制问题**：

> 其他患者报告中的 patient state 是否在生成器中表现为近似奇分量，而疾病术语、解剖关系等
> 可运输知识表现为偶分量；这种奇偶分解能否跨模型、跨医院并被因果验证？

只有这个规律成立，工作才可能超出 CAFP 的医学应用。方法效果本身不能承担新颖性。

## 五个最接近的公式级方法

| 排名 | 已有工作 | 核心公式/操作 | 与 TSD 的关系 | 裁决 |
|---:|---|---|---|---|
| 1 | [CAFP: Counterfactual Averaging for Fair Predictions](https://arxiv.org/abs/2604.07009)（Knowledge-Based Systems 2026） | \(\hat f(x)=\frac12[f(x,0)+f(x,1)]\)：翻转 binary nuisance，查询 factual/counterfactual 两次，平均输出；冻结模型、无需训练 | **机制、干预、理论叙事均同构**；仅输出从分类概率变为 token logits，nuisance 从敏感属性变为 donor polarity | **直接碰撞，杀死算法级 novelty** |
| 2 | [Frame Averaging for Invariant and Equivariant Network Design](https://arxiv.org/abs/2110.03336)（ICLR 2022 Oral） | 对 transformation group/frame 上的模型输出求平均，使任意 backbone 对该变换不变/等变 | TSD 的 \(\mathbb Z_2\) 平均是最小二元素 frame/Reynolds average | **杀死“首次 group symmetrization”** |
| 3 | [Equi-Tuning](https://arxiv.org/abs/2210.06475)（AAAI 2023）与 [\(\lambda\)-Equitune](https://arxiv.org/abs/2305.09900)（NeurIPS 2023） | 平均 group-transformed inputs/features 的输出；后者因不同 transform 质量不等而学习权重 | 不仅覆盖群平均，还预先指出**等权平均可能有害**；这正是正/负 twin 质量不等时的风险 | **原理强碰撞 + 致命反例来源** |
| 4 | [Invariant and Equivariant Reynolds Networks](https://www.jmlr.org/papers/v25/22-0891.html)（JMLR 2024） | 用 Reynolds operator 对群求平均，把一般网络转为 invariant/equivariant function | TSD 所称的“保留偶分量、消除奇分量”就是二元素 Reynolds projection 的标准解释 | **杀死奇偶投影的数学 novelty** |
| 5 | [DExperts](https://aclanthology.org/2021.acl-long.522/)（ACL 2021）与 [Logarithmic Opinion Pool](https://papers.neurips.cc/paper_files/paper/1997/hash/59f51fd6937412b7e56ded1ea2470c25-Abstract.html) | 在 token 级组合 expert/anti-expert logits；一般 log pool 为 \(p\propto\prod_i p_i^{w_i}\) | TSD 的等权 logit 平均满足 \(p_{TSD}(v)\propto\sqrt{p_+(v)p_-(v)}\)，即两成员等权 geometric pool；token 级 logit algebra 也非新 | **杀死“新 logit ensemble”** |

最强的单一碰撞是 CAFP。其论文不仅给出同形公式，还使用“binary attribute flip、two model
queries、no retraining、average to remove direct dependence”的同一论证。若论文标题和贡献仍写
“Transportability Symmetrization Decoding”，审稿人可以合理地将其概括为：

> Counterfactual model averaging applied autoregressively to a polarity-edited medical retrieval context.

## 相邻问题也已拥挤

- [CF-RAG](https://openreview.net/forum?id=9U51rOnGko)（ICLR 2026；[代码](https://github.com/CF-RAG/CF-RAG)）已经用 counterfactual query、dialectical retrieval 和 evidence arbitration 区分决定性证据与相关干扰；不做等权 token-logit average，但占据“用反事实检索中和相关证据”的问题。
- [RULE](https://aclanthology.org/2024.emnlp-main.62/)（EMNLP 2024；[代码](https://github.com/richard-peng-xia/RULE)）直接研究医学多模态 RAG 对错误或过多检索上下文的过度依赖，以 calibrated retrieval 和 preference alignment 提高 factuality。
- [MMed-RAG](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a559a5a8aa5ae6682ced009ad97cdb16-Abstract-Conference.html)（ICLR 2025；[代码](https://github.com/richard-peng-xia/MMed-RAG)）已有 domain-aware retrieval、adaptive context selection 和错误/无关 context preference training。
- [FactMM-RAG](https://aclanthology.org/2025.naacl-long.28/)（NAACL 2025；[代码](https://github.com/cxcscmu/FactMM-RAG)）和 [RADAR](https://aclanthology.org/2025.acl-long.1279/)（ACL 2025；[代码](https://github.com/wjhou/Radar)）分别用 fact-aware retrieval 与 expert-gated knowledge injection 保留有效医学知识。
- [CoFE](https://arxiv.org/abs/2407.14474) 已在放射报告生成中寻找语义相近但诊断标签相反的正负病例，并用 counterfactual image patch 与 contrastive training 解耦共同解剖内容和诊断差异；它不是 training-free decoding，却与“matched positive/negative cases 揭示 patient state”高度邻近。
- [Calibrate Before Use](https://proceedings.mlr.press/v139/zhao21c.html)（ICML 2021）已通过 content-free input 估计并抵消 demonstration/prompt 引入的 label bias；因此“平衡正负 context 可抵消标签先验”也不能单独成为主张。

未发现 RULE、MMed-RAG、CF-RAG 或上述医学工作逐 token 等权平均同报告 polarity twins；这只能说明
**医学实例尚无 exact duplicate**，不能抵消 CAFP/Reynolds/log-pool 的公式同构。

## 原始 matched-donor 版本为什么数学上不成立

两份检索到的自然报告 \(R^+\) 与 \(R^-\) 即使疾病相同、长度相近，也不是同一对象的群轨道。
称为 \(\mathbb Z_2\) action 至少需要一个确定变换 \(g\)，满足：

\[
g(gR)=R,
\]

并且除目标 patient-state 外的内容保持不变。两个不同患者的报告通常同时改变解剖描述、设备、
病程、共病、措辞和报告者风格，因此

\[
R^-\neq g(R^+).
\]

它们的平均只是普通 two-context ensemble，不能宣称 Reynolds projection，也不能证明“共有知识
保留、患者状态相消”。当前改用 same-report twin 是必要修正；自然 matched donors 只能检验生态
有效性，不能定义算法或因果结论。

## 即使使用 same-report twin，仍有四个致命假设

### 1. 可逆不等于语义有效

模板把 `no effusion` 改成 `effusion` 再改回来，形式上可以满足 \(g^2=\mathrm{id}\)，但复杂报告中的
不确定性、比较、程度、侧别和多 finding 关系未必仍然临床自然。例如：

```text
原文：Small left pleural effusion cannot be excluded.
错误 twin：No small right pleural effusion is definitely present.
```

形式反转制造了矛盾，也同时改变 uncertainty、laterality 和语言自然度。此时两次前向的差异不再
只代表 patient polarity。

### 2. “奇分量相消、知识保留”只在线性分解下成立

所需假设是：

\[
z(X,K,s)=a(X,K)+s\,b(X,K),\qquad s\in\{-1,+1\}.
\]

真实 Transformer 更可能是：

\[
z(X,K,s)=a+s b+h_s,
\]

其中 \(h_s\) 是 polarity 与图像、prefix、否定句法的非线性交互。平均后仍留下
\((h_++h_-)/2\)，它既可能含 donor contamination，也可能删除当前图像的正确诊断。对称化保证的
只是对**定义良好的群轨道上的输出不变性**，不保证临床正确性或知识可运输性。

### 3. Logit 平均不是“多数投票”，而是几何交集

\[
\operatorname{softmax}\!\left(\frac{z^++z^-}{2}\right)(v)
\propto\sqrt{p^+(v)p^-(v)}.
\]

因此一个 token 只要被任一分支强烈压低，就会被整体压低。它可能保留双方都喜欢的模板词
（如 `There is`、`mild`），却压掉只由图像分支支持的稀有正确 finding。这也解释了为什么必须把
probability average、answer voting、no-RAG 和 matched-random twins 设为强基线。

### 4. 逐步局部对称不推出整段序列对称

实现时两个分支共享已经由平均分布生成的 prefix \(y_{<t}\)，并没有分别沿各自完整序列继续生成
后再做 sequence-level average。当前步的 odd logit 被抵消，不保证后续 exposure 和 claim-level
状态也抵消。若两个 context 导致不同句法路径，local logit symmetry 可能只产生不自然折中句。

## KEEP / KILL

### KILL

- TSD 作为新颖的 group averaging、counterfactual averaging 或 logit ensembling 算法；
- “首次用 \(\mathbb Z_2\) 平均去除 nuisance / 保留 invariant knowledge”；
- 以两份自然 positive/negative reports 直接声称群作用或严格 counterfactual；
- 仅凭 FP 下降就声称 patient-state 被移除；
- 用“医学场景尚未做过”替代机制新颖性。

### CONDITIONAL KEEP

- **Cross-Patient Evidence Transportability** 作为机制问题；
- same-report twin 作为一种严格操纵和诊断工具；
- TSD 作为验证机制预测的最简单 intervention baseline，而非论文标题/主贡献；
- 一个比 CAFP 更窄的经验命题：medical report state 在 autoregressive VLM 内近似形成可测、可因果
  搬运的 odd component，而 terminology/anatomy knowledge 形成 even component。

## 若继续，最低限度的生死实验

1. **Twin validity。** 冻结规则解析器；报告级检查 \(g^2(R)=R\)，claim tuple 级证明只改变预注册
   state fields；人工不可用时至少用两个独立结构化 labeler + contradiction/NLI + rule audit，三者
   不一致样本排除，不能由单一 LLM judge 定义有效性。
2. **方向与对称性。** 同图、同 prefix 计算
   \(T=E[m_+-m_-]\) 和 \(S=E[m_++m_--2m_0]\)。只有 \(T\) 按 donor polarity 定向且 \(S\) 接近
   零，才支持一阶奇偶分解；matched natural donors 必须由 same-report twin 复现。
3. **删除与随机控制。** 删除 target claim、反转无关 claim、随机交换等长 tokens、shuffle twins、
   matched but non-counterfactual donors；只有目标 state flip 产生反对称变化才算机制证据。
4. **强算法基线。** raw RAG、no-RAG、probability average、sequence voting、DExperts-style subtraction、
   contextual calibration、CF-RAG、RULE、MMed-RAG，以及 \(\lambda\)-weighted average。
5. **不能少说换分。** OE/report 固定 claim 数和长度；同时报告 fabrication、omission、invalid、
   uncertainty、拒答率和 non-polarity knowledge accuracy。
6. **跨域 gate。** 至少两种架构、两个医院/数据域、多个 finding；held-out report style 和 retriever
   上仍成立。否则只是 MIMIC 报告模板技巧。

## 最终可投稿判断

| 版本 | 新颖性判断 | 建议 |
|---|---|---|
| 两份自然 matched positive/negative reports 做 logit 平均 | 数学对象不成立，且只是普通 ensemble | **立即淘汰为主方法** |
| same-report polarity twin + 等权 token-logit average | 数学更干净，但与 CAFP + Reynolds average + log pool 同构 | **不能作为 ICLR 主贡献** |
| 上述方法在医学 benchmark 上有效 | 有应用价值，仍偏工程 | 可作 baseline/辅助方法 |
| 证明 patient-state odd / transportable-knowledge even 的跨模型规律，并以 twin/delete/patch 因果验证 | 可能有机制新颖性 | **唯一值得保留的主线** |

最窄且诚实的 novelty sentence 应改为 research question，而不是方法声明：

> We test whether cross-patient report conditioning decomposes into a polarity-odd patient-state component and a polarity-even transportable-knowledge component in medical VLMs, and whether this decomposition causally predicts when counterfactual output averaging can reduce hallucination without increasing omission.

如果 same-report twin 的 \(T\) 不稳定、\(S\) 明显非零，或效果不优于 no-RAG / probability averaging，
则连这条机制主线也应 KILL；不能继续通过调权重包装成新 decoding 方法。

## References

1. Arévalo, I. and Oliva, M. “CAFP: A Post-Processing Framework for Group Fairness via Counterfactual Model Averaging.” Knowledge-Based Systems 342, 2026.
2. Puny, O. et al. “Frame Averaging for Invariant and Equivariant Network Design.” ICLR Oral, 2022.
3. Basu, S. et al. “Equi-Tuning: Group Equivariant Fine-Tuning of Pretrained Models.” AAAI, 2023.
4. Basu, S. et al. “Efficient Equivariant Transfer Learning from Pretrained Models.” NeurIPS, 2023.
5. Sannai, A. et al. “Invariant and Equivariant Reynolds Networks.” JMLR 25(42), 2024.
6. Heskes, T. “Selecting Weighting Factors in Logarithmic Opinion Pools.” NeurIPS, 1997.
7. Liu, A. et al. “DExperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts.” ACL, 2021.
8. Xia, P. et al. “RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models.” EMNLP, 2024.
9. Xia, P. et al. “MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models.” ICLR, 2025.
10. Li, M. et al. “Contrastive Learning with Counterfactual Explanations for Radiology Report Generation.” arXiv:2407.14474, 2024.

## 检索限制

“未发现医学 exact duplicate”不等于 first-claim 证明。2026 文献增长很快，CF-RAG 的 OpenReview
页面存在访问挑战；本轮通过论文页与公开代码核验其高层方法，但提交前仍需下载 PDF 做逐公式
复核。这个限制不改变核心裁决，因为 CAFP 的等式与 TSD 已构成直接结构碰撞。

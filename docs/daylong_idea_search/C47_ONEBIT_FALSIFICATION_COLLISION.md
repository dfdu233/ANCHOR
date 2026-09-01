# C47：One-Bit Falsification 的公式级碰撞审计

日期：2026-08-13  
审计范围：外部医学视觉专家只输出一个 claim-specific 强反证 bit，禁止输出正证据、posterior、ranking 或与 VLM logit 相加；该 bit 只检查 VLM 已生成的阳性 finding。  
资源边界：只做论文与数学审计；未运行 GPU，未触碰 baseline 队列。

## 结论

> **C47 作为新的 hallucination mitigation 原语严格 NO-GO。**

它并没有形成一种区别于 expert gating 的新证据代数：

1. 对单个 finding，最终保留规则精确等于两个二分类器的 **AND / veto fusion**；
2. 对整段生成，它精确等于一个 expert potential 取值为 `0/-infinity` 的 **hard product-of-experts / constrained decoding**，只是把 CCD 的连续专家偏置量化成一位；
3. 按 `s_expert(c) < tau_c` 决定反驳，就是单边阈值检验；“强反证”若要有含义，必须控制真实阳性被 veto 的概率，因此仍需要 claim-conditional calibration；
4. “先由 VLM 提案、再由小专家淘汰明显假阳性”是标准 classifier cascade / medical CADe false-positive-reduction stage；
5. CEBC（ACL 2026）的核心风险函数与 C47 **逐项相同**，而且 CEBC 已经完成 external detector threshold、candidate selection、minimal rewrite 和最终 deterministic filter；
6. CoEV 已在医学报告中对生成的阳性 claims 使用 radiology expert 的二值 grounding indicator，再做 post-hoc rewrite；
7. 如果 C47 删除或改成否定句，它就是 filter / selective rejection，以少说或降低阳性内容获益；如果要求等量 replacement，只有负 bit 又不能识别应补回哪个真实 claim。

因此，`one-bit` 是对已有 expert score 的**信息删减**，`falsification` 是对 threshold veto 的重新命名。剩余的“只使用负证据”不是新机制，而是一个更受限的 expert gate。

## 1. 冻结候选

令 $x$ 为图像，VLM 首次生成的阳性 finding 集为

\[
S_0=\operatorname{Claims}^{+}(y_0), \qquad |S_0|=K.
\]

冻结医学视觉专家内部产生分数 $s_c(x)$，但对外只允许返回

\[
b_c(x)=\mathbf 1[s_c(x)<\tau_c],
\]

其中 $b_c=1$ 表示“强反对 claim $c$”。专家不得输出完整分数、正证据或未生成 finding 的 ranking。

最直接的输出为

\[
S_{\mathrm{filter}}
=
\{c\in S_0:b_c=0\}.
\]

为了满足本项目“不能靠少说”的目标，还希望把每个被 veto 的 claim 一换一替换，使最终仍有 $K$ 个阳性 findings。

## 2. C47 的数学对象并不新

### 2.1 单 claim：精确等于 AND / veto fusion

记 VLM 是否生成阳性 finding 为 $G_c\in\{0,1\}$，专家是否允许该 finding 为

\[
E_c=1-b_c.
\]

C47 的最终阳性决策是

\[
\widehat Y_c=G_cE_c=G_c\land E_c.
\]

这就是最普通的 unanimity / AND fusion：任一阶段给出负决策，阳性 prediction 即被 veto。

在给定真实状态 $Y_c$ 后两路决策条件独立的理想化假设下，

\[
\operatorname{FPR}_{\mathrm{C47}}
=
\operatorname{FPR}_{G}\operatorname{FPR}_{E},
\qquad
\operatorname{TPR}_{\mathrm{C47}}
=
\operatorname{TPR}_{G}\operatorname{TPR}_{E}.
\]

所以它降低 FP 的同时按乘法损失 sensitivity；这正是经典 AND rule 的已知 trade-off，不是 falsification 带来的新性质。

更一般地，设原生成有 $T$ 个 TP、$F$ 个 FP；veto 删除其中 $a$ 个 TP 和 $d$ 个 FP。新 precision 高于原 precision，当且仅当

\[
\frac{T-a}{T+F-a-d} > \frac{T}{T+F}
\quad\Longleftrightarrow\quad
dT>aF.
\]

这只是“veto 命中的 FP 必须足以补偿被误删的 TP”的选择性分类条件。`one-bit` 本身不保证该不等式成立。

### 2.2 整段生成：hard product-of-experts

定义 expert 可行性势函数

\[
\psi_c(x)=1-b_c(x)\in\{0,1\}.
\]

那么 C47 允许的生成分布可写为

\[
p_{\mathrm{C47}}(y\mid x)
\propto
p_{\mathrm{VLM}}(y\mid x)
\prod_{c\in\operatorname{Claims}^{+}(y)}\psi_c(x).
\]

取对数后，expert 对每个 claim 提供的势为

\[
\log\psi_c=
\begin{cases}
0,&b_c=0,\\
-\infty,&b_c=1.
\end{cases}
\]

因此它就是 **hard PoE / hard constrained decoding**。CCD 使用连续 expert log-odds 调整 token logits；C47 只是把连续势改成 `0/-infinity` 的硬极限。数据处理不等式也直接给出

\[
I(Y_c;b_c\mid G_c)
\le
I(Y_c;s_c\mid G_c),
\]

即一位阈值输出不会创造完整 expert score 中不存在的新信息。它可能更便宜或更保守，但不是新的信息源或融合原语。

### 2.3 它仍然是需要 operating point 的单边检验

若把“claim 存在”作为原假设，C47 实际执行

\[
H_0:Y_c=1
\quad\text{vs.}\quad
H_1:Y_c=0,
\qquad
s_c<\tau_c\Rightarrow\text{reject }H_0.
\]

这就是 one-sided threshold test。若 $s_c$ 是 likelihood ratio 的单调函数，Neyman--Pearson 理论已经给出固定 type-I error 下的最优阈值检验；若它不是 likelihood ratio，则连该最优性也没有。

要把 $b_c=1$ 称为“强反证”，至少要控制

\[
\alpha_c
=
P(b_c=1\mid Y_c=1,G_c=1),
\]

即 VLM 已生成该 claim 的条件下，专家误杀真实阳性的概率。这里必须显式条件于 $G_c=1$，因为 VLM 提案会改变病例分布。全数据上的 expert threshold 不能自动转移到 `VLM-proposed claims` 子群。

CEBC 正是用 base captioner 生成的 hallucinated mentions 来校准 detector threshold，而不是在无条件总体上定阈值。因此：

- 若 C47 用有标签 dev set 定 $\tau_c$，它就是 claim-conditional calibration；
- 若直接沿用 expert 默认 `0.5`，不能得到反证 soundness；
- 若完全不用校准，“strong”只是未经验证的形容词。

报告含多个真实阳性 claims 时，若每个 claim 的误 veto 率为 $\alpha_c$，则

\[
P(\text{至少误删一个真实 claim})
\le
\sum_{c\in S_0\cap Y}\alpha_c.
\]

要控制 report-level omission 又回到多重检验、conformal risk control 或 selective prediction；ConfLVLM 与 CEBC 已直接占据该空间。

### 2.4 系统结构：proposal cascade + false-positive rejector

C47 的控制流是

```text
VLM candidate generation -> expert hard-negative screen -> keep/reject
```

这正是 classifier cascade。Viola--Jones 的经典 cascade 中，任何一级返回负结果都会立即淘汰候选；医学 CADe 更长期使用“高 sensitivity candidate generator + 第二级 FP reduction classifier”。Roth 等人的 CADe 工作明确把第二级描述为拒绝困难 false positives 并尽量保持 sensitivity。

把 candidate 从 image region 换成 generated clinical claim，只改变应用单位，不改变 cascade 的决策结构。

## 3. CEBC 是最直接、不可回避的公式碰撞

CEBC 对生成文本中的 object mention $o$ 取得 external detector 分数 $s_o(x)$，定义

\[
R_\tau(x,y)
=
\sum_{o\in\operatorname{Mentions}(y)}
\mathbf 1[s_o(x)<\tau].
\]

这与 C47 在一段输出上的 bit 总和完全相同。CEBC 进一步：

1. 以 $R_\tau=0$ 定义 evidence-safe 输出；
2. 在 greedy、beam、sampled candidate pool 中先最小化风险，再最大化质量；
3. 定义 detector-admissible object set；
4. constrained rewrite unsupported mentions；
5. 用 deterministic final filter 删除仍低于阈值的 object names。

若 C47 用 VLM 自身质量在所有未被 veto 的候选中填满 $K$ 个位置，其优化正是

\[
S^*
=
\arg\max_{S\subseteq\mathcal C}
Q_{\mathrm{VLM}}(S)
\quad
\text{s.t.}\quad
|S|=K,\; b_c=0\ \forall c\in S.
\]

这只是 CEBC 的 detector-admissible set 加一个 fixed-cardinality constraint。把一个全局 $\tau$ 换成每病种 $\tau_c$，属于 class-conditional thresholding，不产生新的机制。

**判定：C47 与 CEBC 在 evidence bit、risk function、hard constraint 和 revise/filter 接口上是直接碰撞。**

## 4. 等量 replacement 的信息论缺口

### 4.1 固定 $K$ 时，什么交换才真正同时修正 hallucination 与 omission

令真实阳性 finding 集为 $Y$，固定内容预算下的集合错误为

\[
E(S;Y)=|S\setminus Y|+|Y\setminus S|.
\]

把已生成 claim $i\in S$ 换成未生成 claim $j\notin S$，得到

\[
S'=S\setminus\{i\}\cup\{j\}.
\]

直接计算可得

\[
E(S';Y)-E(S;Y)=2\bigl(\mathbf 1[i\in Y]-\mathbf 1[j\in Y]\bigr).
\]

因此只有

```text
i 是假阳性，并且 j 是遗漏真阳性
```

时，交换才同时减少一个 fabrication 和一个 omission。C47 的负 bit 最多帮助判断第一半，完全没有提供第二半所需的正信息。

### 4.2 Negative-only non-deletion impossibility

**命题。** 假设算法只观察原集合 $S_0$、VLM 内部偏好 $q$ 和 negative-only bits $b$，且 $b_j=0$ 仅表示“未被反驳”，不蕴含 $Y_j=1$。那么不存在一个固定-$K$ replacement rule，能对所有与这些观察一致的真实标签都保证严格降低 $E(S;Y)$。

**证明。** 设规则因 $b_i=1$ 移出 $i$，并选择某个 $b_j=0$ 的 $j$ 补回。构造两个具有完全相同 $(S_0,q,b)$ 的世界：

- 世界 A：$i\notin Y, j\in Y$，该交换减少两个集合错误；
- 世界 B：$i\notin Y, j\notin Y$，该交换只把一个 FP 换成另一个 FP，错误不变。

若允许 expert false veto，还可构造 $i\in Y,j\notin Y$，使错误增加两个。算法在三个世界看到相同输入，不能选择不同动作，故不存在统一严格改进保证。证毕。

要逃离该命题，必须增加至少一种被当前候选禁止或已关闭的信息：

- 让 expert 给 replacement 正分数或 ranking：回到 CCD / expert fusion / C44；
- 把 `b=0` 假设成“必定存在”：bit 已经成为完整二分类 expert，而非 negative-only certificate；
- 用 retrieval、第二视图或其他病例证据选 replacement：回到已关闭 RAG / C46；
- 把 claim 改成更泛的祖先概念：回到 hierarchical selective classification / C45；
- 删除、否定或 hedge 原 claim：回到 selective rejection，并违反固定阳性内容预算。

因此，**one-bit falsification 可以做删除器，但不能独立构成非删除式纠错器。**

## 5. 2024--2026 VLM 与经典邻域碰撞矩阵

| 工作/领域 | 核心对象 | 与 C47 的公式或系统关系 | 判定 |
|---|---|---|---|
| [CEBC, ACL 2026](https://aclanthology.org/2026.acl-long.2142/) | external detector score 低于阈值的 generated mentions；risk-first selection；minimal rewrite/filter | 风险函数逐项相同；C47 是 CEBC 去掉 conformal 与 positive ranking 后的受限特例 | **直接碰撞** |
| [ConfLVLM, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.576/) | 把每个 generated detail 当 hypothesis，检验后过滤 unreliable claims | 已占据 generated-claim hypothesis test、risk control 与 filter | 强碰撞 |
| [CoEV, 2026](https://arxiv.org/abs/2606.18609) | 抽取 medical positive claims；radiology expert 输出 region/confidence 与 binary grounding indicator；post-hoc rewrite | 医学中“expert 检查已生成阳性 claim 后纠正”已经存在；C47 仅删去其 counterfactual verification | **直接系统碰撞** |
| [CCD, 2025/2026](https://arxiv.org/abs/2509.23379) | expert probabilities、threshold labels、log-odds bias 与 token-level decoding | C47 不使用正分数，但数学上是 expert potential 的硬量化极限，不是不同融合代数 | 受限特例 |
| [REVERSE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5eff08bd064f0cdd92182cdf6fd06b99-Abstract-Conference.html) | generate -> verify -> retrospective resample | 已占据“检测到生成错误后不是拒答，而是回溯纠正”；C47 只替换 verifier 来源 | 系统强碰撞 |
| [Selective Classification via One-Sided Prediction, AISTATS 2021](https://proceedings.mlr.press/v130/gangrade21a.html) | class-wise one-sided risks，寻找低 FP 的最大 decision sets | “只在强反证时撤销阳性”属于 one-sided selective decision；fixed claim count 不改变 gate | 数学邻近 |
| Boolean AND / veto classifier fusion | 两个 hard decisions，阳性仅在双方通过时保留 | $\widehat Y=G\land E$ 精确相同 | **经典直接碰撞** |
| Viola--Jones cascade, CVPR 2001 | proposal 通过每级 gate；任一级 negative 即 reject | C47 是生成 claim proposal 加一个 hard-negative stage | **经典直接碰撞** |
| [Roth et al. CADe, 2015/2016](https://arxiv.org/abs/1505.03046) | 高 sensitivity candidate generation 后以第二级 classifier reject difficult FP | 医学影像里 candidate-specific FP filter 已是成熟流程 | 医学系统直接邻近 |
| Medical CAD / AI second reader | 第一读者提案，第二读者或 CAD 接受、提示、仲裁或否决 | “小专家只负责反对大模型阳性”是 second-reader operating rule，而非新 learning primitive | 临床工作流邻近 |

说明：REVERSE 使用训练过的自验证 VLM，CCD 使用连续 expert signal，因此不与 C47 每个实现细节相同；但 CEBC 已经给出了足以判死的公式级同构，CoEV 又给出医学应用级同构。

## 6. 为什么“只给负证据”不是剩余本质 delta

相对 CCD/普通 score fusion，C47 唯一真正不同的是主动丢弃 expert 的大部分输出，只保留一位 veto。这个限制有三个后果：

1. **没有新信息。** $b=T_\tau(s)$ 是 score 的确定性粗化；
2. **没有免校准。** 阈值决定 false-veto / false-positive trade-off，且必须在 VLM proposal 分布上验证；
3. **没有 replacement 能力。** 未被反驳不等于得到支持，无法选择遗漏真阳性。

若只追求轻量工程，它可以作为 `expert-veto` baseline；若追求本项目要求的“固定内容预算下直接把 claim 改对”，它缺少完成动作所需的病例级正信息。

此外，低 detector score 通常只表示 detector 没有识别到 finding，可能来自 subtle lesion、投照不可见、OOD 或图像质量。它不天然等于逻辑上的反例。把“lack of support”称为“falsification certificate”需要 soundness 证明，而 soundness 又把方法带回 calibrated hypothesis testing。

## 7. Gate 裁决

| Gate | 证据 | 判定 |
|---|---|---|
| 新病例级证据 | bit 来自已有 expert score 的阈值量化 | **FAIL** |
| 公式级新颖性 | AND/veto fusion + hard PoE + one-sided threshold test | **FAIL** |
| 2024--2026 VLM 碰撞 | CEBC 风险函数直接相同；CoEV 已做医学 generated-positive verification/rewrite | **FAIL** |
| 非校准 | `strong refutation` 必须选择并验证 $\tau_c$，且存在 proposal-selection shift | **FAIL** |
| 不靠少说 | 原生动作是 drop/negate/hedge | **FAIL** |
| 固定内容 budget 纠错 | negative-only bit 无法识别 replacement；存在不可识别命题 | **FAIL** |
| 值得 CPU/GPU 致死实验 | G2 直接碰撞已失败，实验不会恢复方法新颖性 | **NO** |

**冻结决定：C47 永久关闭为主方法，不运行 GPU、不调阈值、不以新名称重开。**

若后续 baseline 需要，可把它实现为 `hard expert veto / CEBC without conformal calibration`，但必须如实标注为 CEBC/decision-cascade 的简化消融，而不是新算法。

## 8. 核验来源

1. Mishra et al. “CEBC: Conformal Evidence-Bounded Control for Low-Hallucination Vision–Language Generation.” ACL 2026. https://aclanthology.org/2026.acl-long.2142/
2. Li et al. “Towards Statistical Factuality Guarantee for Large Vision-Language Models.” EMNLP 2025. https://aclanthology.org/2025.emnlp-main.576/
3. Zhou et al. “Hallucination Detection and Correction in Medical VLMs via Counter-Evidence Verification.” arXiv:2606.18609, 2026. https://arxiv.org/abs/2606.18609
4. Zhang et al. “CCD: Mitigating Hallucinations in Radiology MLLMs via Clinical Contrastive Decoding.” arXiv:2509.23379, 2025. https://arxiv.org/abs/2509.23379
5. Wu et al. “Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling.” NeurIPS 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/5eff08bd064f0cdd92182cdf6fd06b99-Abstract-Conference.html
6. Gangrade, Kag, and Saligrama. “Selective Classification via One-Sided Prediction.” AISTATS 2021. https://proceedings.mlr.press/v130/gangrade21a.html
7. Viola and Jones. “Rapid Object Detection using a Boosted Cascade of Simple Features.” CVPR 2001. https://www.cs.columbia.edu/~jebara/4995/papers/violaJones_CVPR2001.pdf
8. Roth et al. “Improving Computer-aided Detection using Convolutional Neural Networks and Random View Aggregation.” arXiv:1505.03046 / IEEE TMI. https://arxiv.org/abs/1505.03046
9. Veeramachaneni, Yan, Goebel, and Osadciw. “Improving Classifier Fusion Using Particle Swarm Optimization.” IEEE MCDM 2007; derives the familiar AND/OR decision-fusion error trade-offs. https://doi.org/10.1109/MCDM.2007.369427
10. Khan and Madden. “One-Class Classification: Taxonomy of Study and Review of Techniques.” arXiv:1312.0049. https://arxiv.org/abs/1312.0049

本地相关审计与实证：

- `docs/daylong_idea_search/C38_COLLISION_AUDIT.md`
- `docs/daylong_idea_search/C46_MULTIVIEW_FALSIFICATION_COLLISION.md`
- `corrected_runs/daylong_idea_search_v1/pareto_claim_exchange_headroom_v1/result.json`

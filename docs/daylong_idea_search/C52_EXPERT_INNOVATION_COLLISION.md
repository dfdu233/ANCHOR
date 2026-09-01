# C52 — “Experts Communicate Innovations, Not Opinions”：公式级碰撞审计

日期：2026-08-13  
范围：只审计数学对象、2020–2026 近邻和本地证据；**不运行 GPU，不修改 baseline**。  
裁决：**严格 NO-GO，不能作为新算法主线。** 它是 correlated classifier fusion / conditional likelihood-ratio stacking 的正确重述；`residual innovation` 在常见假设下又分别退化为 residualized stacking 或 Kalman innovation。

## 1. 候选的最强形式

令：

- `Y∈{0,1}`：claim 的真实极性；
- `M`：主 VLM 对该 claim 已经形成的完整状态，而不只是一位 Yes/No；
- `S`：冻结小医学视觉模型对同一图像和 finding 的输出；
- `c`：finding identity。

候选不把小模型 posterior 直接加到主模型上，而只传它在知道 `M` 后仍然提供的条件证据：

\[
\lambda(S;M,c)
=
\log\frac{p(S\mid Y=1,M,c)}{p(S\mid Y=0,M,c)}.
\tag{1}
\]

主模型 odds 随后更新为

\[
\log O(Y=1\mid M,S,c)
=
\log O(Y=1\mid M,c)+\lambda(S;M,c).
\tag{2}
\]

一个更简单的实现先残差化专家分数：

\[
u=S-\mathbb E[S\mid M,c],
\tag{3}
\]

再用 `u` 修正主模型。

叙事“专家只交流 innovation，而不重复表达 opinion”是合理的工程原则；问题在于式 (1)–(3) 是否形成新的算法对象。

## 2. 式 (2) 是概率链式法则，不是新融合原语

贝叶斯公式直接给出

\[
\frac{p(Y=1\mid M,S,c)}{p(Y=0\mid M,S,c)}
=
\frac{p(Y=1\mid M,c)}{p(Y=0\mid M,c)}
\frac{p(S\mid Y=1,M,c)}{p(S\mid Y=0,M,c)}.
\tag{4}
\]

这里**不需要**假设主模型与专家独立。第二项本来就是在主模型状态条件下，专家输出的 conditional likelihood ratio。因此：

> “只交流新增证据”就是对两个相关证据源做正确的 sequential Bayes update。

等价地，从联合分类器融合写起：

\[
\log O(Y\mid M,S,c)
=
\log O(Y\mid c)
+\log\frac{p(M,S\mid Y=1,c)}{p(M,S\mid Y=0,c)}.
\]

将联合密度按 `p(M|Y,c)p(S|Y,M,c)` 分解，就精确得到式 (2)。所以 conditional innovation 不是 correlated fusion 之外的新机制，只是联合相关模型的一种链式参数化。

## 3. 它和 stacking 的关系也是精确的

理想 stacking 学习

\[
g(M,S,c)=\operatorname{logit}p(Y=1\mid M,S,c).
\]

若主模型的校准 log-odds 为

\[
g_0(M,c)=\operatorname{logit}p(Y=1\mid M,c),
\]

则

\[
g(M,S,c)-g_0(M,c)=\lambda(S;M,c).
\tag{5}
\]

也就是说，式 (1) 是一个以 `g_0` 为 offset 的 conditional density-ratio stacker。它比“直接把两个 posterior 相加”更正确，因为它允许依赖；但**更正确不等于更新颖**。

如果用神经网络、样条、分箱或 logistic regression 估计 `λ`，差异只是 stacker 的函数族。如果只拟合 `aS+bM+d`，则是线性 stacking；如果加入 `S×M`，则是带交互的 stacking。

## 4. 残差 `u` 没有产生新信息

给定 `M,c` 后，式 (3) 是 `S` 的平移：

\[
S=u+\mathbb E[S\mid M,c].
\]

因此在条件均值已知时，`S↔u` 一一对应，并且

\[
I(Y;u\mid M,c)=I(Y;S\mid M,c).
\tag{6}
\]

残差化不会创造或筛选出一类新的 evidence；它只是重新参数化同一个 expert score。若再压缩、阈值化 `u`，数据处理不等式反而给出

\[
I(Y;T(u)\mid M,c)\le I(Y;S\mid M,c).
\]

在线性模型中，先对 `S` 回归掉 `M` 再使用 residual，是 Frisch–Waugh–Lovell / partial regression；在线性高斯状态空间模型中，`S-E[S|M]` 正是 Kalman innovation。两者都直接占据“只传不可由当前状态预测的部分”这一数学叙事。

更重要的是，均值不可预测不等于对标签有新增信息。完全可能有

\[
\mathbb E[u\mid M,c]=0
\quad\text{但}\quad
I(Y;u\mid M,c)=0,
\]

此时 innovation 只是噪声。反过来，新增信息也可能存在于 conditional variance 或非单调结构中，单一残差分数会遗漏它。

## 5. 2020–2026 直接近邻

| 工作 | 已覆盖对象 | 与 C52 的关系 | 裁决 |
|---|---|---|---|
| Trick & Rothkopf, *Bayesian Classifier Fusion with an Explicit Model of Correlation* (2021) | 对固定概率分类器的联合输出显式建模相关性，学习 `p(M,S|Y)` 并做 Bayes-optimal fusion | C52 的 conditional LR 是其联合密度比的 chain-rule factorization | **数学直接碰撞** |
| Gadgil, Covert & Lee, *Estimating Conditional Mutual Information for Dynamic Feature Selection* (ICLR 2024) | 把加入新特征后的最优 log-loss 改善刻画为 `I(Y;S|M)` | C52 的平均 incremental value 正是同一个 conditional-information 对象 | **价值函数直接近邻** |
| Sidheekh et al., *Credibility-Aware Multimodal Fusion Using Probabilistic Circuits* (AISTATS 2025) | 对多个相关/有噪声模态的预测分布做可靠性建模与联合概率融合 | “只利用条件新增信息”属于其更一般 joint probabilistic fusion 邻域 | **系统近邻** |
| CCD (2025) | 冻结 radiology expert 产生 structured clinical signals，并在生成时修改 MLLM token logits | C52 改进了 expert signal 的去重估计，但仍然把 scalar expert evidence 融入主模型 | **医学应用邻域已占** |
| AEGCD / CECAF (2026) | 多专家的一致性、病例可靠性和语义/空间路由决定干预 | 已正面处理相关专家不应被等权重复计数的问题 | **系统级碰撞** |

因此 C52 的真实增量最多是：在医学 VLM expert-guided decoding 中使用一个更规范的 conditional density-ratio estimator。它可作为强 stacking baseline 或消融，但不是 Oral 级新原语。

## 6. 本地结果也不支持把它升级为通用方法

本地 C46 已经直接测过“专家在主模型 final margin 之上是否有新增病例级信息”：

- Huatuo：macro AUROC `+.0598`，CI `[+.0397,+.0783]`，有显著增量；
- Hulu：仅 `+.0102`，未过 `+.02` 门，Brier CI 跨 0。

这正是 `I(Y;S|M,c)` 在弱主模型与强主模型之间不同的表现。C52 可以更好估计这种增量，却不能让 Hulu 上不存在的 conditional information 出现。

C47 进一步显示，把专家创新转成单边 veto 后，FP 降幅与 TP 误伤不能同时过预注册门。C44 显示固定 `K` 的 claim exchange 即使存在 expert oracle headroom，无标签交换仍会增错。因此 conditional innovation 也没有解决开放生成里的 replacement identity：知道草稿 claim 应降权，不等于知道应该补入哪个遗漏 claim。

## 7. 严格裁决

| 门 | 结果 |
|---|---|
| 是否正确避免重复计算相关专家意见 | 是 |
| 是否区别于 naive posterior sum / independent PoE | 是 |
| 是否区别于 joint correlated classifier fusion | **否** |
| 是否区别于 conditional density-ratio stacking | **否** |
| residual innovation 是否是新信息对象 | **否；为一一重参数化或有损压缩** |
| 是否解决 fixed-`K` OE replacement | **否** |
| 跨模型病例级增量前提 | **失败：仅 Huatuo 通过，Hulu 未过门** |
| ICLR 新方法 | **NO-GO** |

最终结论：

> **“Experts communicate innovations, not opinions”是一条好的设计格言，但不是新的算法。** 对 scalar expert output，它精确落入 conditional likelihood-ratio fusion；用 residual 表达又落入 partial regression / Kalman innovation。不得因叙事自然就放宽新颖性门。

## 8. 真正不同的最小操作

本审计还给出一个有用边界：只要算法的输入仍然只是 `(M,S)`，再复杂的更新都是 classifier fusion 的函数。要形成机制上不同的方法，必须改变模型可见的证据 sigma-algebra，而不是继续重写融合公式。

当前最小的合法候选是：

\[
Q=\text{canonical 8-bit render}(X),\qquad
R=X-\mathbb E[X\mid Q],
\]

其中 `X` 是原始 12/14/16-bit DICOM，`R` 是标准渲染真正丢弃的病例级视觉残差。将 `R` 通过一次、灰度保持的输入编码直接交给冻结 VLM，测试

\[
I(Y;R\mid Q,c)>0.
\]

这和 C52 有本质区别：C52 只重新组合两个模型对同一证据的意见；该操作尝试把**主模型原先没有观察到的原始测量信息**加入输入。它仍需接受 pseudo-RGB / multi-window 的碰撞审计以及 CPU 增量信息致死门；目前不是已成立的方法。

若这个候选也失败，正确结论不是继续发明新的 expert fusion 名称，而是寻找另一种真实新增观测。

## 9. 参考

1. Trick, S. & Rothkopf, C. A. “Bayesian Classifier Fusion with an Explicit Model of Correlation.” arXiv:2106.01770, 2021. https://arxiv.org/abs/2106.01770
2. Gadgil, S., Covert, I. C. & Lee, S.-I. “Estimating Conditional Mutual Information for Dynamic Feature Selection.” ICLR 2024. https://openreview.net/forum?id=Oju2Qu9jvn
3. Sidheekh, S. et al. “Credibility-Aware Multimodal Fusion Using Probabilistic Circuits.” AISTATS 2025. https://proceedings.mlr.press/v258/sidheekh25a.html
4. Zhang, X. et al. “CCD: Mitigating Hallucinations in Radiology MLLMs via Clinical Contrastive Decoding.” arXiv:2509.23379, 2025. https://arxiv.org/abs/2509.23379
5. “Adaptive Expert-Guided Contrastive Decoding with Multi-Expert Reliability and Spatial-Semantic Awareness for Radiology MLLMs.” OpenReview, 2026. https://openreview.net/forum?id=gsAgvQ2T8T

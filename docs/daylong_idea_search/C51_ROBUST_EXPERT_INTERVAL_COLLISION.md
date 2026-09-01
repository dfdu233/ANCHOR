# C51 — Cross-Hospital Expert Evidence Interval：公式级碰撞审计

日期：2026-08-13  
范围：只审计数学、文献和现有实现；**不运行 GPU，不修改 baseline**。  
裁决：**作为 ICLR 新方法严格 NO-GO；可保留为 conservative multi-expert baseline / 安全消融。**

## 1. 候选规则

大 VLM 先生成 clinical claim。对 finding `c`，使用分别在 NIH、PadChest、CheXpert 等来源训练的冻结小专家，得到标量分数 `s_d(x,c)`。在 finding-balanced development split 上拟合一维 logistic calibrator：

\[
e_d(x,c)=a_{d,c}s_d(x,c)+b_{d,c}.
\]

因为校准集按真值正负各半，在 logistic 模型正确且 case-control sampling 在类内随机时，posterior log-odds 近似 score-space log likelihood ratio：

\[
e_d(s)\approx
\log \frac{p(s_d=s\mid Y=1,c)}{p(s_d=s\mid Y=0,c)}.
\]

跨专家形成区间

\[
E(x,c)=[e_-(x,c),e_+(x,c)]
=
[\min_d e_d(x,c),\max_d e_d(x,c)].
\]

规则为：

\[
\widehat y=
\begin{cases}
1,& e_->0,\\
0,& e_+<0,\\
\widehat y_{\rm VLM},& 0\in E.
\end{cases}
\tag{1}
\]

已有实现位于 `anchor/corrected_sgta/screen_prior_free_expert_interval_v1.py`。本审计不把“已有代码”当作方法有效或新颖的证据。

## 2. 还原后，它精确是什么

定义第 `d` 个专家的二元决定 `h_d=1[e_d>0]`。则式 (1) 的两个干预区域恰好为

\[
A_+=\bigcap_d\{h_d=1\},\qquad
A_-=\bigcap_d\{h_d=0\}.
\]

因此：

> **该方法在操作层面精确等于“所有专家一致时 hard override，否则回退 VLM”的 unanimity gate。**

`min/max interval` 的数值宽度没有进入最终规则；只要每个 `e_d` 经过任意保持零点和符号的单调变换，输出完全不变。它不是新的跨专家融合代数，而是 consensus voting + selective intervention。

随着专家数量增加，`A_+` 与 `A_-` 只能缩小：

\[
A_\pm^{(D+1)}\subseteq A_\pm^{(D)}.
\]

所以更“安全”的表象可能仅来自越来越少地修改 VLM，而不是更准确地识别错误病例。这必须用 matched-intervention-rate placebo 审计。

## 3. 它为何看起来像 robust Bayes

若 `e_d(x)` 真是同一观测 `x` 在源域 `d` 下的精确 log likelihood ratio，且目标域的两个类条件分布共享同一组 mixture weights，

\[
p_w(x\mid y)=\sum_d w_d p_d(x\mid y),
\]

则目标 likelihood ratio 满足

\[
L_w(x)
=\frac{\sum_d w_dp_d(x\mid1)}{\sum_d w_dp_d(x\mid0)}
=\sum_d \widetilde w_d(x)L_d(x),
\]

其中 `\widetilde w_d(x)\ge0` 且和为 1。因此

\[
\min_dL_d(x)\le L_w(x)\le\max_dL_d(x).
\]

于是所有 `L_d>1` 或所有 `L_d<1` 确实给出对该 convex-mixture ambiguity set 稳健的决策符号。这是标准 robust Bayes / imprecise probability 的 lower/upper expectation 逻辑，不是新定理。

### 关键断裂：当前实现不满足上述对象

当前 `balanced_lr` 在**同一个 VinDr development split** 上分别校准 NIH/PadChest/CheXpert 专家的输出。因此 `e_d` 是

\[
\log p_{\rm VinDr}(S_d=s\mid Y=1)
-
\log p_{\rm VinDr}(S_d=s\mid Y=0),
\]

即多个不同 score statistic 的 target-calibrated LR；它们不是 `p_d(x|1)/p_d(x|0)` 这组 source-domain image likelihood ratios。对这些异质压缩统计量取 convex hull，没有“目标医院属于源医院 convex hull”的概率语义。

两条路线都不能救回主张：

1. **用 VinDr balanced dev 校准：** 能得到实用阈值，但这是 supervised target calibration / stacking；不再是 source-prior-free DG guarantee。
2. **只减 source prevalence log-odds：** 只有专家 posterior 在各源域已校准、label semantics 一致、且 label shift 假设成立时才是 LR；TorchXRayVision discriminative logits 和异构标签映射不提供该保证。

此外，balanced logistic 只在真实 score log-density ratio 对 `s_d` 近似仿射时成立；正负各半本身不能把任意 discriminative score 神奇地变成 likelihood ratio。

## 4. “区间冲突时回退 VLM”不是 robust Bayes 推论

当 `0\in E` 时，credal/robust 分类的含义是 ambiguity set 内不同模型偏好不同动作。标准选择包括 set-valued prediction、reject option，或给定损失后的 minimax action。式 (1) 却直接回退 VLM：

\[
0\in E\Longrightarrow \widehat y=\widehat y_{\rm VLM}.
\]

VLM 并未被证明属于该 credal set，也未进入 worst-case risk，因此这个 fallback 没有 robust Bayes 保证。它只是一个工程 gate。

如果把 VLM odds 与每个 expert LR 相乘后再取最坏情况，就变成 robust product-of-experts / likelihood-ratio fusion；如果在冲突区输出 uncertain，则变成 selective classification；如果学习 fallback 权重，则变成 calibration/stacking。三条自然扩展均是成熟对象。

## 5. 与医学 hallucination 方法的碰撞

| 工作 | 已覆盖对象 | 与 C51 的关系 |
|---|---|---|
| **CCD** | 冻结 radiology expert 提供 structured findings、threshold labels 和 probabilities；在生成时修改 MLLM token logits | C51 把连续 expert bias 量化成“全体同号时 hard override”，属于更保守的 claim-level expert gate，而非新的证据来源 |
| **AEGCD / CECAF** | 多专家、case-dependent reliability、cross-expert consistency-aware fusion，再按 token semantics 与空间对齐路由 | 直接占据“多个医学专家的一致性决定何时以及多强地干预”这一系统级新意；`min/max` 是不学习权重的硬一致特例 |
| **LEAD** | 用 latent entropy / uncertainty 决定何时干预 decoding | 不与多医院专家公式等价，但已占据“仅在可靠区域触发 mitigation”的 uncertainty-gated decoding 邻域；不能把 conservative trigger 本身当贡献 |
| CCD/LEAD/AEGCD 的共同邻域 | training-free 或 post-hoc 地按外部/内部可靠度调节生成 | C51 的剩余差异只是 robust/unanimous aggregation rule |

最关键的碰撞是 AEGCD：它已经明确把 CCD 的单专家均匀融合缺陷升级为 cross-expert consistency-aware、reliability-aware integration。C51 虽然更简单，但不是机制上不同；它是把 reliability weights 退化为 `{0,1}` 的全体一致门。

## 6. 与统计学习对象的碰撞

| 领域 | 已有标准对象 | C51 的位置 |
|---|---|---|
| Platt / logistic calibration | 用开发集把任意 score 映射到 posterior log-odds | `a_ds_d+b_d` 正是 finding-specific calibration |
| case-control / prior correction | balanced sampling 改变 intercept；减 prevalence log-odds 做 label-prior correction | “balanced 后近似 LR”是标准密度比/label-shift 思路 |
| robust Bayes / credal classification | 对概率模型集合取 lower/upper posterior；只在所有模型支配同一动作时给 determinate answer | `[min e_d,max e_d]` 与 sign dominance 是该对象的二元特例 |
| classifier ensemble | unanimous voting、intersection of decision regions、consensus gating | 式 (1) 在操作上精确等价 |
| classification with reject option / selective prediction | disagreement 时拒绝或交给后续模型 | C51 不公开拒答，而是交给 VLM，属于 cascade/selective intervention |

因此“区间”“医院不变证据”“robust likelihood ratio”可以作为解释语言，但不能掩盖其核心算法就是校准后的 unanimous expert cascade。

## 7. 对开放生成目标的额外致命问题

本项目要求固定阳性 claim 数 `K`，不能靠删除来减少 hallucination。C51 对一个草稿 claim 的负共识最多能做两件事：

1. 删除该 claim：claim 数减少，违反 fixed-`K`；
2. 用某个遗漏 claim 替换：必须另外依赖正专家 ranking，立即回到 C44 fixed-`K` expert exchange / ordinary expert fusion。

仅有“该 claim 应被否决”的共识不能决定应补入哪个真实遗漏 claim。C47 已给出 fixed-`K` 不可识别性：存在相同草稿与 negative bits、但候选 replacement 真值相反的两个世界。跨专家 unanimity 并未增加 replacement identity。

所以即使 C51 在 CE 上降低 FP，它也不能自然满足本项目 OE 的“hallucination 降且 omission 不增”要求。

## 8. 本地可执行性不等于研究价值

本地已有：

- NIH、PadChest、CheXpert 三套冻结 DenseNet121 checkpoint；
- `encode_xrv_domain_experts_v1.py`；
- source prevalence 审计；
- C51 calibration/interval screen 代码。

但当前正式 `xrv_logits.npz` 只有 all-domain 单专家 `[1003,18]`，尚无三域专家 `[D,1003,18]` 完整 artifact 和 C51 confirmation 结果。即使编码成本低，公式级新颖性已经失败；不应为寻找一个好数字把它升格为主方法。

## 9. 严格裁决

| 门 | 结果 |
|---|---|
| 病例级新增视觉信息 | 有可能；多个冻结专家确实读取图像 |
| 新融合数学 | **失败：calibrated unanimity / credal dominance** |
| 相对 CCD 的机制差异 | **失败：hard expert gate** |
| 相对 AEGCD 的系统差异 | **失败：cross-expert consistency 的保守特例** |
| robust Bayes 保证 | **失败：实现中的 target-calibrated score LR 不是 source image LR；fallback VLM 也不由 minimax 推出** |
| fixed-`K` OE 纠错 | **失败：否决不提供 replacement identity** |
| ICLR 方法候选 | **NO-GO** |
| 可否保留 | **仅作 multi-expert conservative baseline / ablation** |

最终结论：

> **C51 不能作为新方法主线。** 最诚实的描述是“finding-balanced calibration 后的 multi-source XRV unanimous override”。它可能是一个很强的工程 baseline，尤其适合比较 continuous CCD、reliability-weighted AEGCD 与 hard consensus 的安全—覆盖权衡；但不能以“credal interval”重新包装为 ICLR 创新。

若仍运行，只允许以 baseline 身份报告 intervention coverage、FP/FN、matched-intervention placebo 和专家相关性；不得声称 distributionally robust 或 prior-free，除非先证明每个输入量是 coherent source-domain likelihood ratio、目标域属于预设 ambiguity set，并将 disagreement action 纳入同一风险优化。

## 10. 参考

- Zhang et al. [CCD: Mitigating Hallucinations in Radiology MLLMs via Clinical Contrastive Decoding](https://arxiv.org/abs/2509.23379), 2025/2026.
- [Adaptive Expert-Guided Contrastive Decoding with Multi-Expert Reliability and Spatial-Semantic Awareness for Radiology MLLMs](https://openreview.net/forum?id=gsAgvQ2T8T), 2026.
- Xu et al. [Thinking in Uncertainty: Mitigating Hallucinations in MLRMs with Latent Entropy-Aware Decoding](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_Thinking_in_Uncertainty_Mitigating_Hallucinations_in_MLRMs_with_Latent_Entropy-Aware_CVPR_2026_paper.pdf), CVPR 2026.
- Chow, *On Optimum Recognition Error and Reject Tradeoff*, 1970.
- Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS 2017.
- Lipton et al., *Detecting and Correcting for Label Shift with Black Box Predictors*, ICML 2018.
- Geifman & El-Yaniv, *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*, ICML 2019.


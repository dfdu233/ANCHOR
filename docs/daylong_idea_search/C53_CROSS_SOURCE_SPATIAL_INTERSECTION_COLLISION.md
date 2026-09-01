# C53 — Cross-Source Spatial Evidence Intersection：公式级碰撞审计

日期：2026-08-13  
范围：只审计公式、理论失败模式、2020–2026 近邻与本地可执行性；**不运行 GPU，不修改 baseline**。  
裁决：**严格 NO-GO。** “多个源域专家的 CAM 交集”已被胸片 Ensemble-CAM 以完全相同公式发表；接入 VLM 后又落入 CoEV / AEGCD 的空间专家门控邻域。它也没有解决 fixed-content OE 的 replacement 问题。

## 1. 候选规则

对同一 finding `c`，令分别在不同医院/数据源训练的冻结胸片专家输出：

- 分类 score `s_d(x,c)`；
- 归一化空间热图 `A_d(p;x,c)∈[0,1]`，其中 `p` 是像素或 patch。

候选假设：医院先验会使 `s_d` 不一致，但真正病灶应在空间上共同出现。因此阈值化

\[
B_d(p)=\mathbf 1[A_d(p)>\tau_d]
\]

并取空间交集

\[
B_\cap(p)=\prod_{d=1}^{D}B_d(p)
=\bigcap_{d=1}^{D}B_d(p).
\tag{1}
\]

若 `|B_∩|`、交集峰值或交集内总响应超过阈值，则允许主 VLM 保留 claim；否则抑制该 claim。

连续版本通常写成

\[
A_\cap(p)=\min_d A_d(p)
\quad\text{或}\quad
A_\Pi(p)=\prod_d A_d(p),
\tag{2}
\]

再对 `max_p A_∩(p)` 或 `∑_p A_∩(p)` 设门。

## 2. 公式发生了直接碰撞

2024 年胸片工作 **Ensemble-CAM** 已明确提出：对多个胸片分类模型生成 Grad-CAM++，再取

\[
L_{\mathrm{EnsembleCAM}}^c
=h_{m_1}^c\cap h_{m_2}^c\cap\cdots\cap h_{m_n}^c.
\tag{3}
\]

该论文的任务就是 thoracic disease localization，动机也是通过模型间交集减少无关区域和噪声。式 (3) 与 C53 的式 (1) 只差“模型由不同医院训练”这一数据来源描述；机制、操作和主张完全相同。

因此，即使源域模型带来更强的 DG 解释，C53 也只是 **source-specific Ensemble-CAM**，属于 setting delta，不是新方法。

## 3. 接入 VLM 也不能形成新的算法对象

若用 `B_∩` 决定是否保留生成 claim，整个流程是：

```text
VLM draft claim -> multiple radiology experts -> CAM intersection -> keep/reject
```

这与以下系统邻域重叠：

- **CoEV**：对生成的医学阳性 claim 使用 radiology expert 的区域、置信度和 binary grounding indicator，再重写报告；
- **AEGCD / CECAF**：多 radiology experts 的 case-dependent reliability、cross-expert consistency 和 spatial-semantic alignment 共同路由 decoding；
- **D-CAM**：跨域 weakly supervised medical segmentation 中直接学习 domain-invariant CAM；
- 普通 weakly supervised localization ensemble：把多个 CAM 聚合成更稳定的 pseudo-region。

C53 的剩余差异只是把 learned reliability / soft aggregation 换成最保守的 hard intersection。

## 4. 为什么“后验不一致但空间一致”不是充分机制

CAM 不是 `p(Y|x)` 的空间分解，也不是像素级 likelihood ratio。对于 CNN 分类器，CAM/Grad-CAM 表示局部特征对某个 logit 的归因；它可以在阴性病例上仍然高亮共同解剖结构，也可以只覆盖一个足以分类的 discriminative fragment，而不是完整病灶。

因此

\[
A_1(p)\approx A_2(p)\approx\cdots\approx A_D(p)
\]

至少有三种互不等价的解释：

1. 各专家都定位到了真病灶；
2. 各专家共享同一正常解剖 shortcut；
3. 各专家共享同一设备、边缘、体位或训练标签生成 artifact。

仅凭空间一致无法区分三者。不同医院训练也不保证错误独立：多个模型可以共享 ImageNet 初始化、DenseNet 架构、报告标签器和胸片构图偏差。

## 5. 交集天然偏向“少区域、少阳性”

加入专家后，hard intersection 单调收缩：

\[
B_\cap^{(D+1)}\subseteq B_\cap^{(D)}.
\tag{4}
\]

若每个专家在真病灶像素上的覆盖概率为 `t_d`，在非病灶像素上的误覆盖概率为 `f_d`，只有在一个通常不成立的 conditional-independence 假设下才有

\[
\Pr(B_\cap=1\mid lesion)=\prod_dt_d,
\qquad
\Pr(B_\cap=1\mid background)=\prod_df_d.
\tag{5}
\]

式 (5) 说明即使理想独立，降低假阳性也必然伴随召回连乘下降。现实中专家正相关时，背景 shortcut 的误交集不会按乘法消失；而任何一个专家漏掉低对比病灶，交集都会清零。

所以 C53 尤其不适合：

- 弥漫性肺间质病变，不存在单一紧致共同区域；
- 小结节，一个专家的低分辨率 CAM 漏检即可杀死；
- 心脏增大等全局 finding，不同模型可依赖不同边界片段；
- 多病灶病例，各专家关注不同真阳性实例，交集反而为空。

若最终 FP 下降，首要替代解释是式 (4) 导致更少 claim 通过，而不是空间交集识别了更正确的证据。

## 6. 无法满足开放生成的固定内容预算

C53 对草稿 claim `i` 得到的只是一个 keep/reject gate。若它拒绝 `i`：

1. 直接删除会减少 claim 数，违反 fixed-content 目标；
2. 若保持 `K`，必须选择 replacement claim `j`；
3. 选择 `j` 又需要对 ontology 全部 claims 运行专家排序，并退回 C44 的 expert claim exchange / 普通多专家融合。

空间交集没有给出哪一个遗漏 claim 应替换当前 hallucinated claim。它因此不能自然同时保证 hallucination 下降和 omission 不增。

## 7. 对本地资源的判断

本地有 NIH、PadChest、CheXpert 三套 TorchXRayVision DenseNet121 checkpoint，技术上可以计算 CAM。可执行性不能挽救公式新颖性：

- 运行结果好，只能说明 Ensemble-CAM gate 是一个有用 baseline；
- 运行结果差，则符合交集召回塌缩的理论风险；
- 两种结果都不能形成新方法贡献。

因此不启动新计算。若未来 baseline 需要，可如实命名为 `source-specific Ensemble-CAM gating`，并必须匹配 intervention/claim coverage。

## 8. 严格裁决

| 门 | 结果 |
|---|---|
| 病例级视觉输入 | 有；各专家读取当前图像 |
| 新空间聚合公式 | **失败：与胸片 Ensemble-CAM 的 CAM intersection 完全相同** |
| 新 DG 机制 | **失败：source-specific 只是 setting delta；D-CAM 已占 domain-invariant CAM** |
| 新 VLM 干预 | **失败：CoEV / AEGCD 已覆盖 spatial expert grounding/gating** |
| 能否特异降低 FP | 未证明；共享 shortcut 可共同定位，交集单调收缩会伪装成安全 |
| clear-case / omission 风险 | 高；任一专家漏检都会清空交集 |
| fixed-`K` replacement | **失败** |
| ICLR 新方法 | **NO-GO** |

最终结论：

> “专家可以在概率上意见不同、但在空间上共同指向病灶”是值得检查的现象；然而“取 CAM 交集再 gate VLM”已经是已发表的胸片 Ensemble-CAM 加已有医学 expert-guided correction。它不能作为新算法主线。

真正不同的空间方法至少需要一个不是 attribution overlap 的可观测量，并且必须产生 **which replacement is supported** 的正信息，而非只有 keep/reject；当前候选没有这样的对象。

## 9. 参考

1. *Toward Explainable AI in Radiology: Ensemble-CAM for Effective Thoracic Disease Localization in Chest X-ray Images Using Weak Supervised Learning.* 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11096460/
2. *D-CAM: Learning Generalizable Weakly-Supervised Medical Image Segmentation from Domain-invariant CAM.* MICCAI 2025. https://papers.miccai.org/miccai-2025/0211-Paper0830.html
3. *Adaptive Expert-Guided Contrastive Decoding with Multi-Expert Reliability and Spatial-Semantic Awareness for Radiology MLLMs.* OpenReview 2026. https://openreview.net/forum?id=gsAgvQ2T8T
4. *CoEV: Counterfactual Evidence Verification for Medical Report Hallucination Correction.* arXiv 2026. https://arxiv.org/abs/2606.18609

# Spatial Hazard / Point-Process Clinical Evidence：碰撞与关闭审计

日期：2026-08-12  
范围：文献与数学审计；不占 GPU，不修改 baseline。

## 0. 裁决

**关闭为主方法，不值得再占一个 GPU 实验。** “按面积守恒的 hazard 聚合”有很清楚的
概率解释，但其数学精确等于经典 noisy-OR / Poisson 生存概率；在现有 patch cache 上，
它的两种可实现形式又分别被 `patch mean` 和已经 NO-GO 的 `e-mixture` 覆盖。

现有 `full negative FP=8.1%`、`random crop FP=62.9%` 不支持点过程假设。更直接的解释是：
裁剪移除了帮助模型判断“没有病灶”的全局正常解剖与尺度信息，并产生 crop/resize 分布偏移。
这是 **negative context destruction**，不是“未知位置阳性事件被 coherent aggregation 修复”。

## 1. 候选公式为何自然，但不新

把 finding 的潜在病灶看成空间点过程。令区域 `A` 内的病灶强度为

\[
\Lambda(A)=\int_A \lambda(u)\,du.
\]

背景：Poisson 点过程规定区域中的事件数满足
`N(A) ~ Poisson(Lambda(A))`，所以该区域至少出现一个病灶的概率是

\[
P(N(A)>0)=1-e^{-\Lambda(A)}.
\]

若把全图切成互不重叠区域 `A_1,...,A_M`，记局部阳性概率
`q_j=1-exp(-Lambda(A_j))`，则

\[
P(\text{image positive})
=1-\prod_j(1-q_j)
=1-\exp\left(-\sum_j\Lambda(A_j)\right).
\]

由于 `sum_j Lambda(A_j)=Lambda(image)`，无论怎样继续细分区域，整图概率不变。这正是
所希望的“分割细化 coherence”。但右侧就是标准 noisy-OR；Poisson 强度对不相交集合可加、
void probability 为 `exp(-Lambda)` 也是点过程的定义性结果，不能作为新定理或 ICLR 贡献。

更重要的是，它没有自动解决候选增多问题。若把每个 patch 的未经校准概率 `q` 原样复制，
`1-(1-q)^M` 会随 `M` 快速趋近 1。只有让局部 hazard 随面积缩放，
`Lambda_j = area_j * lambda_j`，才有守恒；这要求局部分数已经是单位面积强度，而普通 VLM
margin/attention 并不是。

## 2. 为什么在本项目中被已有实验覆盖

### 2.1 常数/线性强度退化为 patch mean

若用固定单调校准 `lambda_j=f(s_j)` 将 patch score `s_j` 变成强度，则

\[
\Lambda(\text{image})=\sum_j area_j f(s_j).
\]

等面积 partition 下，它只是 `f(score)` 的空间平均；若 `f` 近似线性，就是普通
`patch mean`。现有 sparse-patch gate 已把 `final margin + patch mean + patch max + top-5%`
作为强基线，加入 multiscale scan 后 confirmation macro AUROC 只增加 `+0.00396`，
95% CI `[-0.01939,+0.02731]`，未达到 `+0.02` 门槛。

### 2.2 非线性 null 校准退化为已测 e-mixture

若先在 development negatives 上把每个位置的 score 变成局部有效 evidence
`e_j=f(p_j)`，再按面积求和，正是已经运行的 evidence-conserving `e-mixture`。
Huatuo fresh confirmation 上：

- 相对强基线 macro AUROC 仅 `+0.00158`；95% CI `[-0.05896,+0.06451]`；
- NLL 反而从 `0.6078` 变为 `0.7041`；
- 相对 final margin，16/64/576 区域的 AUROC 差约为零；
- development 按 5% FPR 定标后，confirmation FPR 漂到 `18.8%/19.6%/25.6%`。

因此，换成 hazard/noisy-OR 不会产生新的病例排序信息，只会再做一次单调映射；若重新
校准阈值，最多改变工作点，不能修复已确认的增量信息缺失。

### 2.3 full-image 与 crop 的巨大差异反驳简单阳性 hazard 叙事

全七项 reader-unanimous negative 的 62 张图上：

- full image FP 为 `8.1%`；
- same-area random crop FP 为 `62.9%`；
- selected crop FP 在最大搜索下为 `79.0%`。

随机 crop 尚未选择任何“高 hazard”位置，却已经产生 `+54.8pp` FP。这说明主要变化发生在
**观察通道**：胸片的完整轮廓、左右对称、肺野边界和尺度被裁掉，模型不再获得分布式
negative/context evidence。只对阳性病灶出现建模的 noisy-OR 无法表达这种信息损失。

可以人为引入 positive/negative 两个 signed point processes，但“正常解剖”不是一组独立
negative lesions；一旦允许全局依赖、空间相关和 crop OOD，Poisson 独立增量假设也失去
优势，最终变成另一个需监督拟合的空间分类器，而不再是简洁通用原则。

## 3. 文献碰撞

| 工作 | 已覆盖内容 | 对本候选的影响 |
|---|---|---|
| Wang, Li, Metze, *Comparing Max and Noisy-Or Pooling*, Interspeech 2018 | 明确定义 `1-prod(1-q_i)`，并证明相关的大 bag 中 noisy-OR 会因大量小概率累积而失真 | 直接覆盖聚合公式，并给出失败机制 |
| Kraus et al., *Classifying and Segmenting Microscopy Images with Deep MIL*, Bioinformatics 2016 | 在医学显微图像中比较 noisy-OR、ISR、generalized mean、LSE；报告 noisy-OR 对 outlier 敏感 | 医学 MIL 里的方法级直接碰撞 |
| Hess et al., *Object Detection as Probabilistic Set Prediction*, ECCV 2022 | 用 random finite set、Poisson multi-Bernoulli 与 proper NLL 对整组对象建模 | “把空间对象改写为点过程”不是新抽象 |
| Riedlinger et al., *Towards Reliable Detection of Empty Space*, ICLR 2026 Poster | conditional marked Poisson point process，并把 void/empty-space confidence calibration 作为中心问题 | 直接占据“点过程 + 空区域/无目标置信度”高层位置 |
| Chan & Walther, *Detection with the Scan and the Average Likelihood Ratio*, Statistica Sinica 2013 | scan/max 与 average likelihood ratio 的经典检测比较 | 未知位置边际化与 max 的理论边界已有 |
| Xu et al., *BCEA*, arXiv 2026 | 自适应 crop/zoom 后必须对完整 acquisition policy 重新校准 | 已占据 crop 后风险漂移与修复 |

这里最致命的是联合碰撞：MIL 已覆盖 noisy-OR，point-process detection 已覆盖 void
probability，average-LR 已覆盖未知位置边际化，BCEA 已覆盖 crop 后校准。把四者应用于
医学 VLM 不改变任何一个核心假设。

## 4. 是否值得一个 CPU 短实验

**作为主线，不值得。** 现有两个 CPU 实验已经构成更强的等价致死测试：线性 hazard 被
`patch mean` 覆盖，非线性校准 hazard 被 `e-mixture` 覆盖；两者都不超 final margin。

若只为附录完整性，可做一个不超过 5 分钟、不得复活主线的 sanity check：

1. development 0/3 图上按 finding 和位置，用 complementary-log-log 回归冻结
   `lambda_j=exp(a_f+b_f s_j)`；offset 固定为 `log(area_j)`；
2. 在 16/64/576 三个真 partition 上算
   `H=sum_j area_j lambda_j` 与 `P=1-exp(-H)`；
3. fresh 266 claims 上与 `patch mean`、`e-mixture`、final margin 比较 macro AUROC/NLL；
4. 只有 hazard 相对 `final + mean + max + top5 + e-mixture` 增量 AUROC `>=0.02`、
   95% CI 下界 `>0`、NLL 改善 CI 下界 `>0`、至少 5/7 findings 正向，才允许继续。

根据现有结果，该 gate 的预期是失败；而即使意外通过，也只能说明一种 nonlinear MIL
pooler 有增量，仍需解决上述直接文献碰撞。因此不建议为它新增代码或运行。

## 5. 可保留的启发，不保留的方法

可保留的一句话不是“用 Poisson/noisy-OR 缓解幻觉”，而是：

> 医学影像的阳性证据常局部稀疏，阴性证据却依赖完整解剖；任何只裁取可疑局部、再按
> existential MIL 聚合的方法，会系统性破坏这种正负证据不对称。

这句话与当前 `full vs random crop` 结果一致，可作为 local-enhancement baseline 的失败解释；
但要成为新论文命题，必须另行设计能区分 **crop OOD、尺度变化、全局解剖丢失** 的因果实验。
它不为 Spatial Hazard 方法提供继续放量的理由。

## 参考文献（均已核实）

1. Wang, Y., Li, J., Metze, F. “Comparing the Max and Noisy-Or Pooling Functions in Multiple Instance Learning for Weakly Supervised Sequence Learning Tasks.” Interspeech, 2018. <https://arxiv.org/abs/1804.01146>
2. Kraus, O. Z., Ba, J. L., Frey, B. J. “Classifying and Segmenting Microscopy Images with Deep Multiple Instance Learning.” Bioinformatics, 2016. <https://pmc.ncbi.nlm.nih.gov/articles/PMC4908336/>
3. Hess, G., Petersson, C., Svensson, L. “Object Detection as Probabilistic Set Prediction.” ECCV, 2022. <https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700545.pdf>
4. Riedlinger, T., Maag, K., Gottschalk, H. “Towards Reliable Detection of Empty Space: Conditional Marked Point Processes for Object Detection.” ICLR, 2026. <https://openreview.net/forum?id=M2KLWLHzX0>
5. Chan, H. P., Walther, G. “Detection with the Scan and the Average Likelihood Ratio.” Statistica Sinica, 2013. <https://arxiv.org/abs/1107.4344>
6. Xu et al. “Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable Vision-Language Models.” arXiv, 2026. <https://arxiv.org/abs/2606.16667>

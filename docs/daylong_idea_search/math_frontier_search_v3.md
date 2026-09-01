# 数学前沿候选 v3：只保留从现有事实自然长出的命题

日期：2026-08-12  
范围：不占 GPU，不改 baseline；候选必须同时满足“本地现象先于数学”“有非教科书待证命题”“30 分钟内可致死”“与 2024--2026 近邻明确区分”。

## 结论

严格筛选后只有 **1 个高风险方法候选 + 1 个评测/机制候选**。没有第三个合法候选。e-process、Blackwell acquisition、Le Cam/DG、普通 rare/weak、BBP/PCA、因果 attention editing 都只能解释或审计已有对象；把它们直接包成方法会再次成为 `A+B`。

ICLR 近两年获奖工作的可学习之处不是“使用更生僻的数学”：2026 年获奖工作分别依赖清楚的新概念、真实部署缺口的可扩展诊断、以及由现代计算约束自然产生的最优近似；2025 年获奖工作也强调机制深度、学习动力学和受约束编辑。对本项目的约束是：**先发现一个社区以前没有测量过、能预测错误的规律，再让数学给出最小方法；数学名词本身不产生新颖性。**

---

## 1. 首选但仍需致死：Search-Calibrated Visual Evidence

### 一句话 insight

> 局部视觉方法在许多区域中挑出“最像病灶”的响应后，又用相关的同一模型确认它；候选越多，最大噪声越像证据——这可能同时解释小病灶遗漏和局部增强带来的假阳性。

### 为什么它由本地事实自然推出

- 两模型、两份 split 上，病灶越小，最终 claim margin 越低；这是目前第一个复现的自然现象。
- 全局 pooling、答案位置早层、风格、多 render、任意 mask/搬运都没有给出 final margin 之外的可靠增量。
- 现有 mitigation 大量改变 Yes-rate，LET 提高 recall 同时制造 FP；因此“选择出的响应是不是被当成了新证据”比“再增强视觉”更贴近失败模式。

### 数学背景（不是贡献）

设局部选择分数为 \(A_1,\ldots,A_M\)，确认分数为 \(B_1,\ldots,B_M\)。即使没有病灶，若每对 \((A_j,B_j)\) 相关系数为 \(\rho>0\)，选择 \(j^*=\arg\max_j A_j\) 后，确认分数也会被抬高：

\[
\mathbb E[B_{j^*}]\approx \rho\sqrt{2\log M}.
\]

这里 \(M\) 是搜索的 claim×区域×尺度数量。极值近似、winner's curse、scan statistic 与 higher criticism 都是经典结果，不能申报为新定理。

### 真正可能新的非平凡命题

需要证明并实证验证一个 **VLM 选择--复用相变律**，而不是再次证明高斯最大值公式：

> 对一类冻结 VLM 局部干预，存在由病灶占比 \(k/N\)、有效搜索规模 \(M_{\rm eff}\) 和选择器--验证器条件相关 \(\rho_c\) 决定的边界；边界一侧局部搜索提高真病灶支持，另一侧它主要提高 negative claim 的确认分数。用开发集估计的三元组能够在未见模型、finding 和搜索方法上预测 FP/FN 交换方向。

这条命题只有在下面两点同时成立时才不是教科书换皮：

1. \(\rho_c\sqrt{2\log M_{\rm eff}}\) 能预测 **最终 claim margin/最终 FP**，而不只是内部 patch max；
2. 同一相边界能统一 SECOND/AGLA/zoom/scan 等至少三种不同 selector，而不依赖某个实现的调参。

若成立，最小方法才是：将局部响应减去由完整选择流程产生的 search tax，只在校正后仍显著时增强 claim。它比“加一个阈值”更重要的地方是：惩罚由搜索空间和选择--验证相关性决定，而非由答案置信度决定。

### <30 分钟致死门（patch cache 落盘后纯 CPU）

1. 只取七个 findings 全为 `0/3` 的图；固定嵌套的 \(M=K\times R\times S\)。
2. 对 Huatuo/Hulu 分别测 raw selected score、相关 verifier score、最终 positive margin 和最终 FP 随 \(M\) 的斜率。
3. 冻结门：两模型中，`M=最大` 相对 `M=1` 的最终 FP 至少 `+5pp`，image-bootstrap CI 下界大于 0；且开发估计的 \(\rho\sqrt{2\log M_{eff}}\) 对确认集 inflation 有显著预测力。
4. 若只见内部 max 增长、没有传到最终 margin/FP，立即关闭，不能把极值统计包装成 hallucination 论文。

### 直接碰撞

- SECOND / AGLA / CAI / VisFlow 已占“选局部再增强”；
- BCEA 已占“自适应选择后重新校准”与 Blackwell acquisition；
- CEBC / ConfLVLM 已占 conformal verifier；
- rare/weak、scan、HC、winner's curse 都是经典数学。

**唯一剩余新颖性**是跨方法的 `sparsity × search size × selector–verifier dependence` 最终错误相变律。当前状态：**有自然现象支撑，但未通过最终错误门；不是 oral-ready。**

---

## 2. 次选，仅作机制/评测：Blackwell Clinical Dominance

### 一句话 insight

> 真正更好的 hallucination mitigation 应让模型输出成为“更有信息的实验”，而不应只在某一个 Yes/No 规则下看起来更准。

### 为什么它由本地事实自然推出

同一份 LLaVA CXR 输出，仅改变合法评分规则就有 `5/10` 方法对排名反转；最明显的是 VCD 相对 VISTA 从 strict 的 `−4.58pp` 变成 official proxy 的 `+4.98pp`。此外，多种方法主要移动 Yes-rate。这说明“方法变好”与“评判工作点改变”目前没有被分开。

### 数学背景（不是贡献）

把一个方法的连续 score 视为由真值 \(Y\) 产生的统计通道 \(E\)。Blackwell order 的含义是：若方法 \(E_1\) 的输出经随机后处理可以模拟 \(E_0\)，则 \(E_1\) 对所有有界决策损失都不差。二元任务中，这与 ROC 支配密切相关；Le Cam deficiency 则量化“还差多少随机化才能模拟”。这些均是经典决策论。

### 真正可能新的非平凡命题

> 对含 `supported/refuted/undetermined`、reader disagreement 和 fixed-coverage claim set 的临床决策，构造一个可由配对有限样本估计的 **asymmetric clinical deficiency**；证明它给出所有预注册 hallucination–omission cost ratios 下风险退化的统一上界，并把任意方法改进唯一分解为 information gain 与 criterion relocation。

关键新难点不是普通 ROC 定理，而是三态、reader-valued truth、固定 claim 数和患者聚类同时存在时，如何得到有限样本可验证的偏序/上界。如果最后退化成 AUROC、ROC convex hull 或 ECE，这个候选没有论文新意。

### <30 分钟致死门（现成 baseline 输出，纯 CPU）

1. 在完整 LLaVA CXR-VisHal 的 Greedy、VCD、DoLa、OPERA、PAI、VISTA 上画 paired ROC/风险曲线，并按 image bootstrap 给 simultaneous bands。
2. 估计每个方法相对 Greedy 的双向 empirical deficiency；检查是否有方法在主要阈值范围内统一支配，还是全部互相交叉。
3. 冻结门：至少两个方法的所谓收益可被近零双向 deficiency（近似同一通道的重标定）解释，同时该量跨 strict/official/parseable 三种规则稳定预测排名反转。
4. 若 deficiency 与普通 AUROC/ROC crossing 没有额外预测力，立即降为一张审计图，不立项。

### 直接碰撞

- Blackwell/Le Cam comparison 是经典理论；BCEA 已在 acquisition 场景显式使用 Blackwell order；
- calibration、ROC dominance、selective prediction 和 2026 年“accuracy 激励 hallucination”已占问题表面；
- 本地 criterion-shift 已经是评测发现。

因此它最多成为一个 **criterion-independent mitigation audit**。除非三态 clinical deficiency 定理和跨任务经验规律都成立，否则不能作为降低幻觉的方法，更不够 oral。

---

## 3. 为什么没有第三个合法候选

| 数学入口 | 本地或文献上的致命问题 | 决定 |
|---|---|---|
| e-process / anytime evidence | 需要合法的条件 null 或 e-increment；当前 anatomy-conditional null 已 fail，CEBC/BCEA 又覆盖风险控制和顺序采集 | 关闭方法，只可作统计工具 |
| Blackwell value-of-information / optimal design | BCEA、active MRI、cost-effective diagnosis 已直接覆盖 acquire-or-answer；IU-Xray 又缺逐 view truth | 真实第二观测实验可跑，但不是新算法 |
| Le Cam + DG | 只会把已失败的 style/FedDG 换成 directional simulability；2025--2026 已有 Le Cam transfer 近邻 | 关闭 |
| random matrix / BBP | “common-mode 大 spike 掩盖 sparse clinical spike”很漂亮，但 NCD centering 已 `0.736→0.655`；BBP/稀疏 spike 边界本身经典 | 只可解释候选 1，不单独立项 |
| causal representation / head intervention | HalluTrace、CausalLens、head suppression、Ghost Context 已覆盖来源分解、因果头和错上下文；本地还没有跨模型正例 | 等证据，不先命名 |
| rare/weak / HC | 最优检测边界是经典；若没有“搜索膨胀进入最终 FP”的 VLM-specific law，就只是换 aggregator | 并入候选 1，不独立申报 |

## 最终排序与行动

1. **Search-Calibrated Visual Evidence**：唯一值得等 patch cache 后做 30 分钟致死实验的算法候选；PASS 后才做跨 SECOND/AGLA/CAI 的真实解码实验。
2. **Blackwell Clinical Dominance**：可以立即 CPU 审计，但定位是判别“证据增益还是工作点迁移”的机制/评测原则，不应冒充 mitigation。
3. **无合法第三候选。** 在这两门给出正证据之前，引入 e-value、OT、diffusion、拓扑、Fisher geometry 或 causal editing 都是装饰。

当前最诚实判断：**没有已完成的 ICLR Oral 级方法**。候选 1 有成为 Oral 级“新规律 → 简单方法”的路径；候选 2 有成为主会评测/机制论文的路径。两者都必须先过预注册门，不能用数学审美替代正结果。

## 主要核验来源

- ICLR. [2026 Outstanding Papers announcement](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/).
- ICLR. [2025 Outstanding Papers announcement](https://blog.iclr.cc/2025/04/22/announcing-the-outstanding-paper-awards-at-iclr-2025/).
- Park et al. [SECOND](https://proceedings.mlr.press/v267/park25c.html). ICML 2025.
- An et al. [AGLA](https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html). CVPR 2025.
- Mishra et al. [CEBC](https://aclanthology.org/2026.acl-long.2142/). ACL 2026.
- Xu et al. [BCEA](https://arxiv.org/abs/2606.16667). arXiv 2026; 未确认正式会议。
- Li et al. [VISTA](https://openreview.net/forum?id=7BKcLeHQsm). ICML 2025.
- Hwang et al. [Perceptual Hallucination in Vision--Language Models](https://aclanthology.org/2026.findings-acl.1237/). Findings ACL 2026.
- Chan et al. [System-Mediated Attention Imbalances Make Vision-Language Models Say Yes](https://aclanthology.org/2026.findings-acl.1940/). Findings ACL 2026.
- Mariucci. [Le Cam theory on the comparison of statistical models](https://arxiv.org/abs/1605.03301). 2016.（数学背景）
- Donoho & Jin. *Higher Criticism for Detecting Sparse Heterogeneous Mixtures*. Annals of Statistics, 2004.（数学背景）

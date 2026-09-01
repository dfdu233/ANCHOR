# CECD / PAEL 新颖性碰撞审计：从“组合等价失效”收缩到“临床方向选择机制”

**日期：** 2026-08-03  
**状态：** outcome-blind literature audit；未读取 human return、sealed confirmation outcome、模型输出或 GPU 状态。  
**冻结对象：** clinician-admitted render × wording product nonseparability，以 reader-distribution Brier loss 衡量，并减去保持 centered interaction 全部奇异值的 Haar left/right orientation reference（PAEL）。

## 摘要

本轮没有检索到与 PAEL **整套对象完全相同**的论文：尚未发现工作同时具备（i）两个轴在看到模型输出前由临床医生独立承认为等价，（ii）完整 render × wording factorial product 及两条边际，（iii）独立 multi-reader vote distribution，（iv）以 proper loss 比较实际 centered interaction 与 isospectral left/right Haar orientation reference，以及（v）在医学生成式 VLM 内进行层级和因果定位。

但这并不支持宽泛新颖性。2026 年相邻工作已经把 PAEL 的几乎每个外层概念分别占据：MetaRA 同时施加 question paraphrase 与 benign/style/background image transformation；composite metamorphic relation 文献直接研究组合关系比单项关系更能揭露 DNN failure；PSF-Med 已建立临床问题等价改写导致 Med-VLM answer flip；Semantic Robustness Certification 已用 text proxy、semantic plane、定范数旋转和 prediction-invariant interval 形式化 VLM semantic transformation；functional ANOVA 已给出从主效应中纯化交互的标准方法；group-invariant learning 与 invariant probabilistic prediction 已分别占据 orbit-averaged risk 和 proper-score invariance。

因此裁决是：

> **PAEL 作为论文主贡献：KILL。PAEL 作为预注册、reader-grounded 的 confirmatory estimand：CONDITIONAL KEEP。**

剩余可投稿空间不再是“等价变换组合会失败”，也不是“新 robustness metric”，而必须是一个可证伪机制：**cross-modal fusion 在不显著增加 interaction spectrum/energy 的情况下，把原本无临床方向的 product interaction 旋入 reader-loss gradient；保持谱与边际不变的上游 orientation intervention 能选择性消除临床损害。** 若只有正 PAEL 而没有这种能量—方向解耦和因果干预，最多是一个严谨的临床 robustness measurement paper，不足以形成 ICLR oral 级机制论文。

**最窄可写 novelty sentence：**

> We test whether a generative medical VLM selects a reader-harmful orientation of a full render-by-wording interaction—despite both axes being independently clinician-admitted and despite matched interaction spectra—and whether that orientation emerges and can be causally removed at a cross-modal fusion-to-decoder transition without changing either marginal effect.

这句话中的任何一个限定被删除，都会落入本报告中的已有工作；在实验通过前只能写成 research question，不能写成已经完成的贡献。

## 1. 冻结研究问题

- **RQ1：** equivalence composition failure、transformation-product interaction、orbit-averaged proper loss 或 isospectral orientation-adjusted risk 是否已被整套实现？
- **RQ2：** 哪些组成部分已经被 2024–2026 顶会、顶刊或可信 arXiv 工作占据，PAEL 的真实剩余边界是什么？
- **RQ3：** PAEL 是否只是 metric novelty；若要成为 ICLR 机制论文，需要什么独立、可反驳、可因果验证的预测？

## 2. 检索与核验方法

检索沿五条相互对抗的路径展开：（1）`equivalence/composition failure`、`composite metamorphic relation`；（2）医学 VLM 的 prompt paraphrase、image style/context 与 coordinated image–text perturbation；（3）VLM semantic robustness certification；（4）functional ANOVA、interaction matrix、SVD/AMMI 与 orientation；（5）group orbit averaging、proper scoring risk 与 probabilistic invariance。时间主窗为 2024-01 至 2026-08；为判断数学组件是否已知，纳入少量 2020–2023 基础工作。

每条 2026 引用均回到 arXiv、ACL Anthology、Crossref/IEEE DOI 或作者公开出版记录核验。需要特别区分：MetaRA、PSF-Med、Medical Context Distorts Decisions、CheXthought 当前是 arXiv 论文；Semantic Robustness Certification 的 arXiv 页面标注 “Accepted to ICML”；How Composite Metamorphic Relations Enhance Test Effectiveness 是 Crossref 已登记的 IEEE TSE 52(4):1617–1636；MedFocusLeak 有 ACL 2026 正式页面。本报告不把 arXiv 状态伪装成顶会录用。

## 3. 最近的十项工作

**表 1 的结论：没有单篇完全覆盖 PAEL，但 MetaRA + composite MR + PSF-Med 的组合已覆盖“多模态等价变换组合揭露额外 failure”这一高层故事；functional ANOVA + group/proper-risk 文献又覆盖其统计骨架。**

| 排名 | 工作与核验状态 | 已占据的对象 | 未覆盖的 PAEL 部分 | 碰撞强度 |
|---:|---|---|---|---|
| 1 | [MetaRA: Metamorphic Robustness Assessment for Multimodal Large Language Model-based Visual Question Answering Systems](https://arxiv.org/abs/2605.19307)（Xu et al., arXiv 2026） | 对 VQA 同时施加 question paraphrase 与 benign/local/style/background image transformation；检查保持 GT 的 joint metamorphic relation。MR3 与“wording × style”几乎同形。 | 没有完整 factorial marginals，没有组合相对两单轴的 nonseparability estimand，没有 clinician admission、reader distribution 或机制干预。 | **极强概念碰撞** |
| 2 | [How Composite Metamorphic Relations Enhance Test Effectiveness of DNN Testing: An Empirical Study](https://doi.org/10.1109/TSE.2026.3675285)（Wu et al., IEEE TSE 2026；DOI/Crossref 已核验） | 大规模比较 composite MR 与 individual MR 的 failure/fault revelation，并用 latent geometry 衡量 MR complementarity。 | 图像分类而非医学 VLM；不以临床 proper loss 定向，不做完整 render × wording reader-grounded residual。 | **极强问题碰撞** |
| 3 | [Semantic Robustness Certification for Vision-Language Models](https://arxiv.org/abs/2606.18839)（Yang et al., accepted to ICML 2026，状态据 arXiv） | 用 text proxies 构造 2D semantic plane，在定范数的 embedding rotation 上给出 closed-form prediction-invariant intervals，覆盖 style/shape/scene 等 semantic variations。 | 单条 image-embedding semantic path；text 是 transformation proxy 而非同时变化的 wording input；无 product interaction、reader loss、生成式 decoder 或两轴组合证书。 | **强几何/认证碰撞** |
| 4 | [PSF-Med: Measuring and Explaining Paraphrase Sensitivity in Medical Vision Language Models](https://arxiv.org/abs/2602.21428)（Sadanandan & Behzadan, arXiv 2026） | 临床等价 paraphrase、Med-VLM flip、text-only control、SAE feature 与 causal clamping。 | 只有 wording 轴；没有 image-render product、multi-reader Brier 或 isospectral interaction reference。 | **强医学措辞碰撞** |
| 5 | [Medical Context Distorts Decisions in Clinical Vision Language Models](https://arxiv.org/abs/2605.17436)（Restrepo et al., arXiv 2026） | 在 CXR 中系统改变 image–text alignment、irrelevant history 与 semantically equivalent prompts，显示文本支配及 prompt flips。 | 未以两个 independently admitted equivalence axes 构造完整 product，也未分离 interaction 与 marginals。 | **强临床多模态碰撞** |
| 6 | [When Background Matters: Breaking Medical Vision Language Models by Transferable Attack](https://aclanthology.org/2026.acl-long.1768/)（Ghosh et al., ACL 2026） | MedFocusLeak 联合优化背景图像扰动与文本/融合表示，并用 attention distraction 诱导临床合理但错误的诊断。 | 是 targeted adversarial attack，不是自然、独立承认的 equivalence；没有 reader-distribution product residual。 | **强 joint-axis 边界碰撞** |
| 7 | [CheXthought: A global multimodal dataset of clinical chain-of-thought reasoning and visual attention for chest X-ray interpretation](https://arxiv.org/abs/2604.26288)（Sharma et al., arXiv 2026） | 大规模 multi-reader CXR reasoning/attention，直接预测 human–human 与 human–AI disagreement，并改善 uncertainty communication。 | 不研究 render × wording composition，也不做 observed-vs-isospectral interaction loss。 | **强 reader-disagreement 组件碰撞** |
| 8 | [Modularity Trumps Invariance for Compositional Robustness](https://arxiv.org/abs/2306.09005)（Mason et al., arXiv 2023） | 明确指出对 elemental corruptions 的 invariance 不能保证对 corruption compositions 的 robustness。 | 单模态图像分类、非临床、无 clinician-equivalence/product proper loss。 | **强一般 composition 碰撞** |
| 9 | [Purifying Interaction Effects with the Functional ANOVA](https://proceedings.mlr.press/v108/lengerich20a.html)（Lengerich et al., AISTATS 2020） | 将不能由较低阶项表示的方差定义为 pure interaction，并给出可识别的 fANOVA decomposition。 | 不研究 transformation invariance、临床方向或 matched isospectral orientation。 | **统计骨架已占据** |
| 10 | [Invariant Probabilistic Prediction](https://arxiv.org/abs/2309.10083)（Henzi et al., Biometrika 2025） | 用 strictly proper scoring rules 定义跨 interventions/environments 的 probabilistic invariance 与 robustness，并证明任意 shift 下通常不存在统一 invariant probabilistic predictor。 | 环境风险而非同一输入的 product orbit；没有 factorial interaction 或 spectrum-matched orientation reference。 | **proper-risk 骨架已占据** |

两项基础工作进一步压缩“数学工具新颖性”。[A Group-Theoretic Framework for Data Augmentation](https://proceedings.neurips.cc/paper/2020/hash/f4573fc71c731d5c362f0d7860945b88-Abstract.html) 已把 augmentation 写成对近似不变群 orbit 的 loss averaging；AMMI/SVD 文献早已把 two-way centered interaction matrix 分解为 singular values 与左右方向。它们没有 PAEL 的临床对象，但意味着 orbit averaging、two-way centering、SVD 和 Haar rotation 都不能单独列为贡献。

最接近的 human/model transformation-orbit 参照是 [Assessing Visually-Continuous Corruption Robustness of Neural Networks Relative to Human Performance](https://arxiv.org/abs/2402.19401)（Shen et al., arXiv 2024）：它以 7,718 名参与者建立 14 类连续视觉 corruption 的 human-aware robustness 曲线，说明“人类仍等价而模型失稳”本身也不是新范式。它没有 wording 轴、完整 product interaction 或 reader-grounded medical target；但它要求 CECD 把 clinician admission 写成构造有效性的必要条件，而不能把“引入人类参照”列为独立贡献。

## 4. 对四个“是否已做”的逐项回答

### 4.1 Equivalence composition failure：高层问题已做，PAEL 的临床条件版本未做

MetaRA 已在 MLLM-VQA 中把 paraphrase 与 image transformation 同时施加，composite MR 文献更直接研究组合关系相对单项关系的 failure revelation；Mason et al. 也已证明 elemental invariance 不推导 compositional robustness。因此：

- **KILL：** “两个语义保持变换各自安全但组合会失败”作为一般新问题；
- **KILL：** “首次在 VLM 中联合改变 image 与 wording”；
- **CONDITIONAL KEEP：** 模型输出出现前由独立临床角色承认的两个轴、完整 factorial marginals、reader-distribution clinical loss 与 human product control 共同定义的窄对象。

这不是措辞上的细小区别。MetaRA 只有 canonical 与 joint-mutated cases，不能识别 joint failure 是否只是某一边际；PAEL 的完整 product 才能问 nonseparability。但如果 CECD 实验最后只报告 joint cell 比 clean 差，它会被 MetaRA 直接覆盖。

### 4.2 Transformation interaction energy / orientation：能量和分解已做，临床 harmful orientation reference 未检索到

Two-way ANOVA/fANOVA 已标准化地分离 grand mean、两个 main effects 与 centered interaction；AMMI 把 interaction matrix 的奇异值解释为低秩 interaction components；Haar orthogonal transformations 与 fixed-spectrum matrix orbits 也是成熟数学工具。Semantic Robustness Certification 进一步在 VLM embedding semantic plane 内以旋转参数化变化，同时保持相关分量范数。

本轮没有检索到把下列四项放在一起的工作：

1. 对同一 model–image–claim 的两轴 centered interaction；
2. 保持其全部 singular values 的 left/right Haar reference；
3. 用独立 reader probability 的 proper-loss gradient 定义“有害方向”；
4. 以 observed orientation 相对上述 reference 的 excess loss 为 estimand。

因此，**“isospectral reader-harm orientation contrast”可能是一个新 estimand，但不是新的线性代数、interaction decomposition 或 semantic rotation 方法。** 论文必须避免把 factor-contrast space 的 Haar orientation 与 Yang et al. 的 embedding semantic plane rotation 混称为“semantic rotation”。

### 4.3 Orbit-averaged proper loss：两个组件均已有，但现有工作未得到 PAEL

Chen et al. 已证明 group data augmentation 等价于在 transformation orbit 上平均 loss；Henzi et al. 已用 strictly proper scores 定义 distributional invariance，并警告不受限制的 transformation class 下 invariant probabilistic prediction 通常不存在。PAEL 与二者的区别是：

- 它不训练一个 invariant predictor，也不对原始 transformation group 直接平均 risk；
- Haar expectation 是对 **interaction orientation reference** 的积分，而不是对 observed input orbit 的 group average；
- Brier target 是 `q = reader votes / panel size`，而非单一 hard outcome；
- 采样推断来自 image-cluster bootstrap，Haar 仅是 deterministic stress reference，不能被写成 exact randomization law。

所以“orbit-averaged proper loss”不能声称新颖；只有“reader-oriented observed-minus-isospectral interaction loss”这一组合可能新。

### 4.4 Isospectral rotation-adjusted interaction risk：未发现 exact duplicate，但属于 metric-level synthesis

Exact-term、公式结构与相邻领域搜索均未找到直接重复。最接近的数学组件是 fANOVA/AMMI 的 centered interaction 与 SVD，最接近的 VLM 几何组件是 Semantic Robustness Certification 的 semantic-plane rotation，最接近的 testing object 是 MetaRA/composite MR，最接近的 clinical target 是 PSF-Med/CheXthought。

这意味着没有 exact duplicate，但也意味着 PAEL 是把成熟部件精确拼接成一个新临床 estimand。**这种 novelty 默认属于 metric/protocol synthesis，而不是 mechanism novelty。**

## 5. Semantic Robustness Certification 对 CECD 的具体压缩

Yang et al. 的工作不覆盖 CECD product interaction，原因很明确：其输入是 dual-encoder VLM 的 image embedding 与固定 label prompts；text prompts 用于指定一条 source-to-target semantic path，并不是第二个同时被改变的 user-wording factor。证书沿单一 extent `phi` 给出类别不变区间，没有 `render × wording` factorial cells、两条 marginal main effects或 mixed derivative。

但它杀死三种潜在包装：

- 不能声称首次用 language 指定 VLM semantic transformation；
- 不能声称首次在 semantic plane 中做定范数 rotation 并研究 prediction invariance；
- 不能把 PAEL 的 Haar geometry 本身当 ICLR 方法贡献。

它还提供一个强替代解释：如果 CECD 的 product harm 完全由两个 transform 将样本推近一个普通 semantic decision boundary 所预测，那么所谓“clinical orientation selection”只是 generic boundary proximity。应在 dev 冻结一个 Yang-style semantic-boundary proximity control；若它吸收 held-out PAEL，机制表述必须 KILL，最多保留 measurement result。

## 6. PAEL 是否只是 metric novelty？

**是，按当前定义，PAEL 本身只是一个严谨而有用的 estimand。** 它有三个优点：以 reader distribution 而非 binary cutoff 评判、明确移除两个 marginals、用同 spectrum reference 区分 interaction energy 与 clinical orientation。但这些优点回答的是“如何测”，不是“为什么发生”。

以下结果即使统计显著，也不足以单独构成 ICLR 机制贡献：

- observed PAEL 大于零；
- observed interaction 比 Haar orientation 更有害；
- 两模型都复现；
- CE listing 中同样出现 product failure。

它们可以建立一个新 clinical robustness object，但没有排除“decoder 非线性把任意 joint grid 映射到错误方向”“样本靠近普通决策边界”“prompt-copy head”“generic multimodal synergy”等机制。

## 7. ICLR 所需的机制预测：Fusion-Induced Clinical Orientation Selection

最紧凑且与 PAEL 数学对象有机一致的机制假设是：

> **模型不一定在 fusion 后制造更多 product interaction energy；它在 fusion-to-decoder transition 选择了 interaction 的临床方向。** 对同一 interaction spectrum，观察到的 render/prompt singular directions 在某个模型特异的 cross-modal transition 后才与 reader-loss gradient 对齐。

这给出四个相互绑定、可被否定的预测：

1. **Energy–orientation dissociation。** Layerwise centered interaction 的 Frobenius norm/singular values 可以在较早层已出现，但 observed-vs-Haar reader-loss alignment 只在一个 dev-selected fusion-to-decoder interval 显著上升。若 energy 与 PAEL 同步等比例增长，结论只是 amplification，不是 orientation selection。
2. **Upstream isospectral causal rescue。** 在最终 logit 之前，对 joint residual 做保持奇异值、activation norm、grand mean 与两轴 marginals 的 orientation replacement/patching，应在 image-disjoint confirmation 上选择性降低 PAEL；直接改 final logits 不算机制证据。
3. **Product selectivity。** 干预降低 joint PAEL 时，单轴 paraphrase sensitivity、单轴 render sensitivity、canonical clear-case accuracy 和 generic interaction energy 应基本保持；否则只是温度、缩幅或全面抑制视觉/语言信号。
4. **Path mediation。** 该 orientation jump 应由一个可定位的 cross-modal routing path 介导，并且在 prompt-copy/System-PIH、late textual override、semantic-boundary proximity、HALP、reader-alias 与 human product controls之后仍保留增量解释；层位置允许随架构变化，不能预设统一“早层”。

建议把因果 gate 冻结为：transition layer 只在 dev 选取；在 untouched confirmation 中，spectrum/norm/marginal-matched orientation intervention 相对 matched random rotations 使 PAEL 至少相对下降 20%，canonical clear-case performance 下降不超过 1 个百分点，且两个模型方向一致。数字是项目级 operational gate，不是临床 MCID。

这套预测若成立，PAEL 就从 metric 变成机制的 readout；若不成立，PAEL 仍可作为可信测量，但不能称 cross-modal mechanism。

## 8. KEEP / KILL 与执行含义

### KEEP

- PAEL 作为唯一 confirmatory statistic：reader-distribution Brier、完整 product、两个 marginals、same-spectrum Haar reference、image-cluster bootstrap。
- 研究对象限定为 **clinician-admitted clinical product nonseparability**，不声称一般 equivalence composition failure。
- 用 MetaRA 的 joint paraphrase+style/background MR 作为最接近 behavioral baseline；必须重算其 joint failure，同时显式加入两个 single-axis cells。
- 用 composite MR 的 “composite vs component” 作为概念基线，并强调 PAEL 多出的不是 failure count，而是 reader-oriented proper-loss residual。
- 用 Semantic Robustness Certification 风格的 boundary proximity 作为强替代解释控制。

### KILL

- PAEL 作为 standalone metric/method novelty；
- “首次发现 VLM 的 image × wording 组合脆弱性”；
- “首次发现 individually safe transformations do not compose”；
- “首次提出 semantic plane / norm-preserving rotation / orbit-averaged loss”；
- 仅凭正 PAEL 就声称 support-to-language 或 cross-modal fusion mechanism；
- 将 Haar reference 写成 exact randomization distribution 或 causal removal effect。

### 终止条件

以下任一成立，终止正面机制 framing：

1. clinical admission 或 human product control 显示组合对人类也不等价；
2. PAEL 被 MetaRA-style joint failure、full-grid generic geometry 或 semantic-boundary proximity 吸收；
3. layerwise 只有 interaction energy 增长，没有 spectrum-matched harmful orientation jump；
4. upstream orientation patch 不优于 random/norm/marginal-matched patch，或收益来自 clear-case accuracy/coverage 损失；
5. 机制仅在一个模型或少数 findings 成立。

**最致命的单一碰撞条件：** 若一个只使用 MetaRA/composite-MR joint failure 与 Yang-style semantic-boundary proximity、完全不知道 reader votes 的 generic model，在 held-out images 上吸收 PAEL，并且上游 isospectral orientation intervention 不再提供独立 rescue，则 CECD 退化为已知的 compositional metamorphic fragility；无论 Brier PAEL 多显著，机制主张都应 KILL。

## 9. 对 RQ 的直接回答

- **RQ1：** 未发现整套 exact duplicate；但 equivalence composition failure、joint image-question metamorphic testing、interaction purification、semantic-plane rotation、orbit averaging 与 proper-score invariance 均已分别完成。
- **RQ2：** PAEL 的剩余边界只在“独立临床 admission + 完整 factorial marginals + multi-reader proper loss + same-spectrum clinical orientation reference”这一 conjunction；MetaRA 与 TSE composite MR 使这一边界比先前判断更窄。
- **RQ3：** PAEL 单独是 metric/protocol novelty，不能支撑 ICLR oral。所需机制预测是 fusion-induced clinical orientation selection，并必须通过上游、isospectral、marginal-preserving causal intervention验证，而不是用 final-logit edit 自证。

## References

[1] Quanxing Xu et al., “MetaRA: Metamorphic Robustness Assessment for Multimodal Large Language Model-based Visual Question Answering Systems,” arXiv:2605.19307, 2026.  
[2] Huayao Wu et al., “How Composite Metamorphic Relations Enhance Test Effectiveness of DNN Testing: An Empirical Study,” IEEE Transactions on Software Engineering, 52(4):1617–1636, 2026.  
[3] Peiyu Yang et al., “Semantic Robustness Certification for Vision-Language Models,” accepted to ICML; arXiv:2606.18839, 2026.  
[4] Binesh Sadanandan and Vahid Behzadan, “PSF-Med: Measuring and Explaining Paraphrase Sensitivity in Medical Vision Language Models,” arXiv:2602.21428, 2026.  
[5] David Restrepo et al., “Medical Context Distorts Decisions in Clinical Vision Language Models,” arXiv:2605.17436, 2026.  
[6] Akash Ghosh et al., “When Background Matters: Breaking Medical Vision Language Models by Transferable Attack,” ACL, 2026.  
[7] Sonali Sharma et al., “CheXthought: A global multimodal dataset of clinical chain-of-thought reasoning and visual attention for chest X-ray interpretation,” arXiv:2604.26288, 2026.  
[8] Ian Mason et al., “Modularity Trumps Invariance for Compositional Robustness,” arXiv:2306.09005, 2023.  
[9] Benjamin Lengerich et al., “Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models,” AISTATS, 2020.  
[10] Alexander Henzi et al., “Invariant Probabilistic Prediction,” Biometrika, 112(1), 2025.  
[11] Shuxiao Chen, Edgar Dobriban, and Jane Lee, “A Group-Theoretic Framework for Data Augmentation,” NeurIPS, 2020.  
[12] Kun Qiu et al., “Theoretical and Empirical Analyses of the Effectiveness of Metamorphic Relation Composition,” IEEE Transactions on Software Engineering, 48(3):1001–1017, 2022.  
[13] Huakun Shen et al., “Assessing Visually-Continuous Corruption Robustness of Neural Networks Relative to Human Performance,” arXiv:2402.19401, 2024.  

## 检索限制

“未检索到 exact duplicate”不是 first-claim 证明。2026 文献仍在快速增加，且部分 IEEE/统计全文的公式级索引不完整；提交前必须按同一 exact-object query 再刷新一次。当前最重要的修正不是继续扩大 benchmark，而是把 MetaRA、composite MR 与 Semantic Robustness Certification 纳入 frozen collision controls，并在实验前锁死 energy–orientation dissociation 的因果预测。

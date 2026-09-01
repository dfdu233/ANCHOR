# C46：真实配对视图 Falsification Decoding 的公式级碰撞审计

日期：2026-08-13  
范围：2024--2026 多视图医学报告生成、VLM 多视图幻觉、test-time verification、外部视觉证据编辑，以及经典 likelihood-ratio / sequential evidence accumulation。  
资源约束：仅检索论文、审计本地文档与既有结果；未运行 GPU，未改动 baseline 队列。

## 结论

> **C46 作为新 hallucination mitigation 算法严格 NO-GO。**

“用同一患者第二真实视图的 claim-specific likelihood ratio 反驳草稿中的阳性 finding，再以固定 claim 数一换一纠错”没有留下足够的公式级新颖性：

1. 两个视图的 likelihood ratio 相加，精确等价于标准 Bayesian 多传感器融合；
2. 只接收第二视图的负响应，是 one-sided / clipped product-of-experts，而非新的证据代数；
3. 低于阈值才反驳，是固定样本 Neyman--Pearson 检验；推广到顺序视图就是 SPRT；
4. 固定正 claim 数后替换最低分 claim，是 constrained top-K MAP / uniform-matroid one-exchange，已与 C44 的数学和失败结果重合；
5. 2024--2026 医学 RRG 已广泛融合 frontal/lateral，多视图 hallucination 本身及 training-free mitigation 也已被 RSCD 直接提出；
6. 更严重的是，第二视图的“不支持”可能表示病灶在该投照下不可见，而不是病灶不存在。当前本地逐视图三态临床真值为 **0**，所以所谓“反证”在构念上不可识别；
7. 本地 IU-Xray 64-study 真实第二视图 pilot 只带来 `+1.56pp` accuracy，95% CI `[-4.69,+7.81]pp`；Brier 相对改善 `+5.92%` 的 CI 为 `[-2.04,+12.60]%`，均未通过预注册门。

唯一未被逐字实现的窄组合是“同患者互补投照 + 显式 `supported/refuted/unobservable` 视图状态 + 只接受有临床可证反例的非删除式 claim exchange”。但它目前既没有可执行真值，数学主体也仍是 likelihood-ratio test 加受限交换，不能支撑 ICLR 级方法贡献。

## 1. 冻结候选及研究问题

### 1.1 候选操作

给定主视图 \(x_1\)、真实配对视图 \(x_2\)，以及 VLM 从 \(x_1\) 生成的固定大小阳性 claim 集 \(A\)，对每个 claim \(c\) 计算第二视图的证据：

\[
\lambda_2(c)
=
\log\frac{p(x_2\mid Y_c=1)}{p(x_2\mid Y_c=0)}.
\]

若第二视图强烈反对某个已生成 claim，则删除或改写该 claim；为禁止“少说获益”，从未生成 ontology 中补回一个更受支持的 claim，使 \(|A|=K\) 不变。

### 1.2 审计问题

- RQ1：第二视图的 claim-specific likelihood ratio 是否超出普通 multi-view fusion？
- RQ2：asymmetric falsification 是否超出 ensemble/consistency、external verifier editing 或 sequential testing？
- RQ3：固定内容预算的一换一是否留下新的优化对象？
- RQ4：现有数据和本地结果是否足以识别“第二视图反驳了 claim”？

## 2. 公式级等价关系

### 2.1 它首先是标准 Bayesian 多视图融合

若两次投照在疾病状态给定后条件独立，主视图与第二视图的联合 posterior odds 为：

\[
\log\frac{p(Y_c=1\mid x_1,x_2)}{p(Y_c=0\mid x_1,x_2)}
=
\log\frac{p(Y_c=1)}{p(Y_c=0)}
+\lambda_1(c)+\lambda_2(c).
\]

因此，若主视图 margin \(s_1(c)\) 已表示 prior log-odds 加第一视图证据，那么

\[
s_{12}(c)=s_1(c)+\lambda_2(c)
\]

就是 textbook posterior-odds accumulation。把“第二视图”称作 falsifier 不改变其代数；它仍是 product-of-experts / score-level sensor fusion。

如果两个视图不条件独立，则简单相加会重复计算共享解剖和患者因素；正确对象应直接建模 \(p(x_1,x_2\mid Y_c)\) 或学习 joint discriminator，这又回到普通 multi-view representation/fusion，并且通常需要训练。

### 2.2 “只用反证”是单边截断，不产生新的统计对象

候选可能只允许第二视图降低 claim 分数：

\[
s_{\mathrm{neg}}(c)=s_1(c)+\min\{0,\lambda_2(c)\}.
\]

这只是对 likelihood ratio 做负半轴截断。它有一个显然的单调性质——第二视图永远不能提高 claim 分数——但这不是新的理论，而且并不推出错误率下降：真实阳性在侧位图不可见时也会被系统性降低。

若规则是

\[
\lambda_2(c)<-\tau \quad\Rightarrow\quad \text{refute }c,
\]

则它就是固定样本 one-sided likelihood-ratio test。允许逐视图继续观察时，累计量

\[
S_t(c)=\sum_{j=1}^{t}\lambda_j(c)
\]

越过上下阈值才停止，正是 Wald sequential probability ratio test 的基本形式。`claim-specific`、`medical view` 与 `falsification` 都没有改变这个等价关系。

### 2.3 固定 claim 预算回到 constrained top-K 与 C44

若最终必须输出 \(K\) 个阳性 claims，最自然规则为：

\[
\widehat A_K
=
\arg\max_{|A|=K}\sum_{c\in A}s_{12}(c),
\]

其解就是对融合分数取 top-K。若从原草稿开始每次移出最低分 claim、移入最高分候选，则是 uniform-matroid bases 上的 one-exchange local search。

这与 C44 Pareto Claim Exchange 的末端优化对象相同。区别只在于第二个分数源从“小视觉专家”变成“同患者第二视图”；“如何在没有真值时选择正确 replacement”这一核心困难没有消失。本地 C44 confirmation 中，真实无标签最陡一换一令固定-K 总错误增加：Huatuo `+3.35%`，Hulu `+19.43%`。这不直接证伪第二视图分数，但证明“固定 K”只是公平性约束，不能自动把 verification 变成可靠纠错。

## 3. 2024--2026 直接与邻近碰撞

| 工作 | 已覆盖的核心操作 | 与 C46 的关系 |
|---|---|---|
| MCL / EVOKE，2024--2025 | 用同患者多视图对比学习表示，并融合 patient-specific information 生成报告 | 已覆盖“第二真实 CXR view 提供互补 finding 信息”；区别仅是训练型 symmetric fusion，而非 post-hoc asymmetric test |
| MLRG，CVPR 2025 | 显式整合 current multi-view spatial information 与 longitudinal information；报告多视图相对单视图改善 | 已覆盖多视图 CXR 作为报告输入和 flexible fusion；C46 不能主张“第二视图用于更准确报告” |
| KCLVA，MIUA 2025 | 从报告抽取 view-specific terms，使用 view-specific attention 与 many-to-many contrastive learning | 已覆盖 view-specific claim/term 与不同投照的绑定；同时指出共享 study report 不给出逐图诊断归属 |
| View-PNDF，arXiv 2026 | 独立生成 view-specific reports，检测 view-specific neurons，再由 LLM consolidate 各视图报告 | 已覆盖 per-view generation、跨视图 consistency 与报告合并；C46 的 `draft → per-view verify → merge` 系统形态高度接近 |
| RSCD / *Revealing Multi-View Hallucination*，arXiv 2026 | 明确定义 cross-view hallucination，并用 training-free contrastive decoding 缓解；在多视图输入上抑制非目标视图干扰 | 直接占据“多视图 hallucination 是独立问题 + training-free mitigation”。C46 的医学互补视图用途不同，但不能再以 multi-view hallucination 作为新问题 |
| REVERSE，NeurIPS 2025 | 在生成中检测 hallucinated span、回溯并重采样纠正，而非只拒绝 | 占据 `generate → verify → revise`；C46 只把 verifier 的证据源换为第二图像 |
| CEBC，ACL 2026 | 外部 visual detector 给 evidence constraint，training-free minimal edit/suppress unsupported mentions | 若 C46 用 classifier/detector 估计 \(\lambda_2\)，系统上就是 detector evidence editing 的 multi-view 特例；固定内容预算不是新的 verifier |
| Visual Evidence Prompting，ACL 2025 | 把 object detector / scene graph 等视觉证据加入 prompt 来纠正 hallucination | 占据“额外视觉证据提示 VLM 纠错”的宽泛机制 |

这里最重要的区别是：MLRG/MCL/KCLVA 等通常训练融合器；C46 计划冻结 VLM，在测试时只用第二视图作反证。然而这种 `training-free + asymmetric` delta 在数学上已经被 likelihood-ratio test 和 clipped PoE 完全描述，系统上又位于 RSCD、REVERSE、CEBC 的交集。它是合理工程变体，但不是新的算法原语。

## 4. 医学构念上的致命问题：不可见不等于反驳

对于每个 claim，第二视图不仅存在疾病状态 \(Y_c\)，还存在该病灶在此投照中的可观察状态 \(O_{2c}\)：

\[
O_{2c}\in\{\text{visible},\text{unobservable}\}.
\]

真实判断至少依赖

\[
p(x_2\mid Y_c,O_{2c}),
\]

而不是只依赖 \(p(x_2\mid Y_c)\)。侧位片或正位片没有呈现某病灶，可能有三种含义：

1. 图像确实反驳该病灶；
2. 该投照对该病灶不敏感，属于 `unobservable`；
3. 图像质量或遮挡使证据不足。

若没有显式区分这三态，负 margin 同时混合了 `absent` 与 `not visible`。于是第二视图不能成为 sound falsifier。这个问题不是“评测不够好”，而是干预动作本身不可识别：算法不知道何时有权撤回 claim。

本地严格 substrate 审计已经确认：现有 MIMIC/IU 共享 study report 不提供 sibling-view 的 `supported / refuted / unobservable` 独立真值；Tam boxes 精确 join 后 354 张带框影像来自 354 个不同 studies，配对标注 study 为 0；MS-CXR、REFLACX、Chest ImaGenome、PadChest-GR、VinDr-Mammo 也不提供所需成对三态。正式可用 paired claim-view truth 数量为 **0**。缺 box 或共享 report 未提及不能伪造 negative。

## 5. 本地已有致死证据

### 5.1 真实第二视图 pilot 已未过门

IU-Xray 64-study Huatuo pilot 使用同 study 两视图和共享 report claim truth：

| 条件 | 结果 |
|---|---:|
| view0 accuracy | `90.63%` |
| 两真实视图固定融合 | `92.19%` |
| accuracy 增量 | `+1.56pp`, 95% CI `[-4.69,+7.81]pp` |
| Brier 相对改善 | `+5.92%`, 95% CI `[-2.04,+12.60]%` |
| two-view oracle headroom | `+6.25pp` |

预注册要求相对最佳同图 compute 至少 `+3pp` accuracy，或 Brier 相对改善至少 `5%` 且 CI 排除 0。两个正式端点均失败，因此未扩 Hulu。Oracle 仅说明两次回答偶尔互补，不提供无标签选择规则。

### 5.2 该 pilot 比 C46 还乐观

上述 truth 来自完整 study 的共享报告，无法知道某 finding 在哪张图可见；它最多证明 study-level complementarity proxy。C46 要声称“第二视图反驳原报告 claim”，需要更强而非更弱的逐视图真值。因此不能以换成 likelihood ratio 或开放生成来重开已经失败的同一 observation branch。

## 6. 剩余最小新颖点及为什么当前仍 NO-GO

检索中未发现一篇工作逐字实现以下完整 conjunction：

> 对同一患者互补投照分别给出 `support/refute/unobservable`；仅当第二视图产生经过临床可观察性验证的 refutation certificate 时，才在固定 claim 预算下把一个阳性 finding 换成另一个 finding。

但这个剩余点当前不能立项为算法贡献：

- `likelihood ratio + threshold` 是标准 hypothesis testing；
- `fixed-K replacement` 是标准 constrained MAP / one-exchange；
- `unobservable` 只是避免错误反证所必需的医学状态，不产生新的纠错定理；
- 当前没有任何合格 paired-view 三态真值，无法验证 certificate 的 soundness；
- 本地最接近的真实第二视图试验没有统计显著增益；
- 若用外部 detector 代替三态真值，则直接落入 CEBC / Visual Evidence Prompting；
- 若把两图同时送入生成器，则回到 MLRG/MCL/KCLVA 的 multi-view fusion；
- 若分别生成再合并，则接近 View-PNDF；
- 若用 attention/logit contrastive mitigation，则与 RSCD 直接邻接，并且属于本项目已排除的 attention-mask/contrastive decoding 路线。

## 7. 冻结决定

| Gate | 证据 | 判定 |
|---|---|---|
| 新病例级信息 | 真实第二视图原则上可增加信息 | PASS in principle |
| 公式级新颖性 | Bayesian LR accumulation + one-sided test + constrained top-K | **FAIL** |
| 2024--2026 系统碰撞 | multi-view RRG、cross-view hallucination mitigation、verification/editing 全部已有强近邻 | **FAIL** |
| 本地自然现象 | n=64 增益 CI 跨 0 | **FAIL** |
| 可识别临床反证 | paired per-view 三态 truth = 0 | **FAIL** |
| 固定内容预算安全纠错 | C44 一换一已显著增错；C46 无新选择保证 | **FAIL** |

**最终决定：C46 作为 mitigation 方法永久关闭，不运行 GPU，不改阈值，不用新的名称续跑。**

若未来获得独立逐视图三态医生标注，它可作为“view observability-aware verification”数据/临床问题重新研究；但需要一个超越标准 likelihood-ratio testing 的新数学或新可执行机制，才能重新竞争方法论文。当前应把探索资源转向不依赖第二视图、不依赖 verifier score fusion、且能从单次患者视觉表示产生新纠错约束的候选。

## 核验过的主要来源

1. Liu et al., “Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation,” CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Enhanced_Contrastive_Learning_with_Multi-view_Longitudinal_Data_for_Chest_X-ray_CVPR_2025_paper.html
2. Liu et al., “MCL: Multi-view Enhanced Contrastive Learning for Chest X-ray Report Generation,” arXiv:2411.10224, 2024. https://arxiv.org/abs/2411.10224
3. Zhu and Lu, “KCLVA: Knowledge-enhanced Contrastive Learning and View-specific Attention for Chest X-ray Report Generation,” MIUA 2025. https://eprints.whiterose.ac.uk/id/eprint/228682/
4. Chen et al., “Seeing Through Multiple Views: Parameter-Efficient Fine-Tuning via Selective Neurons for Consistent Radiology Report Generation,” arXiv:2606.31099, 2026. https://arxiv.org/abs/2606.31099
5. Park et al., “Revealing Multi-View Hallucination in Large Vision-Language Models,” arXiv:2603.23934, 2026. https://arxiv.org/abs/2603.23934
6. Wu et al., “Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling,” NeurIPS 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/5eff08bd064f0cdd92182cdf6fd06b99-Abstract-Conference.html
7. Mishra et al., “CEBC: Conformal Evidence-Bounded Control for Low-Hallucination Vision–Language Generation,” ACL 2026. https://aclanthology.org/2026.acl-long.2142/
8. Jiang et al., “Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models,” ACL 2025. https://aclanthology.org/2025.acl-long.205/

本地证据入口：

- `docs/daylong_idea_search/active_sensing_collision.md`
- `docs/STUDY_IMAGE_SCOPE_ALIASING_PER_VIEW_TRUTH_NO_GO_20260803.md`
- `corrected_runs/daylong_idea_search_v1/iuxray_observation_pilot64_v1/analysis_huatuo.json`
- `docs/daylong_idea_search/C38_COLLISION_AUDIT.md`


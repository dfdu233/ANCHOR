# Witness–Certificate Logic：局部病灶为什么能证明“有”，却不能证明“无”

日期：2026-08-12  
范围：已有结果、数学与文献碰撞审计；不占 GPU，不修改 baseline。

## 0. 严格裁决

**保留为机制解释和实验设计原则，关闭为 hallucination mitigation 主算法。**

这个方向能给出一个简洁、与阈值无关的不可辨识结论：对“图中某处存在病灶”这类
existential claim，一个局部病灶可以成为阳性 witness；但只看不完整区域时，阴性图与
“病灶恰好藏在未观察区域”的阳性图不可区分。因此，任何只用 crop 的二元解码器都不可能
在所有病例上同时降低 FP 且不增加 FN。

但这个结论**不能自然推出**一个同时满足以下条件的方法：

1. 不训练；
2. 不依赖额外真值或阈值；
3. 不输出 `Unknown` / abstain；
4. 相对原模型严格降低 FP 且不增加 FN；
5. 不退化为普通 global–local fusion。

能绕开下界的办法只有三类：观察完整图或获得新视图、加入关于未观察区域的分布假设、
或把不可判定样本输出为 `Unknown`。前两类不再是“从 crop 免费恢复证据”，第三类就是
selective prediction / abstention。若把 full 与 crop 的 logits 做 min、max、加权或对比，
则是 HALC、SECOND、FGVP、PND 等工作的直接拥挤区，而且没有无交换损失保证。

所以：**这个方向有一条干净的 theorem，但没有一条新的、非拒答的 mitigation algorithm。**
它最多可以成为“为什么局部增强系统性制造医学 FP”的机制章节或 negative-result 论文的一部分，
不能单独承担 ICLR Oral 主线。

## 1. 当前现象到底支持什么

Huatuo、VinDr 的同一批 62 个 0/3 阴性病例与 62 个 3/3 阳性病例中，neutral prompt 下：

| 输入 | 阴性病例报阳性（FP） | 阳性病例报阳性（recall） |
|---|---:|---:|
| 完整胸片 | 8.1% | 62.9% |
| 保持 ROI 原位置和尺度、模糊 ROI 外部 | 71.0% | 87.1% |
| ROI 原尺度 + sham panel | 64.5% | 91.9% |
| 放大 ROI + sham panel | 67.7% | 91.9% |
| 放大 ROI + true-context panel | 58.1% | 90.3% |

最硬的配对结果是：

- 在阴性病例上，仅移除外部 context 就让 FP 增加 `+62.9pp`，95% CI
  `[+50.0,+74.2]pp`；
- 保持 sham panel，只把 ROI 从 native scale 放大，FP 仅再增加 `+3.2pp`，
  95% CI `[-6.5,+11.3]pp`；
- 把 true context 放回 zoom panel，相对 sham panel 降低 FP `9.7pp`，
  95% CI `[3.2,17.7]pp`，同时阳性 recall 只从 `91.9%` 变为 `90.3%`。

这说明 **scale 不是 8.1%→71.0% 的主要解释，外部图像内容很重要**。它与“阴性判断需要
更完整的观察”一致。

但现在还不能把它写成“已经发现分布式 negative certificate”，原因有三点：

1. crop 同时把阳性与阴性 margin 都向 `Yes` 推动；它可能主要是 visual-prompt-induced
   criterion shift，而不是模型读取了病例特异的反证。
2. 阳性 ROI 来自专家 bbox，而阴性 ROI 来自随机网格；这是 label-dependent oracle
   selection。`91.9%` 的阳性 crop recall 与 `67.7%` 的阴性 FP 不能作为公平分类器比较。
3. 模糊外部区域正是 FGVP 的 Blur Reverse Mask visual prompt。它改变模型的输入语义和
   注意力分配，不等价于在自然 partial observation 下仅“少看一些像素”。

作为补充审计，现有 124 例在阈值 0 下的 BAcc 从 full 的 `77.4%` 降到
native-context-removed 的 `58.1%`；尽管 crop 的排序 AUROC 受 oracle ROI 影响可看起来较高，
其部署工作点发生了严重漂移。因此目前证据支持的是 **context removal causes a large
positive shift**，而不是一个已确认的 certificate reader。

## 2. 必要的数学背景

### 2.1 把一张图写成局部命题

先只考虑真正适合局部存在性表达的 finding，例如小结节。把全图划分为 `M` 个区域，
令

\[
z_i=1
\]

表示第 `i` 个区域中存在目标病灶。整图 claim 是 Boolean OR：

\[
Y(z)=z_1\lor z_2\lor\cdots\lor z_M.
\]

背景说明：在 decision-tree / property-testing 文献中，**certificate** 是一小组已观察变量，
其取值足以唯一确定函数输出；“witness”通常指证明输出为真的 certificate。

对 OR 函数：

\[
C_1(\mathrm{OR}_M)=1,\qquad C_0(\mathrm{OR}_M)=M.
\]

证明很简单：看到任意一个 `z_i=1` 就足以证明 `Y=1`；但要证明 `Y=0`，必须确认所有
`M` 个区域均为 0。只要还漏一个区域，该区域就可能藏着病灶。

这不是新定理。它是 certificate complexity 的标准例子，也精确等价于经典
multiple-instance learning (MIL) 的假设：positive bag 只需一个 positive instance，
negative bag 要求所有 instances 都为 negative。

### 2.2 Partial-observation 不可能性

令 `S` 是 crop 实际显示的区域集合，且 `S` 不是全图。观察记为 `O_S(z)`。因为至少有一个
区域 `j` 没被看见，可以构造两个完整病例：

- `z^-`：所有区域均为 0，所以真值是阴性；
- `z^+`：除未观察的 `j` 为 1 外，其余均为 0，所以真值是阳性。

二者在 crop 中完全相同：

\[
O_S(z^-)=O_S(z^+),\qquad Y(z^-)=0,\quad Y(z^+)=1.
\]

设任意随机二元解码器看到该 crop 后，以概率 `p` 回答 Yes。由于输入相同，它在两个病例上
必须使用同一个 `p`。于是

\[
\mathrm{FPR}(z^-)=p,\qquad
\mathrm{FNR}(z^+)=1-p,
\]

因此有一个完全不依赖 logit 阈值、模型架构或训练数据的边界：

\[
\boxed{\mathrm{FPR}(z^-)+\mathrm{FNR}(z^+)=1.}
\]

如果原解码器在这个 crop 上总答 Yes，则它在这对病例上是 `(FP,FN)=(1,0)`；任何降低
Yes 概率的修正都会等量增加另一完成状态的 FN。换言之：

> 对观察完全相同、真值相反的 completion pair，crop-only 方法不可能 Pareto 改善 FP 与 FN。

这条结论比“需要调一个更好的 threshold”更强，因为它说的是**信息本身不足**。

### 2.3 为什么 crop 不能创造新证据

crop `C=T(X)` 是完整图 `X` 的确定性变换。在 Blackwell information order 中，`C` 是
`X` 的 garbling：知道 `X` 就能复现 `C`，反过来通常不行。对 Bayes-optimal 决策者和任意
损失函数，完整图的最优风险不会高于 crop。

因此 crop 让一个固定 VLM recall 上升，只能说明 crop 改变了有限模型的表示、分辨率或
决策工作点；它没有在信息论意义上产生新临床证据。这一点与当前“recall 上升、FP 更剧烈上升”
完全一致。

### 2.4 适用边界

这个 OR 模型不能覆盖所有医学 claim：

- Nodule/Mass、局灶性气胸等比较接近 local existential witness；
- Cardiomegaly、Aortic enlargement 依赖全局比例或结构关系，阳性本身也需要 global
  certificate；
- diffuse opacity、fibrosis 可能需要多个区域的分布模式；
- location、size、relation claim 的 certificate complexity 与纯 OR 不同。

所以即使 theorem 正确，也不能把“阳性局部、阴性全局”写成所有医学 finding 的普遍规律。
论文必须先按 claim 的逻辑结构分层，而不能把极性当作唯一划分。

## 3. 三值逻辑能做什么，又为什么不构成所需算法

若每个区域有一个**可靠**局部验证器，输出 `T/F/U`（有病灶、确定无病灶、无法判定），
则 OR claim 可以用 Kleene-style 三值规则合成：

- 任一区域为 `T`：整图为 `T`；
- 所有区域均为 `F`：整图为 `F`；
- 其余情况：整图为 `U`。

这个规则很优雅，也与 witness/certificate 完全一致。但它没有解决两个关键问题：

1. 当前 VLM 的 patch score 不是 sound verifier；把分数变成 `T/F/U` 仍需要监督校准、
   conformal threshold 或额外模型。
2. 当观察不完整时，规则的正确输出必然是 `U`。如果禁止 abstention，又必须把 `U` 强制
   压成 Yes 或 No，不可能性立即回来。

因此三值逻辑能保证的是“不要把未知伪装成确定”，不是“在维持覆盖率时同时降低 FP 和 FN”。
这与 Reliable VQA、Selective VQA、Unanswerable VQA、CLEVR-POC 以及 BCEA 的问题设定直接
相邻；若把 `U` 计作成功而不报告 coverage，只是在指标层隐藏错误。

## 4. 能否推出真正降低 FP 且不增 FN 的方法

### 4.1 四种自然尝试

| 尝试 | 为什么看起来合理 | 严格结果 |
|---|---|---|
| `full AND crop` 才报阳性 | 用全局图否决 crop FP | 会删除 full miss / crop hit 的真阳性，不能保证 FN 不增 |
| `full OR crop` 报阳性 | 保留 crop 找到的小病灶 | 不会减少 full 的 FP，通常增加 crop FP |
| full/crop logit 加权、min/max、contrast | 可移动工作点 | 普通 global–local fusion；无分布无关 Pareto 保证 |
| `T/F/U` certificate gate | 与逻辑最一致 | 安全来自 Unknown/abstention，而不是提高二元准确率 |

### 4.2 一个必要条件

设 base 模型报阳性的样本集合为 `B`，某个 gate `G` 决定保留哪些阳性。若想降低 FP 而完全
不增加 FN，必须满足：

\[
P(G=1\mid Y=1,B=1)=1,
\]

同时

\[
P(G=1\mid Y=0,B=1)<1.
\]

第一式要求 gate 对 base 的所有真阳性有 **100% recall**，第二式要求它又能排掉一部分假阳性。
在没有标注、没有新增观察、没有关于 unseen completion 的假设时，partial-observation theorem
说明不存在这样的通用 gate。任何论文若声称“FP 降、FN 不增”，都只能是特定测试分布上的
经验现象，不能由 witness/certificate 逻辑推出。

### 4.3 唯一诚实的算法边界

若允许输出 `Unknown`，可以定义 **Certificate-Bounded Reporting**：

- 只有获得局部 positive witness 时，允许 definite-positive；
- 只有观察覆盖所需解剖且所有区域通过 negative verifier 时，允许 definite-negative；
- 其余输出 uncertain / recommend additional view。

它可能降低“确定性 hallucination”，但一定要同时报告 matched coverage、abstention rate、
FP/FN among all claims 和 among answered claims。这个终点已经是 selective/conformal VQA，且
BCEA 还进一步覆盖了 acquire-more-evidence；它不是所要求的新 mitigation 主线。

## 5. 与最接近工作的碰撞

| 工作 | 已覆盖的对象 | 与本候选的关系 | 裁决 |
|---|---|---|---|
| Maron & Lozano-Pérez, *A Framework for MIL*, NeurIPS 1997 | positive bag 存在一个 positive instance；negative bag 的所有 instances 都为 negative | witness/certificate 的统计学习版本完全相同 | 数学核心直接碰撞 |
| FGVP, NeurIPS 2023 | crop、box、mask、Blur Reverse Mask 等 visual prompts；模糊目标外区域强化局部 | 当前 `native_context_removed` 正是其核心输入族 | 不能把 blur-outside 当新方法 |
| HALC, ICML 2024 | auto-focal local grounding + global beam search，training-free | 已占 local/global 互补与 focal contrast | 普通融合直接碰撞 |
| SECOND, ICML 2025 | 熵驱动选择、多尺度细化与逐尺度 contrastive decoding | 已占“逐步找 witness 并精细化” | 不能再做 region mask/scale contrast |
| VISTA, ICML 2025 | 早层 visual evidence steering 与 logit augmentation | 已占“恢复被语言层压掉的视觉信息” | certificate 不能只换名为 steering |
| PND, 2026 preprint | positive path 放大 salient local evidence；negative path degrade core object；训练免费 global/local grounding | 与正负路径 + global/local decoding 高度重叠 | 方法空间拥挤 |
| Reliable VQA, ECCV 2022；Selective VQA, CVPR 2024 | 允许 abstain，以 risk–coverage 评价错误回答 | 已占 `Unknown` 的实用语义 | 三态输出非新 |
| CLEVR-POC, LREC-COLING 2024 | partial scene + logical constraints 下推断隐藏对象 | 已占 partial observability + logic 的 VQA setting | 仅换成胸片不够 |
| BCEA, 2026 preprint | answer/abstain/acquire；crop/zoom 后对完整 acquisition policy 做 conformal 校准 | 已占证据不足时重看与有限样本风险保证 | conformal certificate 直接拥挤 |
| Riedlinger et al., ICLR 2026 | point-process empty-space / absence confidence calibration | 已占“如何可靠证明没有目标”的检测问题 | absence certificate 邻近碰撞 |

在本次 exact/synonym/adjacent-field 检索下，没有找到把 Boolean certificate complexity 明确用于
医学 VLM hallucination 的同名工作；但 **MIL 已经包含相同逻辑，partial/selective VQA 已包含
相同决策语义，HALC/SECOND/FGVP/BCEA 已包含相同干预族**。剩余差异只是一个更清楚的失败解释，
不足以形成新算法。

## 6. 若只验证机制，唯一值得的致死实验

这不是新方法实验，而是判断“外部 context 真的是病例级 negative certificate，还是 generic
visual-prompt bias”的最小实验。必须先消除当前的 oracle ROI confound。

### 数据与渲染

1. 只选真正接近 existential OR 的 focal findings；cardiomegaly 等全局 claim 单列。
2. 用固定网格或冻结、label-independent selector 为阳性和阴性选择 ROI；不能 positive 用 bbox、
   negative 用 random。
3. 对同一 ROI 保持像素、位置、尺度完全不变，只改变外部：
   - true patient context；
   - anatomy-matched donor context；
   - 同像素 coverage 但 spatially permuted context；
   - blur context。
4. 加入完整图与完整图 tile-permutation：前者保留 coverage+relations，后者只保留 coverage。

### 唯一预测

- **病例级 certificate**：true context 对阴性 claim 的抑制显著强于 donor/blur，且不会同幅压低
  真阳性；tile permutation 可区分“覆盖”与“关系”证据。
- **generic criterion shift**：只要外部像完整胸片就整体降 Yes，true/donor 差异不提供 label
  增量；positive 与 negative 都同向移动。
- **FGVP / saliency effect**：blur-outside 的变化主要由 visual prompt 格式决定，空间语义
  交换后仍近似保留。

### 预注册门

只有同时满足以下条件，才允许把“negative certificate”写为确认机制：

1. `true-context vs matched-donor/shuffle` 的 class interaction 在两模型、多数 focal findings
   同方向，image-bootstrap 95% CI 排除 0；
2. context statistic 在 `full margin + finding identity` 上提供 fresh-test macro AUROC
   `>=+0.02`，且 CI 下界 `>0`；
3. label-independent ROI 下复现；
4. matched positive recall 损失不超过 1pp；
5. cardiomegaly 等非 OR findings 显示预期不同的 coverage law。

即使全部通过，最大可信结论仍是“医学 claim 的 certificate complexity 能预测局部增强何时
制造 FP”，不是已经得到一条新 mitigation algorithm。若增量门失败，则把 8.1%→71.0% 归为
输入格式导致的 criterion shift，关闭 certificate 机制。

## 7. 最终定位

### 可以保留的一句话

> 对局部存在性 claim，病灶是短阳性 witness，而无病灶是长阴性 certificate；删除图像上下文
> 会保留前者的可能性，却从信息上破坏后者，因此 crop-based enhancement 不可能获得分布无关的
> “降 FP 且不增 FN”保证。

### 不能声称的内容

- 不能声称所有医学阳性都是局部 witness；
- 不能把目前 oracle bbox positive crop 的高 recall 当公平证据；
- 不能把 `Unknown` 后 answered-risk 下降包装成覆盖不变的 hallucination mitigation；
- 不能把 full/crop logit 融合包装成 certificate theorem 的自然算法；
- 不能把 OR 的 `C1=1,C0=M` 或 partial pair 下界作为新数学贡献。

### ICLR Oral 判断

作为**算法**：关闭。  
作为**单独理论**：标准 certificate/MIL/Blackwell 数学，不够。  
作为**跨方法机制规律**：若上述 label-independent、跨模型实验成立，并能预测 SECOND/HALC/FGVP
何时由 recall gain 转为 FP inflation，可成为一篇机制/评测论文的重要章节；当前尚未达到
ICLR Oral-ready。

## 参考文献

1. Maron, O., Lozano-Pérez, T. “A Framework for Multiple-Instance Learning.” NeurIPS 1997. <https://papers.nips.cc/paper_files/paper/1997/hash/82965d4ed8150294d4330ace00821d77-Abstract.html>
2. Buhrman, H., de Wolf, R. “Complexity Measures and Decision Tree Complexity: A Survey.” *Theoretical Computer Science*, 2002. <https://doi.org/10.1016/S0304-3975(01)00144-X>
3. Blackwell, D. “Equivalent Comparisons of Experiments.” *The Annals of Mathematical Statistics*, 1953. <https://doi.org/10.1214/aoms/1177729032>
4. Shtedritski, A. et al. “Fine-Grained Visual Prompting.” NeurIPS 2023. <https://papers.nips.cc/paper_files/paper/2023/hash/4e9fa6e716940a7cfc60c46e6f702f52-Abstract-Conference.html>
5. Chen, Z. et al. “HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding.” ICML 2024. <https://openreview.net/forum?id=EYvEVbfoDp>
6. Park, W. et al. “SECOND: Mitigating Perceptual Hallucination in Vision-Language Models via Selective and Contrastive Decoding.” ICML 2025. <https://proceedings.mlr.press/v267/park25c.html>
7. Li, Z. et al. “The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering.” ICML 2025. <https://openreview.net/forum?id=7BKcLeHQsm>
8. Whitehead, S. et al. “Reliable Visual Question Answering: Abstain Rather Than Answer Incorrectly.” ECCV 2022. <https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960146.pdf>
9. Khan, Z., Fu, Y. “Consistency and Uncertainty: Identifying Unreliable Responses From Black-Box Vision-Language Models for Selective Visual Question Answering.” CVPR 2024. <https://openaccess.thecvf.com/content/CVPR2024/papers/Khan_Consistency_and_Uncertainty_Identifying_Unreliable_Responses_From_Black-Box_Vision-Language_Models_CVPR_2024_paper.pdf>
10. Abraham, S. S., Alirezaie, M., De Raedt, L. “CLEVR-POC: Reasoning-Intensive Visual Question Answering in Partially Observable Environments.” LREC-COLING 2024. <https://aclanthology.org/2024.lrec-main.293/>
11. Xu, J. et al. “Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable Vision-Language Models.” arXiv, 2026. <https://arxiv.org/abs/2606.16667>
12. Riedlinger, T., Maag, K., Gottschalk, H. “Towards Reliable Detection of Empty Space: Conditional Marked Point Processes for Object Detection.” ICLR 2026. <https://openreview.net/forum?id=M2KLWLHzX0>
13. Jiang, Y. et al. “Global Context or Local Detail? Adaptive Visual Grounding for Hallucination Mitigation.” arXiv, 2026. <https://arxiv.org/abs/2604.24396>


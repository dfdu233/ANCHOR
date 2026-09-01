# Search-Calibrated Visual Evidence：数学与新颖性审计

日期：2026-08-12  
任务边界：只审计“局部视觉搜索是否产生可测的 false evidence，以及如何校准”；不把经典 scan、multiple testing 或 conformal theorem 重新包装成原创数学。

## 结论先行

这个方向**有一个自然且 VLM-specific 的可验证命题**，但当前还不是完成态方法：

> 许多局部增强方法先用 claim 选择响应最大的区域，再在高度相关的同一图像/同一模型上验证或放大该区域；即使 claim 不存在，选择噪声的赢家也会在第二次响应中部分保留。这个“selection–reuse inflation”随有效 claim×region 搜索空间增长，可制造看似局部、实则由搜索产生的视觉证据。

“最大值要支付 `sqrt(2 log M)`”本身是经典极值/scan 结论，不是贡献；split conformal 的有限样本有效性也不是贡献。可能的新东西只能是：

1. 在 VLM local enhancement 中实证确认 **选择—复用膨胀律**；
2. 证明它统一预测不同分辨率、ontology 大小和局部方法的 false-positive 变化；
3. 用**完整流水线级**的零假设校准去掉该膨胀后，仍保留 bbox 对齐的真临床证据，并在 matched operating point 下减少 FP。

如果前两项不成立，应立即关闭，不应只留下一个 conformal verifier。后者已与 [ConfLVLM, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.576/) 和 [CEBC, ACL 2026](https://aclanthology.org/2026.acl-long.2142/) 强碰撞。

---

## 1. 随机变量与合理假设

### 1.1 固定对象

- `x`：一张影像；测试与校准必须按 patient/image 独立。
- `c in C`：临床 claim，例如 `Nodule present`。CE 中 `C={c}`；OE 中 `C` 是**在看测试图前冻结**的 ontology 或由一条冻结生成程序产生的候选集。
- `W in W_s(x)`：尺度 `s` 下的候选区域；所有尺度与窗口生成规则必须在测试前冻结。
- `f`：冻结 VLM；不更新任何权重。
- `Y_c in {0,1}`：claim 是否存在。VinDr 的严格零假设先用 reader vote `0/3`，清晰阳性用 `3/3`；`1/3,2/3` 单独分析，不混入 null。

对每个 claim×region 定义两个量：

\[
A_{c,W}(x)=\text{用于选择区域的标准化局部响应},
\]

\[
B_{c,W}(x)=\text{选择后用于确认 claim 的标准化响应}.
\]

它们可以来自 patch-token affinity、keep-region 后的 claim log-odds、erase-region 前后的 signed margin，或 local-enhancement 后的 margin；但必须在预注册中固定一种定义，不能看到结果后切换。`A` 和 `B` 分开非常重要：AGLA、SECOND 一类方法并非只计算一个静态 patch 分数，而是“先选局部，再用局部改变解码”。[AGLA](https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html) 组合 prompt-relevant local 与 global features；[SECOND](https://proceedings.mlr.press/v267/park25c.html) 逐步选择并对比多尺度视觉信息。它们都没有显式审计选择产生的统计溢价。

### 1.2 搜索与输出

固定候选集合

\[
\mathcal I=\{(c,s,W):c\in\mathcal C,\;W\in\mathcal W_s(x)\},
\qquad M=|\mathcal I|.
\]

选择器为

\[
\hat i=\arg\max_{i\in\mathcal I} A_i,
\]

局部增强实际使用的证据可写为 $T=B_{\hat i}$；若同一个分数既选又验，则 $A=B$，选择偏差最强。普通 scan 则是 $T=\max_i A_i$。

### 1.3 最低限度假设

解析近似需要强假设，正式校准不应依赖它们：

1. **单点 tail 假设**：在 `Y_c=0` 时，每个已标准化 `A_i` 至少近似零均值、1-sub-Gaussian。它只足够给保守 union bound。
2. **patch 不独立**：重叠窗口、共享视觉 token、同一 decoder 会造成强相关；主分析必须允许任意图内依赖。
3. **跨图 exchangeability**：同一 claim、模型、投影分辨率和预设 nuisance stratum 内，新的 null 图与校准 null 图可交换。这是 split-conformal 有效性的真正条件。
4. **完整选择可复现**：若测试时会在 8 个 claims、3 个尺度、256 个区域和 2 种 rendering 中选择最好者，校准时必须完整重跑同一搜索；只校准最后被选中的单一分数是无效的 post-selection inference。
5. **域漂移不自动满足**：跨医院、设备、AP/PA view 时 exchangeability 可能失效；需分层（Mondrian calibration）、加权校准或单独报告，不能声称天然 domain invariant。

---

## 2. 两个标准但必要的定理（均不原创）

### 定理 A：sub-Gaussian 搜索上界

若 `Z_1,...,Z_M` 每个都满足 `E exp(lambda Z_i) <= exp(lambda^2/2)`，则**不要求它们相互独立**：

\[
\Pr\!\left(\max_{i\le M} Z_i
\ge \sqrt{2\log(M/\alpha)}\right)\le\alpha.
\]

证明轮廓：单点 Chernoff bound 给出 `P(Z_i>=t)<=exp(-t^2/2)`；再用 union bound 得到 `P(max Z_i>=t)<=M exp(-t^2/2)`，令右侧等于 `alpha`。

这条结果的重要提醒是：常写的 `sqrt(2 log M)` 只是 iid Gaussian 最大值的一阶位置，不是 level-`alpha` 阈值；缺少 `alpha` 就没有 false-positive 保证。多尺度 scan 还需要 scale-dependent critical values；Walther 等人的经典工作已经系统研究该问题，见 [Walther 2010](https://arxiv.org/abs/1002.4770) 和 [Walther & Perry 2022](https://arxiv.org/abs/2008.06136)。因此本定理只能作背景，不可作为论文贡献。

### 定理 B：完整流水线的 split-conformal rank 有效性

令 `X_0` 是测试 null 图，`X_1,...,X_n` 是 null calibration 图；若它们在给定 nuisance stratum 后可交换，并对每张图运行完全相同的自适应搜索流水线得到

\[
T_j=\mathcal A(X_j)=B_{\hat i(X_j)}(X_j),
\]

则

\[
p(X_0)=\frac{1+\sum_{j=1}^n\mathbf 1\{T_j\ge T_0\}}{n+1}
\]

满足 `P(p<=alpha)<=alpha`（ties 可随机化得到精确 level）。

证明轮廓：在 null 下 `T_0,...,T_n` 仍可交换，因此测试样本的降序 rank 在 `1,...,n+1` 上均匀；rank 小于等于 `alpha(n+1)` 的概率不超过 `alpha`。

这个定理允许**任意图内 patch 依赖、任意 max/scan、任意确定性 selector**，前提是校准重放完整 selector。它同样是标准 conformal/randomization 结论，不原创；相关分布无关 scan 可参考 [Arias-Castro et al., 2018](https://arxiv.org/abs/1508.03002)，conformal p-value 的依赖与多重检验问题可参考 [Bates et al., 2021](https://arxiv.org/abs/2104.08279)。

边界必须说清：按 claim `c` 在 `Y_c=0` 图上校准，只保证这个**预先指定 claim** 的 null p-value；若 OE decoder 先看图再从大 ontology 中挑一个 claim，逐 claim p-value 不再自动对 selection 有效。此时要么把“生成候选 + 区域搜索 + 取最大”全部纳入同一个 calibration statistic，得到更保守的 family-wise 保证；要么另做 selective/FDR 推断。并且在部分 claims 为真、部分为假的复合情形，单一 global-null max calibration 不提供 false-claim FDR 保证。第一篇论文最安全的正式定理应限定 CE/预指定 claim，OE 只作为无保证的经验扩展，除非补出正确的 dependent selective inference。

---

## 3. 可能新颖的 VLM 命题：Selection–Reuse Inflation

### 3.1 一个可证的 toy proposition

在 absent claim 下，假设区域对 `(A_i,B_i)` 在不同 `i` 间独立同分布、每对为标准双变量 Gaussian，且 `corr(A_i,B_i)=rho`。选择 `J=argmax_i A_i` 后，

\[
\mathbb E[B_J]
=\rho\,\mathbb E[\max_i A_i]
\approx \rho\sqrt{2\log M}.
\]

证明只有两步：双变量 Gaussian 给出 `E[B_i|A_i]=rho A_i`；`J` 只由全部 `A` 决定，因此

\[
\mathbb E[B_J\mid A_1,\ldots,A_M]
=\rho A_J=\rho\max_i A_i.
\]

再取期望即可。极值近似给出最后一项。

含义：每个**预先固定**区域的确认响应均值都是 0，但如果先挑 `A` 最大的区域，再在与 `A` 相关的 `B` 上确认，平均会得到正响应；候选越多，假证据越强。若同一响应既选又验，`rho=1`，这是最坏情形。

这段 Gaussian 代数属于标准 winner's curse / post-selection inference，不是新数学；选择后推断的经典框架见 [Fithian, Sun & Taylor 2014](https://arxiv.org/abs/1410.2597)。**可能新颖的是下面的经验规律，而不是公式：**

> 在冻结 VLM 的局部 hallucination mitigation 中，negative-claim 上的“增强收益”可由 selector/evaluator 相关性 `rho` 与有效搜索规模共同预测；AGLA/SECOND/zoom/mask 等方法的 recall–FP 交换中有相当部分是 selection–reuse inflation，而非新增视觉证据。

如果跨模型、跨分辨率、跨 ontology 和跨局部方法都成立，并且完整流水线校准能消除该项同时保留真病灶增益，这才可能形成 ICLR 级“统一机制 + 简洁方法”。

### 3.2 反例与可证伪边界

1. **独立验证器**：若 `rho=0`，选择哪个区域都不改变 `E[B_J]`；此时搜索税对第二次确认没有该膨胀。
2. **反相关干预**：若 `rho<0`，选择最大 `A` 反而降低 `B`；公式预测相反方向。
3. **固定或平均区域**：方法不自适应选 max，而用预设肺野/全局平均，`M` 不进入同一机制。
4. **强相关 patch**：若所有 `A_i` 完全相同，`max A_i=A_1`，增加 patch 数不增加假证据。
5. **结构性伪影**：管线、文字标记或体位与 claim 相关时，null 响应不是零均值噪声；search calibration 只能校正“挑最大”而不能消除因果混杂。
6. **弥散真证据**：肺水肿等信号分布在大量 patch 上，max/scan 可能不如全局平均；过强搜索税会损害真实 diffuse findings。

这些不是 limitations 装饰，而是必须预注册的分层预测。若 `rho×search size` 不能预测 negative inflation，这个核心命题失败。

---

## 4. 为什么解析 `sqrt(2 log M)` 在相关 patch 下通常不够

### 4.1 哪部分仍然正确

- 用 `sqrt(2 log(M/alpha))` 的 union-bound 阈值，在每个变量确为 1-sub-Gaussian 时即便相关也控制 FWER；但通常很保守。
- `sqrt(2 log M)` 作为 iid Gaussian 最大值的一阶期望/位置近似，需要近似独立、同方差、尾部近 Gaussian；它不是通用 threshold。

### 4.2 相关性为何破坏简单公式

胸片 patch 与多尺度窗口高度重叠。举等相关 Gaussian：

\[
A_i=\sqrt{\gamma}U+\sqrt{1-\gamma}\epsilon_i.
\]

则

\[
\mathbb E\max_i A_i
=\sqrt{1-\gamma}\,\mathbb E\max_i\epsilon_i,
\]

而不是 `sqrt(2 log M)`。当 `gamma->1`，候选虽有 `M` 个，实际搜索溢价接近 0。

常用 participation ratio

\[
M_{\mathrm{eff}}=(\mathrm{tr}\,\Sigma)^2/\mathrm{tr}(\Sigma^2)
\]

只能作 heuristic。最大坐标的分布不只由协方差特征值决定，还取决于 eigenvectors、局部相关结构、异方差和非 Gaussian tails；两个相同 effective rank 的协方差可以有不同 max tail。因此把 `M` 机械换成 `M_eff` **不能给有效的 type-I guarantee**。GWAS 中也主要把 effective number of tests 当近似；dependency-aware permutation 更可靠。

### 4.3 推荐修正顺序

1. **首选：null-image split conformal。** 对每个 vote-0 calibration image 完整重跑 claim×scale×region 搜索，直接校准最终 `T`；图内依赖完全保留。代价是需要少量已知 null 图，且保证只在同分布下成立。
2. **次选：合法的 permutation/randomization。** 只有在 transformation 保持 null joint law 时才精确。随意打乱 patch 会破坏解剖与空间相关，左右镜像也不保持心脏/胃泡分布，均不合法。若使用 permutation，应按 patient-level label exchangeability 或经验证的 block/toroidal invariance；CXR 通常缺少后者。
3. **可解释近似：scale-stratified empirical quantiles。** 每个尺度单独估计 null max，再对“尺度间再选择”做第二层 joint calibration；不要各尺度校准后在测试图上挑最小 p 而不再付税。
4. **仅作低成本预筛：effective rank / Gaussian multiplier bootstrap。** 它们可估计有效搜索量和误差条，但正式结论需与 empirical null 对照。
5. **跨分辨率/ontology 可比的对象应是 p-value 或 null quantile residual，而不是 raw max。** 对配置 `g=(K,resolution,scales)` 定义

\[
E_g=T_g-Q_{1-\alpha}^{(0)}(T_g),
\]

或直接报告 `p_g`。若测试时还会从多个 `g` 中选最优，则校准 statistic 必须是 `max_g T_g`。

---

## 5. 最简 training-free 方法

暂不命名新算法，先用描述性名称：**Select–Validate–Calibrate (SVC)**。

1. **Select**：对固定 claim `c`，用冻结 VLM 的局部响应 `A` 在固定多尺度窗口族中选 `W_hat`。
2. **Validate**：用第二个冻结响应 `B` 评估 `W_hat`。优先使用不同 benign rendering / 不同 evidence operator，以降低 selector 与 validator 的共享噪声；但不能把它宣称为独立观测。
3. **Calibrate**：在 `c` 为 vote-0 的 dev 图上，完整重放前两步，得到 `T_j=B_{W_hat_j}`，用 split-conformal rank 得 `p_c`。
4. **Use, not suppress blindly**：CE 中只有 `p_c<=alpha` 才允许 local enhancement 改变 baseline positive commitment；否则回退原 greedy，不自动输出 negative。OE 中保持 positive claim 数 `K`，只按 calibrated evidence 做一换一 exchange，防止“少说即低 hallucination”。

它冻结所有模型权重，因此是 **training-free, calibration-only**；若不允许任何有标签 null calibration，就不能诚实给有限样本保证。此方法的 conformal 部分本身与 ConfLVLM/CEBC 碰撞，只有 selection–reuse 机制和 joint search-space law 经验证后才有论文新颖性。

---

## 6. 三个致死实验

### 实验 1：Null Search Expansion（核心现象门）

**数据**：VinDr vote-0，4 个 findings；Huatuo + Hulu；image-disjoint dev/test。  
**操控**：嵌套增加 region 数 `R in {16,64,256}`、尺度数 `S in {1,2,4}`、ontology 大小 `K in {1,4,8}`；语法、图像和输出预算固定。  
**记录**：raw `max A`、selected `B_hat`、`rho=corr(A_i,B_i)`、conformal p、最终 claim margin/FP。

**GO**：

- 两模型上 negative raw max 与 selected validation response 均随搜索空间显著增长；patient-bootstrap 95% CI 排除 0；
- 各配置的 inflation 至少方向上由 `rho E[max A]` 预测，且真实 selector 明显高于同面积随机 selector；
- 完整流水线 conformal 在 `alpha=.05` 下 empirical type-I CI 包含 `.05`，且不随 `K,R,S` 单调恶化。

若 raw response 不随搜索空间增长，或增长不传递到最终 margin/FP，则“视觉搜索税解释 hallucination”失败，关闭主线。

### 实验 2：True Evidence Survives the Tax（临床证据门）

**数据**：同 findings 的 vote-3 positives，使用 VinDr bbox；Nodule/Mass 与 Pleural Effusion/Consolidation 分层。  
**对照**：同面积平移框、随机框、patch-score permutation、final margin、finding identity。  
**指标**：calibrated statistic 对真值的增量 AUROC；选中窗口 bbox IoU/inside enrichment；病灶面积×aggregator interaction。

**GO**：

- `final margin + finding identity + calibrated local evidence` 比前两项增加至少 `0.02 AUROC`，image-bootstrap 95% CI 排除 0，两模型同向；
- 最大 calibrated window 的 bbox enrichment 显著优于同面积 control；
- focal findings 上 scan 优于 mean，而 diffuse findings 不出现 >1pp 明显损害。

若 tax 后没有信号，SVC 只是在安全地“不做事”，没有 mitigation 价值。

### 实验 3：Matched-Operating-Point Mitigation（结果门）

只有前两门全过才运行。CE 对比 greedy、raw local enhancement、SECOND/AGLA-style 方法、普通 threshold/temperature、SVC；OE 使用 fixed-K claim exchange。

**GO**：

- FP hallucination 相对 raw local enhancement/greedy 下降至少 20%；
- matched Yes-rate / fixed-K / matched length 下仍显著；
- clear-positive recall 下降不超过 1pp；
- 至少两个模型、focal 与 diffuse 多数 findings 复现；
- 在自然图像小物体基准上至少复现“search expansion -> raw FP 上升 -> calibrated FP 稳定”的机制规律。

若只通过减少 positive 数、缩短回答或拒答获益，判失败。

---

## 7. 新颖性与 ICLR Oral 判定

| 组成 | 是否新 | 审计结论 |
|---|---:|---|
| `sqrt(2 log M)` / scan penalty | 否 | 经典极值与 scan statistics |
| permutation / split conformal 校准 max | 否 | 标准 distribution-free inference |
| 用 detector/conformity score 过滤 VLM claim | 否 | ConfLVLM、CEBC 已覆盖 |
| 选局部、多尺度、再增强 VLM | 否 | AGLA、SECOND 等已覆盖 |
| **local selector 与 evaluator 复用导致 `rho × search premium` 的 VLM 规律** | 尚未检索到机制等价工作 | 值得用致死实验确认；未证实前不能声称新颖 |
| **该规律统一预测 resolution×ontology×method 的 FP，并由 full-pipeline calibration 消除** | 条件性新颖 | 若跨方法/模型/自然与医学成立，才接近 oral 级统一机制 |

当前判定：**值得跑 Experiment 1；不是 ICLR-ready。** 成功概率受本地历史负结果限制：普通 bbox erase、ROI response 和全局 visual incremental evidence 均偏弱。最诚实的策略是把 null search-expansion 作为唯一第一门；它失败即停，不再通过换 layer、换 top-k 或加小模型挽救。

# ICLR Oral adversarial story audit：从“小病灶”到“视觉搜索税”

日期：2026-08-12  
角色：苛刻 ICLR oral 审稿人  
结论先行：**当前没有已被实验支持的 oral-ready idea。唯一值得立即做致死实验的候选，是把“小病灶稀释”升级为“视觉搜索的统计代价”；其余两个候选只作竞争解释，不应先占 GPU。**

## 0. 先否定最容易讲、也最容易被拒的故事

“小病灶只占少数 patch，因此用 max/top-k/zoom 保留病灶”不是足够的新问题：

- [AGLA, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html) 已用 prompt-relevant local features 与 global features 联合缓解 hallucination；
- [SECOND, ICML 2025](https://proceedings.mlr.press/v267/park25c.html) 已做选择性、多尺度、object-centric 视觉增强；
- [Perception Magnifier, ACL 2026](https://aclanthology.org/2026.acl-long.2059/) 已在逐 token 解码时定位并放大相关区域；
- [Med-VCD](https://arxiv.org/abs/2512.01922) 已在医学 VLM 中选择视觉相关 token 并做稀疏 contrastive decoding；
- [IF-Prune, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_IF-Prune_Information-Flow_Guided_Token_Pruning_for_Efficient_Vision-Language_Models_CVPR_2026_paper.html) 甚至已使用小 VLM 和 information bottleneck 判断不重要 token。

因此，如果我们的算法最后可被概括为“找病灶 patch，再放大/保留/加权”，最多是医学适配，不是 oral 级洞察。

本地证据也不允许直接讲“小病灶已被找到但被平均掉”：

- pooled visual feature 虽有 AUROC 0.72--0.74，但控制 final margin 后的增量 NLL/Brier bootstrap CI 跨 0；
- 普通 bbox erasure 的 lesion drop 仅 0.029，95% CI `[-0.049, 0.109]`；
- ROI-vs-background 响应 AUC 0.545，95% CI `[0.462, 0.628]`；
- Nodule/Mass 的简单擦除甚至出现错误方向。

这些结果**没有排除 patch 尾部或空间联合统计**，但使其先验成功率偏低。必须先做 fatal test，不能先写故事。

---

## 排名 1：Search-Calibrated Visual Evidence

### 一句话贡献

> **VLM hallucination 不只是“没有看见小物体”，而是从大量 claim×region 候选中挑最大响应时产生的 winner's curse；视觉证据必须在支付搜索税后仍显著。**

直观例子：在一张胸片的 576 个 patch 中寻找“结节”，即使每个 patch 都只有随机噪声，也很容易找到一个看似很像结节的最大值。分辨率越高、候选病种越多、扫描尺度越多，最大的随机值会越大。直接 zoom 最显著区域会把这个随机赢家放大，而不一定放大真实病灶。

### 数学背景与需要验证的非平凡规律

设图像有 $n$ 个 patch，$X_{c,i}$ 表示第 $i$ 个 patch 对 claim $c$ 的标准化支持分数。

- 无病灶时 $H_0$：所有区域只有零均值噪声；
- 有病灶时 $H_1$：某个未知连通区域 $S$ 含 $k$ 个 patch，每个 patch 有弱均值偏移 $\mu>0$。

三种读法的差别是：

1. 全图平均的有效信噪比约为 $\mu k/\sqrt n$。病灶小的时候被全图稀释；
2. 单点最大值不用平均，但为了压过 $n$ 个噪声候选，单个 patch 必须强到约 $\sqrt{2\log n}$；
3. 空间 scan 对候选区域内分数求和并除以 $\sqrt{k}$，信号变成 $\mu\sqrt{k}$，同时按同尺度候选区域数 $M_k$ 支付约 $\sqrt{2\log M_k}$ 的搜索代价。

可使用的统计量为：

\[
T_c=\max_{k,W\in\mathcal W_k}
\left[
\frac{\sum_{i\in W}X_{c,i}}{\sqrt{|W|}}
-q_{k,\alpha}
\right],
\]

其中 $W$ 是候选连通窗口，$q_{k,\alpha}$ 是按窗口尺度校准的零假设阈值。经典 multiscale scan 已证明 scale-dependent calibration 可在大小尺度上接近最优，见 [Walther, 2010](https://arxiv.org/abs/1002.4770)；稀疏但不要求连通的对应理论是 [Donoho & Jin, 2004](https://doi.org/10.1214/009053604000000265) 的 Higher Criticism。

**这些经典定理本身不能算我们的贡献。** oral 级贡献必须是下面这条新的 VLM 规律：

> 局部增强方法的 FP 增益随其搜索空间有效大小增长，而经过尺度与 claim 数联合校准的证据在分辨率、候选 ontology 大小和模型间保持可比；这种“搜索税”解释现有方法的 recall--FP 交换。

若同时搜索 $K$ 个 claims 和 $M$ 个区域，一个最简单的零假设上界含有 $\log(KM/\alpha)$。这给出两个直接预测：增加分辨率或要求模型列更多异常，未经校准的最大响应会系统性变得更“自信”；真证据则应在扣除该搜索代价后仍为正。

### 为什么不只是普通 small-object detection

必须同时满足以下四点，否则审稿人会把它还原成 small-object/attention engineering：

1. **研究对象是选择偏差而非检测器精度。** 方法不重新训练 lesion detector，而审计冻结 VLM 在挑选局部证据后的统计有效性。
2. **预测适用于 focal 与 diffuse evidence。** 不是只对 nodule zoom；不同尺度阈值应同时覆盖小结节、大片实变和自然图像小对象。
3. **控制搜索空间大小。** 核心实验必须改变 region 数、claim 数和尺度数，并验证 raw max 的 FP 随搜索空间增长，而校准统计保持稳定。
4. **匹配报告工作点。** 所有缓解收益在相同 Yes-rate、相同 claim 数/回答长度下仍成立，不能靠少说或统一阴性。

### 与最近邻的实质差别

| 最近邻 | 它做什么 | 我们必须多出的东西 |
|---|---|---|
| AGLA / Perception Magnifier / SECOND | 选相关局部区域并增强 | 证明“选最大区域”本身产生可预测的 false evidence，并显式支付 search tax |
| Med-VCD / IF-Prune | 选择或删除 visual tokens | 不是重要性排序，而是跨尺度有效的零假设显著性；报告 FP 控制而非只报告平均准确率 |
| VIHD / HALP | 用内部响应检测 hallucination | claim×space 的局部替代假设、bbox 富集和搜索空间相变；不只是一个静态 probe |
| CEBC | 外部 detector + conformal minimal edit | 不依赖现成 detector 决定真值；解释局部选择为何制造 hallucination，并在 fixed-K 下纠正 |

### 致命风险

1. **最可能的失败：native patch score 根本没有病例级信号。** 当前 global incremental gate 已失败；局部也可能只剩位置/解剖先验。
2. 若只有 BioMedCLIP/小 detector 有效，故事会退化为“用小模型验证大模型”，与 CEBC/MARINE/AGLA 邻近。
3. 若 scan 只改善 Nodule/Mass、对 diffuse findings 和自然图像无效，它只是医学 small-object 工程。
4. union bound、scan、Higher Criticism 都是旧数学；没有 VLM-specific phase law，理论章节只是装饰。
5. 当前 bbox erasure 和 ROI 结果偏负，因此主观成功概率仅 **20--30%**，但这是尚未被精确实验关闭的概率，不是正证据。

### 48 小时致死实验

**阶段 A：现象门，n≈128--256，不改生成。**

- 数据：VinDr，至少包含 Nodule/Mass（小/局部）与 Pleural Effusion/Consolidation（较大/弥散）；image-disjoint dev/test。
- 模型：Huatuo 与 Hulu；每个 claim 提取 projector 前后 patch map。可以同时加 frozen BioMedCLIP map，但必须独立报告。
- 统计：mean、max、top-k、Higher-Criticism、multiscale scan；阈值仅在 dev 的 vote-0 图上冻结。
- 控制：final margin、finding identity、病灶面积；bbox 富集需优于同面积平移框和 patch-score permutation。

**GO 必须全部满足：**

1. scan/HC 在 final margin + finding identity 上增加至少 0.02 AUROC，image-bootstrap 95% CI 排除 0，两个模型同向；
2. scan 最大窗口在 bbox 内的富集显著优于同面积随机/平移窗口；
3. 存在预注册 area×aggregator crossover：小病灶 scan/HC 优于 mean，大/弥散 finding 不受损；
4. 人为扩大候选窗口数 $M$ 后，negative 图的 raw max/FP 上升，而 scale-calibrated $T_c$ 基本不变；
5. 不是 Yes-rate 漂移：比较的是连续排序与 matched operating point。

任一核心项失败即关闭，不用换层、换 top-k 或重新调阈值救故事。

**阶段 B：只有 A 通过才做生成缓解。**

- 在 CE 中用 $T_c$ 限制 positive commitment；在 OE 中草稿 claim + ontology 形成候选集，但固定 positive claim 数 $K$，只做 weak↔strong exchange；
- 对比 greedy、temperature/threshold、PM/SECOND/AGLA-style local enhancement、CEBC-style verifier；
- 目标：FP hallucination 相对下降至少 20%，matched-K recall/omission 不下降，跨两模型和至少一个自然 VLM benchmark 复现。

### Oral 判定

当前：**NO**。  
若 A 全过但只在 VinDr 有效：可成医学/机制论文，不到 oral。  
若同时证明 claim 数×视觉搜索空间的相变、跨医学与自然场景复现、并在 matched operating point 下解释多种 local mitigation 的 FP：**有 oral 候选资格**。

---

## 排名 2：Open Generation as Multiple Testing

### 一句话贡献

> **开放式报告不是一串 token，而是在大量临床假设中选择一个集合；hallucination 是被选择 claims 的 false discovery，而候选集合越大，未经校正的假阳性越多。**

### 简洁方法与数学

草稿报告被拆成 claims $c_1,\ldots,c_K$。每个 claim 得到一个经过 calibration 的视觉证据 $e_c$ 或 p-value，再用 e-BH/BH 类集合选择规则控制报告中 false claims 的期望比例，而不是逐 claim 用同一个阈值。固定最终 claim 数或做一换一交换，防止“少说即低幻觉”。

这里的非平凡问题不是 BH 公式，而是：**claims 由同一图像和同一模型自适应生成，彼此强依赖，且候选集合大小由 prompt 决定。** 如果不能给这种 adaptive/dependent selection 下的有效性证明，数学仍只是套用。

### 最近邻与碰撞

- [ConfLVLM, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.576/) 已对 LVLM factuality 给出 conformal risk guarantee；
- [CEBC, ACL 2026](https://aclanthology.org/2026.acl-long.2142/) 已做 conformal evidence-bounded minimal editing；
- [Principled Detection via Multiple Testing, 2025](https://arxiv.org/abs/2508.18473) 已把 hallucination detection 表述为 multiple testing；
- 2026 的 post-hoc conformal/e-value selection 已覆盖 adaptive operating-point 思想。

所以它只有在“**同图 claim 集的依赖结构 + fixed-K 生成**”产生新定理和强现象时才有空间。否则是 conformal/BH 的医学包装。

### 致命风险

1. 当前没有医生 review，OE 的真实 false claim 集合不能只由自动 judge 定义；
2. 最简单做法会靠删除 claims 获益；
3. CEBC/ConfLVLM 碰撞极强，数学新颖性门槛高；
4. 本地 abnormality-focused prompt 的 116.6 vs 24 tokens 只证明长度变化，尚未证明 matched-length clinical FP 增长。

### 48 小时致死实验

- 对同一批 VinDr 图构造 ontology 大小 $K=4,8,16,32$ 的 matched-syntax prompts；固定真实 finding 和输出 claim 数；
- 检验 FP 是否随 $\log K$ 单调增长，并在控制 token 长度、prompt-copy、finding prevalence 后仍成立；
- 用现有 reader labels 构造 claim-level truth，dev calibration、image-disjoint test；
- 只有“raw FP 随 K 增长 + multiplicity calibration 在 fixed-K 下同时降 FP 且不降 recall”跨两模型成立，才继续。

**排序理由：**比普通 calibration 更贴近 OE，但与现有 conformal/multiple-testing 文献碰撞太近；先用 CPU/短生成验证现象，不应抢占主线 GPU。

---

## 排名 3：The Observation Frontier（暂缓/近似否决）

### 一句话贡献

> **同一观测上的任何 decoding 都不能创造缺失的临床信息；只有获得真正的新视角/新模态，才可能越过该观测的 Bayes risk 下界。**

这可用 Blackwell order 解释：如果新观测 $X'$ 能通过某个随机“降质”过程变成旧观测 $X$，则 $X'$ 至少和 $X$ 一样有信息；只基于 $X$ 的后处理不能普遍优于利用 $X'$ 的最优决策。这个数据处理结论正确但标准，不是理论贡献。

### 为什么暂缓

[Look Again Before You Abstain / BCEA, 2026](https://arxiv.org/abs/2606.16667) 已经做 budgeted conformal evidence acquisition、顺序自适应 crop、small-object 分析，并证明 acquisition 改善 coverage 当且仅当改善 ROC；[Perception Magnifier](https://aclanthology.org/2026.acl-long.2059/) 也已占据 adaptive zoom。只有“真实新传感观测”（lateral view、prior study、另一 MRI sequence）而非同图 crop 能区分。

### 48 小时致死实验

- IU-Xray/MIMIC 多图 study：第一张图、同图 beam/重复 decoding、第二张真实 view，匹配推理 FLOPs；
- 使用 study-level report labels，只称 benchmark proxy；比较第二 view 是否在 first-view error 子集把 AUROC/BAcc 提高至少 0.03，且明显超过同图重复；
- 若收益仅来自 ensemble/长度或报告 label 本来由多图共同定义，则实验不可识别，方向关闭。

**排序理由：**概念重要但几乎被 BCEA 占据，而且现有标注无法可靠说明哪条 claim 是第二 view 独有；不适合作为当前主赌注。

---

## 最终排序与决策

| 排名 | 故事 | 当前创新性 | 当前证据 | 决策 |
|---:|---|---|---|---|
| 1 | Search-Calibrated Visual Evidence | 条件较高：若证实 search-tax phase law | 尚无正证据，且历史局部结果偏负 | **立即做唯一 fatal pilot** |
| 2 | Open Generation as Multiple Testing | 中等偏低，强 conformal/CEBC 碰撞 | prompt 长度现象不是临床 FP 证据 | CPU/短生成竞争实验 |
| 3 | Genuine Observation Frontier | 概念好但 BCEA 近乎直接覆盖 | 有多视图数据，因果标签弱 | 暂缓，不占主线资源 |

### 对主项目的严格建议

1. 不要把候选 1 预先命名成一个新算法；先验证“raw local selection 随搜索空间制造 FP”这一自然现象。
2. 不要以经典 scan/HC 定理充当原创理论；理论贡献必须围绕 VLM 的 claim×space adaptive search，至少给出一个旧 local-enhancement 方法无法表达的预测。
3. 不要只做 nodule。若没有 focal↔diffuse、medical↔natural 的统一 phase diagram，审稿人会正确地归类为 small-object detection。
4. 不要用外部 detector 的准确性冒充 VLM 内部 evidence。native patch map 必须先过增量、定位和 search-size 三门；小模型只能作为对照或部署变体。
5. 第一个 128--256 样本 gate 失败后，立即关闭整个局部稀疏路线并恢复 baseline 队列；不再通过换 layer/top-k 延长寿命。

**总判断：**最有希望的 oral 故事不是“更精细地找病灶”，而是“所有局部增强方法都忽略了视觉搜索的统计代价；真正的视觉证据是在支付搜索税后仍然成立的区域性信号”。它简洁、通用、能产生可证伪数学预测，也能解释 LET/局部增强常见的 recall 上升与 FP 增加。但在 fatal pilot 通过之前，它仍只是一个高质量假设，不是已完成的 ICLR idea。

# C38 Pareto Claim Exchange：公式级文献碰撞审计

审计日期：2026-08-13  
范围：2022--2026 顶会、期刊与 arXiv 为主；为判断数学原创性，补充必要的更早多目标优化/社会选择文献。  
审计对象：固定 `K` 个阳性 claim 的开放生成草稿，以 VLM 病例分数 `a` 和冻结小视觉专家分数 `b` 形成二维偏序；仅当遗漏 claim `j` 同时满足 `a_j>a_i` 与 `b_j>b_i` 时，以 `j` 一换一替换草稿 claim `i`。

## 1. 先把方法还原成标准数学对象

令 claim ontology 为 `C`，草稿阳性集合为 `S_0 subset C`，且 `|S_0|=K`。每个 claim 对应二维向量

```text
s(c) = (a_c, b_c).
```

定义严格 Pareto 支配

```text
j \succ i  iff  a_j > a_i  and  b_j > b_i.
```

一次 C38 更新为

```text
S <- S - {i} + {j},  i in S, j notin S, j \succ i.
```

因此，C38 精确等价于：**在 uniform matroid `U_{K,|C|}` 的 bases（所有大小为 `K` 的集合）上，用 Pareto 支配定义 1-exchange 邻域，并做只接受双目标严格改进的 Pareto local search。**

它的两个主要性质都不是新定理：

1. 对 `a`、`b` 各自任意严格单调重标定不变，是 ordinal/Pareto rule 的定义性性质；
2. 每次交换后 `sum_{c in S} rank_a(c)` 与 `sum_{c in S} rank_b(c)` 同时严格上升，所以有限候选集上无环并终止，是严格偏序 local search 的直接推论；固定 `K` 则只是 base exchange 保持集合基数。

终点仅保证：不存在一个未选 claim 同时严格支配某个已选 claim。它**不保证**全局最优、临床正确、唯一或最大纠错量。

## 2. 公式级直接碰撞

### 2.1 多目标优化：直接数学碰撞

- Pareto dominance、非支配筛选和 exchange-based Pareto local search 是成熟的 multi-objective combinatorial optimization 对象。2025 年 IEEE TEVC 的 [Targeted Pareto Optimization for Subset Selection with Monotone Objective Function and Cardinality Constraint](https://doi.org/10.1109/TEVC.2024.3431928) 已明确研究带 cardinality constraint 的 Pareto subset selection。
- 2025 年 GECCO 的 [Why Dominance Is Not Enough](https://doi.org/10.1145/3712256.3726414) 证明只依赖 dominance relation 的黑盒多目标算法，在大量候选互不可比时可需指数时间；实际算法需要目标值或 diversity 等额外信息。这正是 C38 在两路分数冲突时的大量“不可交换”风险。
- 2025 年的 [Picking a Representative Set of Solutions in Multiobjective Optimization](https://arxiv.org/abs/2511.10716) 把固定大小 Pareto subset selection 重新表述为 multiwinner voting，并显示固定基数集合仍需额外 quality measure 才能从大量 Pareto 候选中选出代表集。

结论：**“二维 Pareto 支配 + 固定 K + 一换一局部改进”在公式层面是直接碰撞，不可作为数学创新。**

### 2.2 社会选择：几乎逐字对应的邻域对象

[Pareto Optimality in Approval-Based Multiwinner Voting](https://arxiv.org/abs/2605.30490) 研究固定大小 `k` 的 committee，并定义 Single Dominance Only：若 committee 内没有候选被 committee 外候选 Pareto 支配，则该 committee 为 Pareto optimal；论文还研究保持固定 `k`、一次替换一个 candidate 的 Pareto reconfiguration graph。

映射如下：

| C38 | Multiwinner voting |
|---|---|
| VLM 与视觉专家 | 两个 voters / preference sources |
| clinical claims | candidates |
| 固定阳性数 `K` | committee size `k` |
| 未选 `j` 同时优于已选 `i` | outside candidate Pareto-dominates inside candidate |
| `S-{i}+{j}` | one-candidate committee reconfiguration |

两者的 score 类型并非完全相同（C38 是两路 total/weak rankings，该论文主设定是 approval preferences），但**固定基数、单项支配、逐项替换、终点无外部候选支配内部候选**这一数学结构已经被直接研究。

## 3. VLM/医学领域的最近邻

未检索到一篇同时使用“VLM claim ranking + 小专家 ranking + 严格 Pareto 支配 + 固定 K 一换一”的论文。但四个组成部分的应用空间已经拥挤。

| 工作 | 小/外部视觉专家 | claim/object 级 | 直接纠错 | 保持覆盖/长度 | 与 C38 的差别 |
|---|---:|---:|---:|---:|---|
| [Visual Evidence Prompting, ACL 2025](https://aclanthology.org/2025.acl-long.205/) | 是 | object/relation evidence | 通过 prompt 重生成 | 未固定 K | 把 detector/scene-graph 输出符号化为 prompt；不是序数交换 |
| [MARINE, 2024](https://arxiv.org/abs/2402.08680) | 是 | object grounding | decoding 中纠偏 | 强调 detailedness | 以 classifier-free guidance 融入专家特征；需 cardinal score/weight |
| [Pelican, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.470/) | 多外部工具 | sub-claim | verification 后修正 | 未固定 K | claim decomposition + program-of-thought 工具验证 |
| [Kestrel, 2026](https://arxiv.org/abs/2603.16664) | grounding agent | claim | evidence-verified self-refinement | 保守修订 | 结构化证据、多轮 judge/refinement；非 Pareto、非固定 K |
| [CoEV, 2026](https://arxiv.org/abs/2606.18609) | 视觉 evidence region | 医学 assertion | 是 | 未固定 K | 医学报告中 training-free claim 验证与自动 refinement；与目标场景最近 |
| [Verifier-Guided Decoding, 2026](https://arxiv.org/abs/2607.27823) | lightweight verifier | emerging object mention | rollback、抑制同义词、重生成 | 报告 99.6% grounded-object coverage 且不缩短 caption | 与“选择性纠正且不靠缩短”最接近，但 verifier 来自内部 grounding signature，不补 omitted claim |
| [CCS, 2026](https://arxiv.org/abs/2605.30131) | image-report multimodal embedder | report 级 | candidate selection | 选完整报告 | 外部医学视觉 utility + inference-time selection；不是 claim exchange |
| [CEBC, ACL 2026](https://aclanthology.org/2026.acl-long.2142/) | detector evidence | object mention | evidence-bounded selection | quality-aware | 使用 conformal threshold，属于校准路线；不是无权重 Pareto |

补充近邻：Woodpecker/LURE 早已使用外部 object detector/VQA/LLM 对草稿进行验证和重写；VisionWeaver（Findings EMNLP 2025）训练 routing network 聚合多个 specialized visual encoders。它们不是公式等价，但会让“用小视觉模型纠正大 VLM”本身不再具有新颖性。

## 4. 严格碰撞裁决

### 裁决 A：是否存在完全等价的医学 VLM 方法？

在本次记录的检索式与论文中，**未检索到四要素完全等价的实现**：两路独立病例 ranking、strict Pareto veto、固定 `K`、false-positive 与 omitted-positive 一对一交换。

因此不是“同领域 direct implementation collision”。

### 裁决 B：核心算法/数学是否新？

**否。公式层面直接碰撞。** 它是 fixed-cardinality Pareto 1-exchange / multiwinner committee reconfiguration 的直接应用。独立单调变换不变性、无权重、固定 `K` 和有限终止均是标准 ordinal Pareto 结构的自然性质。

### 裁决 C：剩余真实 delta 是什么？

只剩一个窄的应用/现象 delta：

> 医学开放生成是否经常产生一种特殊配对错误——同一未生成真实 finding，在 VLM 的 ontology score 和独立视觉专家 score 上都严格支配一个已生成假 finding；利用该配对能否在完全固定 claim 数下同时降低 fabrication 与 omission？

这个 delta 若被两模型、多数据集强实验证实，是有价值的**经验规律与简单应用原语**；但它不是新的 Pareto 数学。

### 新颖性上限

- 当前作为算法论文：**低到中**。最容易被审稿人概括为 “VEP/medical verifier 外接专家之后做 conservative Pareto rank fusion”。
- 作为医学工程方法：若效果强、速度快、固定内容优势明确，有可发表价值。
- 作为 ICLR Oral 核心：**不足**。必须另外发现一个非标准机制或定理；不能把标准 Pareto 性质包装为理论贡献。
- 若论文主贡献退化为“common-dominance error 很多”，则更像 empirical finding；而用户明确不希望以评测发现替代创新方法。

## 5. C38 自身尚未解决的数学缺口

1. **非唯一与次序依赖。** `K=1` 时，若初始 `i=(0,0)`，而 `j=(2,1)`、`k=(1,2)`，`j` 与 `k` 都支配 `i`，但二者互不可比；先选谁就得到谁。
2. **大量 incomparability。** 两专家出现互补错误本来是采用双专家的理由，却也正好使严格 Pareto rule 无法行动。GECCO 2025 已给出 dominance-only information 不足的理论警告。
3. **没有 accuracy guarantee。** 两个有相关偏差的模型可一致偏好同一个错误 claim；Pareto unanimity 只保证两路 score 同升，不保证临床真值改善。
4. **不是真正 open-vocabulary。** 要得到遗漏 claim `j` 的 `a_j,b_j`，必须先扫描固定 ontology 或构造候选器；算法通用性受候选召回上限约束。
5. **VLM 自洽性前提可疑。** 草稿既然生成了 `i` 而未生成 `j`，却要求独立 claim score 满足 `a_j>a_i`；若这一现象不常见，C38 将几乎不交换。
6. **多个交换冲突未定义。** 一个 `j` 可支配多个 `i`，多个 `j` 也可竞争一个 `i`；必须冻结 maximum-cardinality matching、Pareto-layer tie-break 或确定性次序。任何 cardinal tie-break 都可能破坏“完全无权重”的叙事。

## 6. 必须加入的 baseline

### 6.1 融合/集合选择 baseline

1. Native draft：不修改。
2. VLM-only top-`K`。
3. Expert-only top-`K`。
4. 归一化后加权和：z-score、min-max、temperature-scaled logits；权重只在 dev 冻结。
5. Product-of-Experts / logit sum。
6. Borda rank sum。
7. Reciprocal Rank Fusion。
8. Rank product / geometric mean rank。
9. 两路 top-`K` intersection，剩余位置由 VLM ranking 填充；这是最简单的 calibration-free unanimity baseline。
10. Pareto non-dominated sorting + VLM tie-break。
11. Iterated first-valid 1-exchange（原 seed）与 maximum-cardinality dominance matching；必须报告不同交换顺序的方差。
12. Random fixed-`K` exchange、expert-shuffled exchange。
13. Oracle fixed-`K` exchange，给出候选/交换上限。

### 6.2 VLM hallucination baseline

最低限度应比较：VEP-style expert prompting、MARINE 或同义 expert-guided decoding、Pelican/Woodpecker 类 claim verification；医学报告需纳入 CoEV（若代码可用，否则明确 `N/A`）；开放生成需纳入 VGD（若其正式代码可复现）或至少逐项说明设置差异。CCS 是报告级 Best-of-N/外部视觉 utility 的重要独立基线。

### 6.3 决定“不是专家本身更强”的控制

- expert-only top-`K` 必须不优于 C38；
- 用同一专家随机单调变换，验证实现级 ordinal invariance；
- 打乱病例间 expert 分数，增益应消失；
- 使用与 VLM 同视觉塔的“非独立专家”，检验互补性是否必要；
- 分别报告 `both-correct / VLM-only-correct / expert-only-correct / both-wrong`，尤其是两路一致错误；
- 所有比较固定 `K`，禁止因少说、删 claim、拒答或缩短文本获益。

## 7. 审计结论

**C38 不因同领域存在完全相同论文而必须立即关闭，但不能以当前数学和叙事声称 ICLR 级创新。**

最准确的标签是：

```text
No exact domain implementation retrieved;
direct mathematical collision + crowded system-level neighborhood;
remaining delta is a narrow medical fixed-content correction setting.
```

是否值得做 L0 CPU 致死实验，取决于成本：值得，因为它能很快确认“共同支配错误”是否存在；但即使 L0/L1 效果通过，也只能证明这个简单应用有效，不能自动提升其数学原创性。若目标仍是 ICLR Oral，C38 更适合作为强 baseline 或方法组件，而非最终核心 idea。

## 8. 检索记录

核心检索组合包括：

- `Pareto dominance claim replacement fixed cardinality vision language hallucination mitigation`
- `multi-objective Pareto rank fusion vision language model hallucination visual expert`
- `fixed-cardinality set correction Pareto exchange predictions two classifiers`
- `medical vision language hallucination small visual expert evidence claim correction`
- `Pareto local search fixed cardinality set selection 1-exchange`
- `multiobjective combinatorial optimization uniform matroid Pareto exchange`
- `ordinal rank aggregation Pareto dominance monotonic transformation invariance`
- `VLM hallucination post-hoc correction claim verifier visual expert fixed K`
- `small visual expert hallucination mitigation VLM object detector claims correction`
- `radiology report consensus selection multiple models claim rank fusion`

所有承载结论的工作均用论文主页、ACL/CVF/ACM/IEEE 页面或 arXiv 原始记录核对标题、作者/年份及摘要；搜索摘要未能确认的细节未作为碰撞依据。

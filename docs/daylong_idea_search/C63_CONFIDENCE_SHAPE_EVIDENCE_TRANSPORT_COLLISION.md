# C63 — Confidence-Shape Preserving Evidence Transport 审计

日期：2026-08-13  
资源边界：数学、代码与本地缓存审计；未使用 GPU，未修改 baseline 队列。

## 裁决

> **严格 NO-GO：完整版本是 hard evidence-rank takeover；受限版本是 top-k reranking / linear assignment。**

候选保留冻结 VLM 每一步概率向量 `p` 的概率质量多重集，只按视觉/专科证据分数 `s`
重新分配这些质量。它确实精确保留所有 Rényi entropy、最大概率和排序后 top-k mass；但这些
量不关心“哪一个 token 得到哪一个概率”。Hardy--Littlewood rearrangement 直接给出：最大
证据期望的解把最大概率交给最大 `s`，所以 greedy token 完全由 `argmax s` 决定，与 VLM 的
概率排序无关。

因此该操作不是“温和地运输证据”，而是保留大模型置信度**数值外形**、同时把语义决策权
交给 evidence ranker。若只在 VLM top-k 内做，它精确退化为 top-k auxiliary-score reranking；
若加入身份保持或运输代价，则成为标准 regularized assignment / optimal transport。没有剩余的
新解码原语。

## 1. 冻结候选

对当前 prefix 和图像，冻结 VLM 给出 vocabulary 分布

```text
p = (p_1,...,p_V),  p_i >= 0, sum_i p_i = 1.
```

视觉分支或小专家给每个 token 一个证据排序 `s_i`。候选只允许置换 `p` 的坐标：

```text
P(p) = {P p : P is a V x V permutation matrix}.
```

然后解

```text
q* = argmax_{q in P(p)} <q,s>.                    (1)
```

它希望同时得到：

1. evidence 高的 token 获得更多质量；
2. `q` 与 `p` 有相同 probability multiset；
3. 因而不改变 confidence shape。

## 2. rearrangement inequality 给出精确闭式解

把概率从大到小写成

```text
p_(1) >= p_(2) >= ... >= p_(V),
```

把证据分数从大到小排列的 token 写成

```text
s_[1] >= s_[2] >= ... >= s_[V].
```

Hardy--Littlewood--Pólya rearrangement inequality 说明式 (1) 的最优解为

```text
q*_[r] = p_(r),  r=1,...,V.                      (2)
```

即两种排序 comonotone 对齐。若 `p` 与 `s` 都无 ties，解唯一；有 ties 时只在 tied blocks
内不唯一。这也是一维 optimal transport 的 monotone coupling，或对 permutation matrix 的
线性 assignment。

### 2.1 真正保留了什么

因为 `q*` 只是 `p` 的坐标置换，对任意只依赖数值多重集的 symmetric functional `F`：

```text
F(q*) = F(p).
```

所以以下量确实逐步相等：

- Shannon / Rényi / Tsallis entropy；
- `max_i q_i = max_i p_i`；
- 排序后的 cumulative top-k mass；
- 任意 `l_r` norm 和任意 Schur-symmetric confidence statistic。

这是置换的直接不变量，不是新的置信度定理。

### 2.2 greedy 退化为 hard rank takeover

式 (2) 把最大概率 `p_(1)` 交给最大证据 token，因此

```text
argmax_i q*_i = argmax_i s_i.                    (3)
```

`p` 只贡献“赢家拿到多大的概率”，不参与“谁赢”。所以对 greedy decoding，完整算法等价于：

```text
直接选择 evidence ranker 的 top-1 token。
```

这比 logit fusion 更激进，不是保守更新。

### 2.3 只使用 rank，完全丢弃证据幅度

对任意严格单调函数 `g`：

```text
q*(p,s) = q*(p,g(s)).                             (4)
```

证据差 `10^-9` 与 `10^3` 产生同一个排列。若 top-2 evidence 在任意小扰动下交换，分配给
两者的概率会从 `(p_(1),p_(2))` 突然交换，total variation 跳变 `|p_(1)-p_(2)|`。因此它对
接近 tie 的噪声不连续，也没有“只有强反证才改”的安全机制。

### 2.4 confidence shape 不等于 calibration

简单例子：

```text
p(correct)=0.9, p(wrong)=0.1,
but s(wrong)>s(correct).
```

运输后：

```text
q(correct)=0.1, q(wrong)=0.9.
```

entropy、max-confidence 和 top-1 mass 全部不变，但一个校准的高置信正确预测被变成同样高
置信的错误预测。Calibration 关心概率与事件身份/正确性的配对；置换只保留概率直方图，不能
保留 semantic calibration。

### 2.5 stepwise 不变量不传到完整序列

即使当前 prefix 上 `H(q_t)=H(p_t)`，一旦式 (3) 选择不同 token，下一步进入另一个 prefix：

```text
p_{t+1}(. | y_<t, y_t^q) != p_{t+1}(. | y_<t, y_t^p).
```

所以完整报告的 sequence entropy、长度、claim 数、top-k mass 和置信度分布没有保持保证。
方法只能声称“访问到的每一个 prefix 上，当前 vocabulary 概率多重集保持”，不能把它外推成
整段生成的 confidence preservation。

## 3. 三种可实现版本都退回已有对象

### A. 全 vocabulary permutation

视觉/专科专家必须给语法词、标点、subword、医学 finding 和属性共数万 token 排一个全序。
小医学模型通常只有 7--14 个 finding 分数，没有“the / left / ##tion / punctuation”的视觉
证据。若仍全排列，高概率语法质量可被搬给任意低频医学 token，流畅性与 tokenizer 语义失控。

### B. 只在 VLM top-k 内排列

设 `T_k` 为 VLM top-k candidates。在 `T_k` 内解式 (1) 后：

```text
argmax_{i in T_k} q_i = argmax_{i in T_k} s_i.
```

这正是 top-k reranking：VLM 负责候选 recall，auxiliary score 决定 winner。Contrastive Search
已经在 top-k 中按 degeneration auxiliary score 重排；把 score 换成视觉证据不改变计算原语。

### C. 加入“不要离原分布太远”的代价

若改成

```text
max_P <Pp,s> - lambda C(Pp,p),                    (5)
```

它是带身份/运输代价的 regularized linear assignment。允许 doubly-stochastic coupling 时则是
Birkhoff polytope 上的 OT；线性目标最优点仍是 permutation，熵/距离正则只是标准 OT 变体。
这会引入 `lambda` 和 cost 设计，也不再保持“只由 rearrangement 唯一决定”的简洁性。

若只交换 evidence 冲突的一对 token，则是 pairwise rank swap；若在完整候选 report 上做，
则是 sequence reranking / MBR。换粒度不产生新对象。

## 4. 与现有 decoding 的碰撞

| 已有对象 | 已覆盖部分 | 与 C63 的关系 |
|---|---|---|
| Hardy--Littlewood--Pólya rearrangement inequality | 两个排序同序配对最大化内积 | 式 (2) 的完整数学核心 |
| Linear assignment / Birkhoff--von Neumann | 在 permutation/doubly-stochastic matrices 上最大化线性收益 | 式 (1)/(5) 的优化身份 |
| Monotone coupling / 1-D optimal transport | quantile/comonotone matching | “evidence transport”的精确 OT 名称 |
| Contrastive Search | VLM top-k candidates 由辅助表示分数重排 | 受限实现的直接计算碰撞 |
| RankGen / sequence reranking | 用独立 ranking score 选择 LM 候选 | claim/report 粒度版本碰撞 |
| FUDGE / GeDi / DExperts | auxiliary discriminator/expert 改变 token 选择 | 若把 hard rank 平滑化即回到这些 guidance 家族 |
| VCD, CVPR 2024 | 原图与失真图的视觉差异改变 token 排序 | `s` 来自 distorted-image contrast 时只是更硬的 VCD |
| Clinical-event guidance / WFST | 对完整 event 而非 subtoken 施加专家证据 | 解决多 token claim 后退回 event constrained decoding |

相关公开来源：

- Hardy, Littlewood & Pólya, *Inequalities*, rearrangement inequality；
- Peyré & Cuturi, *Computational Optimal Transport*: <https://arxiv.org/abs/1803.00567>；
- Su et al., *A Contrastive Framework for Neural Text Generation*:
  <https://arxiv.org/abs/2202.06417>；
- Krishna et al., *RankGen*: <https://arxiv.org/abs/2205.09726>；
- FUDGE: <https://arxiv.org/abs/2104.05218>；
- GeDi: <https://aclanthology.org/2021.findings-emnlp.424/>；
- VCD: <https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html>。

没有检索到“保留 probability multiset”这一精确命名的 VLM 方法，但这不留下方法新颖性：
其闭式解是经典 rearrangement，而其 greedy 行为精确等于 auxiliary-score reranking。

## 5. 本地效力上限

### 5.1 Hulu VCD 改动已经显示 ranker 不够可靠

现有 Hulu VCD 配对结果中，VCD 相对 native 改变 `78` 个决定：

```text
fix = 32, harm = 46, net = -14.
```

即发生改动时，只有 `32/78 = 41.0%` 是正确修复，`59.0%` 是伤害。C63 若使用同一视觉
contrast ranking `s`，式 (3) 不会改善“该改谁”的选择：它只是把 ranker top-1 赋予原模型
最大置信度。entropy preservation 与 fix/harm 身份无关，因此不能把 `32/46` 反转。

若增加 gate 只在某些病例启用，就需要识别 fix 与 harm；这重新变成 calibration/error detection，
而不是 shape-preserving transport 自带的性质。

### 5.2 XRV specialist 不能提供通用 token ranking

在 840 个 image-disjoint VinDr confirmation claims/model 上：

- Huatuo：加入 XRV 后 macro AUROC `.7667 -> .8264`，增量 `+.0598`；
- Hulu：`.8606 -> .8708`，仅 `+.0102`，低于预注册 `.02`，Brier 改善 CI 跨 0；
- one-bit veto：两模型均仅移除约 `17.4%` FP，同时误伤 `1.52%/2.33%` TP；
- 固定 K transactional replacement 在小缓存中两模型均 `0` 次触发、`0` 收益。

所以 specialist `s` 在弱 Huatuo 上有条件增量，在更强 Hulu 上没有足够普遍的病例级新信息。
C63 的排列不会创造增量，只会把 specialist 排序提升为更大的概率质量。更激进的控制不能弥补
ranker 的错误。

### 5.3 开放生成的 claim/token mismatch

XRV 只能排列少数 finding，不能排列：

- finding 的多种 surface forms 与 subtokens；
- polarity、uncertainty、location、size 等 attributes；
- function words 和句法选择；
- ontology 外的新 claim。

若先解析“当前正在生成哪个 clinical event”，再只在 event 词表内搬运，就需要 event automaton
或 ontology mapping，退回 C63 clinical-event guidance / FUDGE / WFST 分支。若在生成后对完整
claims 排列，是普通 fixed-K reranking，而且本地 C44/C47/C57 已经失败。

## 6. 是否存在可保留的新性质

唯一可验证且表述准确的性质是：

> 在固定 prefix 的 vocabulary simplex 上，C63 是在 `p` 的 permutation orbit 中最大化
> 线性 evidence utility 的 comonotone coupling，因此保留所有 symmetric confidence
> functionals。

但这不是新 theorem，也不是安全性保证。它同时伴随更关键的反性质：

```text
greedy identity = evidence top-1,
evidence magnitude is discarded,
semantic calibration is not preserved,
sequence-level confidence is not preserved.
```

这些反性质使“confidence-shape preserving”更像表面不变量，而不是 hallucination mitigation
所需的临床约束。

## 7. 最终 fail-closed 判定

| 门 | 结果 |
|---|---|
| 是否精确保留 stepwise entropy / max / sorted top-k mass | **是** |
| 是否保留 semantic calibration | **否** |
| greedy 是否仍综合使用 `p` 与 `s` | **否；完全等于 `argmax s`** |
| 是否利用 evidence magnitude | **否；仅使用 rank** |
| 是否区别于 top-k reranking / assignment | **否** |
| 是否适配开放多 token clinical claims | **否，需 event constraint/codebook** |
| 本地 evidence ranker 是否跨模型可靠 | **否；VCD 32 fix / 46 harm，XRV Hulu 增量仅 .0102** |
| 是否值得占 baseline GPU | **否** |

最终结论：

```text
NO-GO AS A NOVEL MITIGATION PRIMITIVE.
Preserving the multiset of confidence values does not preserve which semantic
event deserves confidence; the proposed optimum is exactly hard reranking.
```

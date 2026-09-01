# C63 — Area / Measure Attention for Visual Refinement：严格审计

> 日期：2026-08-13  
> 裁决：**作为 ICLR Oral 主方法 NO-GO；作为多分辨率输入的正确性单元测试/工程 baseline 可保留。**  
> 本审计只读取文献、代码和已有结果，没有占用 baseline GPU。

## 1. 候选及其最强正确版本

标准 attention 对 query `q` 和视觉 tokens `(k_j,v_j)` 计算

\[
A(q)=\frac{\sum_j e^{s(q,k_j)}v_j}{\sum_j e^{s(q,k_j)}}.
\]

候选认为不同 token 代表的图像面积不同，令 `w_j>0` 为 coverage mass，把 logit 改为

\[
A_\mu(q)=
\frac{\sum_j w_j e^{s(q,k_j)}v_j}
{\sum_j w_j e^{s(q,k_j)}}
=\operatorname{softmax}_j(s(q,k_j)+\log w_j)V.
\]

直观解释：attention 不再把每个 token 当作一张等权“选票”，而把它看成图像积分中的一个带面积采样点。

若 token `j` 被细分成 children `j1,...,jr`，且

\[
k_{ja}=k_j,\qquad v_{ja}=v_j,\qquad
\sum_a w_{ja}=w_j,
\]

则分子、分母都完全不变，所以 `A_mu` 不变。重复同一个 token `r` 次并给每份 `w_j/r` 也是同一个结论。

对于 full view 与 local view 的重叠，可在像素域定义非负函数 `psi_r(u)`，要求

\[
\sum_r\psi_r(u)=1,
\]

再令每个 token 的 mass 为其覆盖区域上 `psi_r` 的积分。这是 partition of unity：同一像素即使进入多个 view，总质量仍只算一次。

**这条单层不变性是正确的，但它正是数值积分权重的定义性性质，不是新 attention 理论。**

## 2. 与四类最接近工作的严格区分

### 2.1 ToMe proportional attention：公式已经相同

[Token Merging: Your ViT But Faster, ICLR 2023](https://openreview.net/forum?id=JroZRaRw7Eu) 在 merge 后跟踪每个 token 代表的原始 patch 数 `s_j`，并明确使用

\[
\operatorname{softmax}(QK^\top/\sqrt d+\log s).
\]

它给出的解释正是：一个 size 为 `s_j` 的 merged token 应等价于 `s_j` 个相同 key 的 copies。

- ToMe 的 `s_j` 是 merge multiplicity，通常为整数且随 merge 相加；
- C63 的 `w_j` 是几何面积/partition-of-unity mass，可为分数，并用于 full+local overlap。

这改变了 `w` 的来源和应用场景，**没有改变 attention 运算或 refinement-invariance 证明**。因此不能声称 `+log w` 或“token 不应一票一权”为新方法。

### 2.2 Continuous / quadrature attention：理论对象已经相同

把图像域记为 `Omega`、面积测度记为 `mu`，连续 attention 是

\[
\mathcal A(q)=
\frac{\int_\Omega e^{s(q,k(u))}v(u)\,d\mu(u)}
{\int_\Omega e^{s(q,k(u))}\,d\mu(u)}.
\]

用不等面积节点做 quadrature 时自然得到权重 `w_j`，即 C63 公式。当前 neural-operator / function-space 文献已经明确写出带 quadrature cell size `Delta_j` 的 softmax attention；等权规则网格时权重才约掉。2026 年的综述 [Principled approaches for extending neural architectures to function spaces](https://www.nature.com/articles/s42256-026-01267-z) 甚至逐式给出这一 operator form。

所以“attention 应对离散化/网格细分保持一致”是 continuum operator 的标准 consistency 要求，不是医学 VLM 新定理。

### 2.3 Partition-of-unity multi-view fusion：只解决 mass bookkeeping

partition of unity 的经典作用就是在重叠 chart/local approximation 间分配权重，使它们加和为一个全局对象。用于 full/local view 时，它保证

\[
\text{total geometric mass of a pixel}=1,
\]

但它不保证 full token 和 crop token 对该 pixel 产生相同的 representation，也不保证哪一种分配 `psi_full/psi_local` 最适合诊断。equal split、偏向高分辨率和按置信度分配都是不同方法；数学条件没有给出唯一选择。

因此 PoU 解决的是 overlap accounting，不是 evidence correctness。

### 2.4 HALC / SECOND：不是 exact duplicate，但不足以救新颖性

- [HALC, ICML 2024](https://proceedings.mlr.press/v235/chen24bi.html) 自适应选择多个 field of view，再做 focal-contrast decoding；它改变 view 和输出 logits。
- [SECOND, ICML 2025](https://proceedings.mlr.press/v267/park25c.html) 用 entropy 选择/细化 mask，并逐尺度 contrast；它改变 evidence-degraded branches。
- C63 不选择、不 mask、不 contrast，只在一次 attention 内修正 token base measure。

因此 C63 与 HALC/SECOND **执行上不同**。但其精确数学前作是 ToMe proportional attention 与 quadrature attention；“不是 SECOND”不等于“是新原理”。若把 C63 用于 full+crop token 拼接，系统层面还会落入已拥挤的 multi-resolution/global-local feature fusion。

## 3. 三个致命的实际架构断点

### 3.1 定理只对静态 cross-attention memory 直接成立

证明假设 children 的 `k,v` 相同。但本项目的 Huatuo、Hulu、LLaVA-Med 都把视觉 token 插入 decoder-only 序列，再经过多层 causal self-attention：

```text
[visual tokens, question tokens, answer tokens]
```

即使复制 token 的输入 embedding 和 position id 完全相同，后出现的复制 token 能看见更早 token，早 token 不能看见后 token。第一层之后它们的 hidden state、key 和 value 就不再相同。RoPE、suffix position shift 与 residual paths 又继续放大差别。

所以单层 theorem 不能推出整网 decoder 的 subdivision invariance。要恢复严格保证，需要把视觉 token 变成静态 cross-attention memory，或为视觉块设计双向/set-equivariant block mask；这已经是架构改造，不再是一行 training-free logit bias。

### 3.2 projected visual token 没有局部 pixel support

CLIP/ViT 的 projected token 在视觉塔内已经做过多层 global self-attention。一个位置 token 的 receptive field 不再只等于它表面的 `14×14` 或 `patch×patch` 像素块。crop token 更依赖整张 crop 的 resize、边界和上下文。

因此把几何 patch area 当成 latent token 的真实 evidence measure 缺少依据：`w_j` 衡量输入 coverage，却不衡量 contextual representation 已携带多少全局信息。

### 3.3 视觉 mass 与文本 mass 之间存在任意 gauge

若 attention 同时看视觉 keys 和文本/history keys，只对视觉 token 乘 `w_j` 时，`w` 的整体常数不会在 softmax 中约掉，它会改变视觉相对语言的总权重。

为了让原始规则网格保持不变，只能约定

\[
\sum_{j\in visual}w_j=N_0,
\]

其中 `N_0` 是原始视觉 token 数。这个绝对 mass 是模型训练时隐含形成的 operating point，不是面积理论决定的。不同模型、分辨率和 prompt 可能需要不同 `N_0`；一旦调它，方法又变成 modality attention calibration。

## 4. 本地结果是否支持“重复覆盖导致幻觉”

### 4.1 exact visual multiplicity 已经 NO-GO

本地 16 个 VinDr 三读者一致阴性 claim，完整 projected visual block 重复 `1/2/4/8` 次：

| Position mode | factor 1 FP | factor 8 FP | margin `8−1` | monotonic positive fraction |
|---|---:|---:|---:|---:|
| sequential | 18.75% | 18.75% | `+0.258`, CI `[+0.063,+0.484]` | 37.5% |
| tied | 18.75% | 6.25% | `−0.906`, CI `[−1.055,−0.742]` | 6.25% |

factor 8 没有增加 FP；tied duplication 甚至把 margin 推向更负。两种模式都未通过预注册 multiplicity gate。

这不证明“局部 subset 重复永远无效”，但已经否定了最简单的机制：**VLM 因把每个视觉 token 当独立票而随重复数单调增加阳性承诺。** 实际行为强烈受 causal order 和 position 影响，正好与第 3.1 节一致。

### 4.2 crop/full 差异也不是 multiplicity 证据

同病例阴性图：

- full FP `8.1%`；
- same-area random crop FP `62.9%`；
- context-removed crop FP `71.0%`。

但 full → crop 的 AUROC 只从 `0.7946` 到 `0.7980`，变化 `+0.0035`，95% CI `[-0.0623,+0.0730]`。这是巨大的 criterion shift / negative-context destruction，几乎没有新增病例排序信息。

crop 的高 FP 发生在单一 crop 输入，并不要求 full+crop overlap，更不能由“同一 pixel 被多 token 重复覆盖”解释。因此 Area Attention 没有命中当前最强的已观察失败机制。

## 5. 是否仍有独特可测预测

只剩下一条**软件性质**，不是临床机制预测：

> 在静态 cross-attention 或单层 attention 中，把任一 piecewise-constant token 拆成多个 children，并令 child masses 求和为 parent mass，输出应 bit/numerically invariant。

这个性质可以用随机 `q,k,v` 在 CPU 上十几行测试，且必然通过；它只验证实现没有写错，不能证明减少 hallucination。

真正的临床预测必须是：在 full+high-resolution-local tokens 中，raw concat 的错误随局部 tiling density 增长，而 measure-corrected concat 在保持小病灶 recall 的同时消除该增长。但：

1. exact block multiplicity 不产生 FP 增长；
2. crop FP 已被证明主要是工作点/上下文破坏；
3. 本项目尚无 raw full+local token concat 的正向结果；
4. `+log w` 对 decoder-only 整网没有严格 invariance；
5. 即使成功，它仍会被概括为“把 ToMe proportional attention 用到 multiresolution VLM”。

因此没有值得打断 baseline 的独特科学预测。

## 6. 若仅作为未来 baseline 的最小实验

不建议当前运行。若将来已有另一主线自然需要 full+local token concat，可附带一个不超过 16 case 的工程 canary：

1. 8 个小病灶 3/3 positive、8 个 finding-matched 0/3 negative；
2. 只复制 bbox 对应 visual-token subset，factor `1/2/4`；
3. matched background subset 作为 placebo；
4. raw concat、ToMe-style `1/r` mass correction、随机正权重三组；
5. 先测 raw ROI duplication 是否相对 background 呈单调 target-margin 增长；未出现即停止；
6. correction 仅检查是否恢复 factor-1 行为，不把“按构造不变”称为 accuracy gain。

即使该 canary 正向，也只授权把 Area Attention 加为 global/local baseline，不授权新论文主线。若一定要做严格整网 invariance，必须先改成 noncausal visual set/cross-attention；其成本和语义都超出 training-free 一行修改。

## 7. 最终判定

| 门 | 判断 |
|---|---|
| 数学公式是否正确 | 是，作为 quadrature/measure attention |
| subdivision theorem 是否新 | 否，ToMe proportional attention + 数值积分 consistency |
| PoU 是否提供唯一 fusion | 否，只守恒 mass，不决定 representation correctness |
| 是否与 HALC/SECOND exact collision | 否 |
| 是否适配现有 decoder-only VLM | 不具严格整网保证；causal order、RoPE、contextual tokens 破坏假设 |
| 本地 multiplicity 机制 | 已 NO-GO：factor 8 不增 FP、非单调 |
| crop/full 是否提供动机 | 否：主要是 criterion shift / negative-context destruction |
| 是否值得当前 GPU | 否 |
| ICLR Oral 主方法 | **NO-GO** |

可保留的一句话工程原则是：

> 多分辨率视觉系统在拼接 overlapping tokens 时，应显式声明其 reference measure，并用 refinement-invariance unit test 防止无意的 coverage double counting。

这是好的实现规范，但当前证据与新颖性都不足以把它包装成医学 VLM 幻觉缓解算法。


# C49 Residual Pursuit / Sparse Claim Support Recovery：公式级碰撞审计

审计日期：2026-08-13  
审计范围：只做公式、文献与本地证据审计；未运行 GPU，未修改或中断 baseline。  
候选目标：把开放回答中的 `K` 个阳性 finding 当作稀疏 support，用冻结图像 embedding 和 finding 文本字典，在保持 claim 数不变时做一次 replacement，以降低视觉重建残差。

## 1. 将候选还原成标准数学对象

令归一化图像表示为 `z ∈ R^d`，finding ontology 的文本向量组成字典

```text
D = [d_1, ..., d_M] ∈ R^{d×M}.
```

VLM 草稿中的阳性 claim 集为 `S_0`，且 `|S_0|=K`。对任意 support `S`，最优线性重建误差是

```text
E(S;z) = min_a ||z-D_S a||_2^2
       = ||(I-P_S)z||_2^2,
P_S   = D_S(D_S^T D_S)^†D_S^T.
```

C49 的一次更新为

```text
S_1 = argmin E(S;z)
      s.t. |S|=K and |S △ S_0|≤2,
```

即枚举 `i∈S_0, j∉S_0`，用 `j` 替换 `i` 后重新最小二乘。它不是新的 VLM decoding object，而是：

> **以 VLM 草稿为 warm start 的固定稀疏度 least-squares subset selection / sparse approximation，并在 1-exchange 邻域做 support refinement。**

令 `T=S_0\{i}`，`r_T=(I-P_T)z`。加入候选 `j` 带来的精确残差下降为

```text
E(T;z)-E(T∪{j};z)
  = <d_j,r_T>^2 / ||(I-P_T)d_j||_2^2,
```

只要分母非零。因此“哪个遗漏 claim 最能解释当前未解释视觉残差”就是标准 orthogonal least-squares / pursuit 的 residual-correlation 选择量，不是 C49 的新性质。

## 2. 直接数学碰撞

### 2.1 OMP with Replacement / swapping pursuit：固定 K 一换一已经存在

[Orthogonal Matching Pursuit with Replacement（NeurIPS 2011）](https://papers.nips.cc/paper_files/paper/2011/hash/500e75a036dc2d7d2fec5da1b71d36cc-Abstract.html)明确从大小为 `K` 的 support 出发：根据当前 residual 加入一个未选 coordinate，删除一个已选 coordinate，再在新 support 上重新求 least squares。其动作与 C49 对应如下：

| C49 | OMPR / swapping pursuit |
|---|---|
| finding 文本向量 `d_c` | sensing/dictionary atom `A_c` |
| 图像 embedding `z` | measurement `b` |
| 草稿 claim 集 `S_0` | 当前 nonzero support `I_t` |
| 加入一个遗漏 claim | add one coordinate using current residual |
| 删除一个草稿 claim | remove one coordinate |
| 固定 `K` | fixed sparsity `k` |
| 重新计算 `P_S z` | least-squares refit on the new support |

C49 若穷举全部 `(i,j)` 并选择残差最小者，比 OMPR 的具体 coordinate-selection heuristic 更彻底，但其数学对象是更早已有的 **swapping-based OMP refinement / best-subset 1-exchange**，不是不同的机制。OMP/OMPR 的 RIP、mutual-coherence 和 exact-support recovery 条件也都是既有 sparse-recovery 理论；不能将它们改写为医学 claim 定理后作为新理论贡献。

### 2.2 SpLiCE：CLIP 表示到文本概念字典的稀疏恢复已被直接实现

[SpLiCE（NeurIPS 2024）](https://papers.neurips.cc/paper_files/paper/2024/file/996bef37d8a638f37bdfcac2789e835d-Paper-Conference.pdf)把 CLIP 图像 embedding 变成 overcomplete 文本概念字典的稀疏、非负线性组合：

```text
z ≈ D w,  w sparse and nonnegative.
```

它是 training-free、task-agnostic，并明确把问题称为 sparse recovery；字典编码完成后其余计算可在 CPU 完成。C49 与它共享：

- 同一个图文联合 embedding 空间；
- 同一个文本 concept dictionary；
- 同一个稀疏线性表示假设；
- 同一个 reconstruction residual；
- 同一个 support 作为“图像包含哪些语义概念”的解释。

C49 剩下的差别仅是：以 VLM 草稿初始化 support、限制只换一个 atom、把 support 直接输出成 medical finding。它改变的是任务和初始化，不产生新的稀疏恢复量。

### 2.3 PCBM-ReD：图像表示由概念文本向量重建 + OMP 已逐式覆盖

[PCBM-ReD（AAAI 2026）](https://arxiv.org/abs/2601.12303)的实例级分解写为

```text
I_i = Σ_j w_j^i c_j + ε_i,
```

并明确用 OMP 获得稀疏 concept scores；其概念筛选本身也以最小化图像表示的投影重建误差为目标。逐项映射为：

| PCBM-ReD | C49 |
|---|---|
| image representation `I_i` | frozen image embedding `z` |
| text concept `c_j` | finding text atom `d_j` |
| sparse concept support | positive finding set |
| residual `ε_i` | unexplained visual residual |
| OMP decomposition | residual pursuit / one-swap refinement |
| fitted representation | selected claim span |

PCBM-ReD 的最终用途主要是 concept bottleneck / classification，而非开放报告纠错；但这只是应用 delta。C49 的核心“用文本概念的稀疏 span 重建图像 representation，再把 support 解释为概念集合”已被逐式覆盖。

## 3. 系统级近邻：不全等价，但进一步压低论文空间

| 工作/本地候选 | 精确关系 | 是否直接公式碰撞 |
|---|---|---:|
| [TagCLIP（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/28139) | frozen CLIP、training-free、open-vocabulary multi-label；用 patch classification、局部 refinement 和 global re-identification生成 tags | 否；不用稀疏投影，但已占据“冻结 CLIP 纠正多标签集合”应用邻域 |
| [ESREAL](https://arxiv.org/abs/2403.16167) | 把 caption 重建成图像并通过区域语义相似度定位 hallucinated token，再用 PPO 抑制 | 否；是像素/区域 cycle reconstruction 且需要训练，但“semantic reconstruction 缓解 hallucination”叙事已被占据 |
| SpLiCE / PCBM-ReD | embedding 由文本概念稀疏重建，support 表示语义概念 | **是，核心表示与目标直接碰撞** |
| OMPR / swapping OMP | 固定稀疏度、加入一个 atom、移除一个 atom、重新最小二乘 | **是，更新规则直接碰撞** |
| 本地 C40 Evidence Capacity Matching | claim–region 容量约束匹配；不是字典 span 重建 | 否；但本地未确认“joint allocation”可纠错 |
| 本地 C44 Pareto Claim Exchange | 同样固定 `K` 一换一，但依据 VLM/专家 Pareto 排序，不依据 residual span | 否；末端动作相同，且真实无标签交换已显著增错 |

因此，C40/C44 的失败**不能单独证伪** C49；但它们排除了“联合集合选择”或“固定 K 一换一”本身就会带来安全增益的说法。C49 必须靠新的 residual statistic 独立证明正确性，而该 statistic 又已是 SpLiCE/PCBM-ReD/OMP 的既有对象。

## 4. 不可约的新性质审计

### 4.1 “联合 residual 会避免重复 claim”不是新性质

当两个 claim atoms 高度相关时，加入第一个后，第二个在正交 residual 上的边际收益变小。这确实区别于逐 claim cosine/top-`K`，但它正是 OMP/OLS 正交化的定义性行为，也是 PCBM-ReD 采用 sparse decomposition 而非独立 CLIP similarity 的理由。不能作为新理论。

### 4.2 子空间 residual 对 claim 身份和极性不可识别

C49 的误差只依赖 `span(D_S)`。对任意可逆矩阵 `A`，

```text
span(D_S A)=span(D_S)
⇒ ||(I-P_{D_S A})z||^2 = ||(I-P_{D_S})z||^2.
```

特别地，将任一 atom 变为其相反方向 `d_c→-d_c`，残差完全不变。`K=1` 时更加直观：

```text
E({c};z)=||z||^2 - <z,d_c>^2/||d_c||^2,
```

它最大化的是**绝对**余弦相似度。因此 unconstrained projection 原则上不能区分“病灶存在”和语义相反的方向；它识别的是一个 line/subspace，而不是一个带符号的 clinical claim。

若改成 `w≥0` 的锥投影以保留方向，便回到 SpLiCE 的 sparse nonnegative linear decomposition；这修复一个缺陷，却进一步加强已有方法碰撞。

### 4.3 稀疏恢复保证不能自动转成 clinical-correctness 保证

OMP/OMPR 的支持恢复定理要求类似

```text
z = D_{S*}a* + ε
```

并且字典满足 RIP / mutual coherence、真系数有足够幅值等条件。冻结 CLIP/BioMedCLIP 的训练目标只要求图文匹配，并未保证一张含多个 finding 的图像 embedding 是这些 finding 文本向量的线性叠加。图像中的正常解剖、体位、设备、人口学与采集域也占据大量表示能量。

所以“重建 z 更好”不蕴含“claim 更真实”。一个相互正交但临床错误的 atom 集可以比真实但高度相关的 findings 更好地覆盖 embedding 空间；projection objective 甚至天然偏好增加 span 维度/多样性。这一缺口不是换阈值可以解决的。

### 4.4 剩余 delta 只是一个可测应用假设

唯一未被逐式覆盖的经验问题是：

> 以医学 VLM 草稿作为 warm start 时，单次 OMPR-style replacement 是否恰好能把 fabricated positive 与 omitted positive 成对交换，并在固定 `K` 下改善报告？

这是可做的应用实验，但不是不可约的新算法性质。即使结果很强，审稿人仍可准确概括为：

> “用 SpLiCE/PCBM-ReD 式文本字典稀疏分解，对 VLM claim set 做一次 OMPR replacement。”

## 5. 与本地已知结果的关系

- **C40**：Huatuo patch cache 上，容量 1 的 matching 在 development 令 exact-set `.158→.100`，confirmation 增益 CI 接触 0；容量 2/3退化为独立选择。它不直接测试 C49，但没有给 joint set reconstruction 提供正现象。
- **C44**：MIMIC-CXR confirmation 中，固定 `K` 最陡一换一使总错误 Huatuo `+3.35%`、Hulu `+19.43%`，均为伤害；说明“从候选中找一个更合理 replacement”最难的是病例级真假识别，不是保持输出长度。
- **C46/C47**：冻结 CXR 专家只对 Huatuo有强条件增量，对 Hulu 很弱；one-bit veto 也未过跨模型安全门。若 C49 使用相同冻结专家 embedding，其信息上限不会因 OMP 包装而自动提高。

这些结果不构成 C49 的经验致死实验，却使它在数学直接碰撞之外还缺少本地自然正信号。

## 6. 严格裁决

| 门 | 结论 | 理由 |
|---|---|---|
| Grounded phenomenon | 弱通过 | fabrication 与 omission 共存真实存在，但未确认其是“错误 support 的稀疏重建问题” |
| 新数学对象 | **失败** | `min_{|S|=K}||(I-P_S)z||²` 是标准 best-subset/sparse approximation |
| 新更新规则 | **失败** | warm-start 固定 `K` 一换一属于 OMPR / swapping OMP |
| 跨模态独特机制 | **失败** | SpLiCE 与 PCBM-ReD 已把 CLIP 图像表示分解为文本概念稀疏组合 |
| hallucination 叙事空间 | 很窄 | ESREAL 已占 semantic reconstruction；TagCLIP 已占 training-free multi-label CLIP |
| clinical correctness 保证 | **失败** | residual 只识别 span，甚至对 atom 符号不变；稀疏恢复假设未由 CLIP 训练保证 |
| 是否值得 CPU/GPU 实验 | **否** | 目标要求的是新算法原语；即便实验成功也只得到已有 sparse pursuit 的医学应用 |

最终判定：

```text
NO-GO — direct mathematical and representation-level collision.
Do not run CPU score optimization or GPU generation.
Do not promote as an ICLR method candidate.
```

C49 可保留为未来 baseline：`SpLiCE/OMP support`、`warm-start OMPR one-swap`、`cosine top-K` 三者可用于检验真正新方法是否只是在做稀疏概念分解。但它不应进入当前主线，也不应通过增加医学 ontology、非负系数或更多 swap 复活；这些改动分别只是 task specialization、SpLiCE 设定和更完整的既有 pursuit。

## 7. 检索记录与证据边界

核心检索式：

- `CLIP image embedding text concept dictionary sparse recovery OMP`
- `sparse concept support reconstruction vision language multi-label`
- `orthogonal matching pursuit with replacement one swap fixed support`
- `best subset selection projection residual one exchange`
- `training-free open-vocabulary multi-label CLIP`
- `semantic reconstruction hallucination image caption`
- `medical claim set sparse support recovery hallucination`

承载裁决的工作均核对了论文主页或论文正文。未检索到一篇标题和应用设置完全相同的“medical VLM claim correction with warm-start OMPR”，但公式、表示假设与更新规则分别已有直接工作；任务换成医学开放报告不构成机制创新。

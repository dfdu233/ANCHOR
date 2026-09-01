# C67：三种“替换聚合原语”的敌对审计

日期：2026-08-13  
边界：只围绕已确认的 sparse-lesion tradeoff；硬排除 crop/mask、attention bias、VCD null、PoE、verifier/rerank、high-bit、laterality。本轮没有占用 GPU。

## 0. 先冻结自然现象，而不是先选漂亮数学

跨模型、开发/确认集都已确认：同一 finding 内，病灶 bbox 越小，VLM 的正确 margin 越低；fresh confirmation 的 Spearman 为 Huatuo `0.323`、Hulu `0.415`。因此“小病灶证据被全局聚合稀释”是合法问题。

但现有 patch field 已给出更严格的边界：在 Huatuo fresh panel `n=266` 上，`final margin + patch mean + max + top-5%` 的 macro AUROC 为 `0.7376`；加入 higher criticism 和 multiscale scan 后仅到 `0.7416`，增量 `+0.0040`，95% CI `[-0.0194,+0.0273]`。所以新对象若只是 max、top-k、scan、平滑或换名的局部加权，应直接关闭。

本轮审计三个计算对象真正不同的候选：局部背景归一化、拓扑 filtration、组合群测试。结论是：**两个缓存致死门失败，一个在公式层失败；三者都不排 GPU。**

## 1. Local-CFAR：检测“相对附近背景异常”，而非绝对高分

### 自然动机

雷达中的小目标和胸片小病灶有相似困难：目标只占少量单元，背景强度又随位置变化。普通 max 会把纵隔、膈肌等天然高响应误当目标；CFAR 的基本想法是把每个候选与它自己的局部背景比较。

### 精确对象与性质

对 patch score `z_i`，令 `R_i` 为去掉近邻 guard band 后的局部环，定义

```text
T_i = (z_i - median(R_i)) /
      (MAD(R_i) + lambda * MAD(z)).
```

这里 median 是中位数，MAD 是“与中位数距离的中位数”。它有一个干净但有限的性质：对任意 `a>0,b`，若全图分数统一变为 `z'=a z+b`，则 `T_i(z')=T_i(z)`。直观上，统一抬高或缩放分数不会制造局部证据。

经典 CA-CFAR 在 iid exponential clutter 下还能得到精确假警率：以 `N` 个背景单元均值为阈值基准时，`P_FA=(1+alpha/N)^(-N)`。但 VLM patch scores 具有解剖非平稳性、强空间相关和未知尾部，这个保证不能搬过来。我们只能使用上面的 affine invariance，不能声称 constant false alarm。

### 碰撞与不可约增量

- CFAR 是成熟雷达检测原语；2024 年 [See Further Than CFAR](https://arxiv.org/abs/2402.12970) 正是把学习式雷达检测与 CFAR 比较。
- 在医学视觉中，“病灶相对局部背景的差异”已是 patch anomaly detection 与小病灶检测的常见对象，例如 [PatchCL-AE, 2024](https://doi.org/10.1016/j.compmedimag.2024.102366)。
- 若用 `T_i` 选择/增强 token，它就是局部对比加权或 attention mask，落入本轮硬排除；若只用 `max_i T_i` 改答案，它是外部 verifier。

因此它唯一可能留下的价值，是在不改输出的缓存层先证明 `T` 含有标准聚合没有的信息。

### 零 GPU 致死结果

冻结 guard radius `1`、outer Chebyshev radius `3`、`lambda=0.1`；开发集只负责拟合，fresh confirmation 上比较：

| Base | + Local-CFAR | Delta | 95% CI | positive findings |
|---:|---:|---:|---:|---:|
| 0.7416 | 0.7495 | +0.0079 | [-0.0063,+0.0226] | 5/7 |

NLL improvement 为 `+0.00018`，95% CI `[-0.00955,+0.00966]`；开发拟合系数为 `-0.303`。预注册要求 `AUROC >= +0.02`、AUROC/NLL 下界均大于 0、至少 5/7 findings 同向，故 **NO-GO**。

解释：局部尖峰确实描述 patch field 的某种形态，但在控制 final/mean/max/top5/scan 后，它没有可靠恢复病灶真值；负系数还提示孤立的局部尖峰更可能是伪响应，而非病灶。

## 2. H0 Persistent Evidence：检测跨阈值存活的连通岛

### 自然动机

一个真实病灶可能不是最高的单 patch，而是一片相邻、跨多个阈值仍连通的弱响应；拓扑数据分析可以不选单一阈值，追踪连通分量从“出生”到“合并”的 lifetime。

### 精确对象与性质

对阈值 `t` 定义 superlevel set

```text
F_t = {patch i : z_i >= t}.
```

逐渐降低 `t`，每个连通岛在局部峰值处出生，在与更老岛合并时死亡。birth 与 death 的差是 H0 persistence。我们缓存审计用最大 finite lifetime 和前三个 lifetime 之和。

标准 stability theorem 给出：若两个 score fields 的逐 patch 最大变化不超过 `epsilon`，其 persistence diagrams 的 bottleneck distance也不超过 `epsilon`。直观上，小幅数值扰动不会让拓扑摘要剧烈变化。这是已有拓扑定理，不是论文贡献。

但它有一个决定性的反例：

- 两个孤立噪声尖峰可制造任意大的 finite lifetime；所以“长寿”不蕴含临床真实性。
- 一个单一、平坦、连通的真实病灶可能只形成 essential component；若只用 finite lifetimes，它得到 0；若给 essential component 人工设 death，则统计量退化为全局 range/max。

因此 persistence 只提供阈值稳定性，不提供“病灶而非噪声”的可识别性。

### 碰撞

- [PHG-Net, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Peng_PHG-Net_Persistent_Homology_Guided_Medical_Image_Classification_WACV_2024_paper.html) 已将 persistence diagram 编码并融合入 CNN/Transformer 做医学分类。
- [TopoCL, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Meng_TopoCL_Topological_Contrastive_Learning_for_Medical_Imaging_CVPR_2026_paper.pdf) 已系统学习医学图像的 topological representations。
- 本地 multiscale scan 已覆盖“连通弱区域跨尺度聚合”的大部分统计效应。

把 PH 特征作为 token 输入需要训练/融合；把 PH scalar 用于输出控制又是 verifier。因此仍先要求缓存增量。

### 零 GPU 致死结果

| Base | + H0 persistence | Delta | 95% CI | positive findings |
|---:|---:|---:|---:|---:|
| 0.7416 | 0.7507 | +0.0091 | [-0.0048,+0.0241] | 5/7 |

NLL improvement 为 `+0.01784`，CI `[+0.00688,+0.02850]`，但 AUROC 未达 `+0.02` 且 CI 跨 0；两个 persistence 系数分别为 `-0.206/-0.301`。故 **NO-GO**。

这个结果允许一个很窄的解释：拓扑碎片度可能帮助校准置信度；但它没有达到“找回被稀释病灶”的门，更不能自然导出不属于 verifier/mask 的 hallucination mitigation。

## 3. Expander / Group Testing：用组合测试找稀疏阳性 token

### 自然动机

若病灶只影响 `k << n` 个 patch，可否不逐 patch 搜索，而用少量互相重叠的 patch groups 编码，让弱病灶在多组中重复出现，再像核酸混检一样解码？

### 精确对象与经典保证

令未知稀疏阳性向量 `s in {0,1}^n`，二值测试矩阵 `A in {0,1}^{m x n}`。标准 group testing 假设每组返回

```text
y_j = OR_{i:A_ji=1} s_i.
```

若 `A` 是 `k`-disjunct，任意不超过 `k` 个阳性项目可被唯一识别；带 `(k,e)` disjunctness 还能纠正有限个测试错误。这一对象已用于 [Multilabel Classification with Group Testing and Codes, ICML 2017](https://proceedings.mlr.press/v70/ubaru17a.html)。这些都是成熟保证。

### 公式级致死：冻结 VLM 不实现 OR oracle

把 visual tokens `x_i` 线性分组为

```text
g_j = sum_i A_ji x_i
```

后，若 claim readout 是线性的，任意组权重 `beta` 满足

```text
sum_j beta_j <w,g_j>
= sum_i (A^T beta)_i <w,x_i>.
```

这与直接给原 patch 加权完全相同，只是 token weighting/merging 的重参数化，不是新计算能力。若把 group tokens 送入非线性冻结 Transformer，经典恢复定理又不再适用：VLM 对组的响应既不保证单调，也不保证只要组内有一个病灶就返回 OR；正负 token 还能相互抵消。

若先外部计算每个 `s_i` 再做 OR，最难的 patch detector 已经被假定存在，群测试只是一个多余的压缩/解码层，并退化为 verifier。应用邻域也已拥挤：[SparseVLM, ICML 2025](https://proceedings.mlr.press/v267/zhang25s.html) 已做 training-free visual-token pruning/recycling；[CS-VLM, 2025](https://arxiv.org/abs/2507.02957) 已把 compressed sensing projection/recovery 引入 VLM attention。

因此 **NO-GO at formula level**，不运行缓存优化或 GPU。

## 4. 统一裁决

| Candidate | 计算对象是否不同 | 数学性质 | 本地结果/致死点 | 判定 |
|---|---|---|---|---|
| Local-CFAR | 局部背景标准化 | positive-affine invariant；经典 PFA 假设不成立 | AUROC `+0.0079`，CI 跨 0 | NO-GO |
| H0 persistence | superlevel-set 拓扑 lifetime | L-infinity stability；不保证 truth | AUROC `+0.0091`，CI 跨 0 | NO-GO |
| Group testing | 组合 OR measurements | disjunct recovery 需 OR oracle | linear 情况等价 token weighting；非线性无保证 | NO-GO |

本轮真正学到的不是“再换一种 patch 聚合”，而是更强的负边界：

> 当前 claim-conditioned patch score field 不像一个只需更聪明 detector 就能恢复的经典稀疏信号。局部对比、连通拓扑、scan 与群测试分别代表背景归一化、形状稳定和组合编码，但都无法越过“冻结 VLM 没有提供可靠病例级局部 evidence”的瓶颈。

因此下一轮不应继续枚举稀疏统计量。只有当新方法改变的是 **VLM 实际使用视觉信息的计算语义**，同时又不退化为 mask/attention bias/verifier，才值得进入公式审计；否则即使数学名字更新，也仍在同一个已失败的 scalar patch field 上打转。

## 5. 可复现材料与边界

- 审计实现：`anchor/corrected_sgta/audit_sparse_operator_triple_v1.py`
- 单元测试：`tests/test_sparse_operator_triple_v1.py`（3/3 passed）
- 结果：`corrected_runs/daylong_idea_search_v1/sparse_operator_triple_huatuo_v1.json`
- 输入 patch artifact：`corrected_runs/daylong_idea_search_v1/patch_scores_huatuo_v1/patch_scores.npz`
- confirmation `n=266` 的标签此前已因其它 endpoints 打开，因此这是 secondary fatal audit，不是新的 blind confirmation；任何边缘正信号都不得宣称确认。


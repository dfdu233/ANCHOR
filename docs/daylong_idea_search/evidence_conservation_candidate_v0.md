# Evidence-Conserving Local Search：候选草案 v0

日期：2026-08-12。状态：**待真实 patch/search-tax 结果与敌对文献审计，不是已成立方法。**

## 自然问题

局部视觉方法通常在很多区域中取最大响应。即使每个区域只有噪声，候选越多，最大值也越大。
更根本的问题不是“应该减去多大的搜索惩罚”，而是：

> 把同一视野切得更细，不应凭空增加一张图像支持某个临床 claim 的总证据。

## 最小原则

把未知病灶位置记为潜变量 `R`，预先给每个候选区域位置先验 `pi_r`，满足
`sum_r pi_r = 1`。若局部模型输出的是该区域在“有病灶”与“无病灶”下的 likelihood
ratio `L_r`，则整图 evidence 不是 `max_r L_r`，而是对未知位置边缘化：

```text
L_image = sum_r pi_r L_r.
```

直观例子：把一个区域一分为二时，它的先验质量也一分为二；若两个子区域没有提供新信息，
总 evidence 保持不变。`max` 则把每个新增候选都当成拥有完整先验质量，可能把搜索机会当成证据。

在全阴性零假设下，如果每个 `L_r` 都是合法 likelihood ratio，
`E[L_r | negative] = 1`，于是无论候选数和候选间依赖如何：

```text
E[L_image | negative] = sum_r pi_r E[L_r | negative] = 1.
```

这只是 Bayes 边缘化和线性期望，不是新数学。可能的新贡献必须来自 VLM 特有的经验规律：
当前局部增强是否系统违反这一 evidence-conservation 原则，并把违反量传入最终 hallucination；
守恒聚合能否在固定回答长度/claim 数下同时保留小病灶 recall、减少 false positive。

## 与已有工作的边界

- average likelihood-ratio detection、Bayes model averaging、MIL mean/log-sum-exp 都是已有工具；
- BCEA 已覆盖 adaptive acquisition 后重校准；
- SECOND、AGLA、VGA 已覆盖局部选择与增强；
- 因此“用平均代替最大”本身不具备论文新颖性。

只有下面联合规律成立才继续：

1. 增加无信息候选会提高 max-based 最终 FP，而守恒聚合不提高；
2. 对真实小病灶，守恒聚合仍在 final margin 与 mean/max/top-k 强基线上提供增量；
3. 同一规律跨模型、finding、分辨率和至少三类局部方法复现；
4. OE fixed-K 下减少 fabricated positive 且 omission 不增加。

## 最小致死实验

1. 使用 Huatuo patch cache 的七项全 `0/3` 图，比较嵌套区域数下 `max` 与
   `sum pi_r L_r` 的 null drift；`L_r` 的校准只使用 development negatives。
2. 在 `0/3`/`3/3` confirmation claims 上，比较
   `final+mean+max+top5` 与再加入守恒 evidence；冻结门仍为 macro AUROC `+0.02`、
   AUROC/NLL bootstrap CI 下界大于 0、至少 `5/7` findings 正向。
3. 只有前两门通过，才在真实解码中用守恒 evidence 限制 local enhancement；
   CE 匹配 Yes-rate，OE 固定 positive claim 数 `K`。

## 当前裁决

这是对 selection--reuse 候选的一个更简洁替代原则，但很可能被经典 average-LR/MIL
完全解释。真实 patch gate 或敌对审计任一失败即关闭；不先命名算法，也不把上述恒等式作为
理论贡献。

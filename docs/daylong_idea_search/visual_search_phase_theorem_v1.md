# Visual Search Tax 的最小相变模型与证据守恒聚合审计

日期：2026-08-12  
任务边界：只做理论推导、原创性审计和有限样本实验设计；不占 GPU，不修改 baseline。

## 0. 结论先行

### 0.1 是否存在非平凡、可证伪的 phase law？

**存在一个干净的两门相变，但通用数学本身不新。**

对一张图，局部方法在 `M` 个候选区域中先用 selector 选区域，再用 verifier 判断该区域是否支持 claim。一个真实病灶要影响最终输出，必须连续通过两道门：

1. **找到病灶**：病灶在 selector 上的信号必须超过约 `sqrt(2 log M)` 的搜索竞争；
2. **验证病灶**：病灶在 verifier 上的信号必须超过 selector--verifier 共享噪声造成的验证税，后者约为 `rho sqrt(2 log M)`。

其中 `rho` 是无病灶时 selector 与 verifier 对同一区域的噪声相关性。候选越多，第一门更难；`rho` 越大，第二门也越难。只要其中一门失败，增加局部搜索就可能只增加 false positive，而不能恢复真实证据。

这比单独说“Gaussian max 随 `M` 增长”更有内容，因为它预测一个二维相图：

```text
                         verifier 通过
                    否                    是
selector 不通过    未找到病灶             不可能利用
selector 通过      找到但不能确认          可纠错区
```

但是，其证明只是高斯极值、选择后推断和稀疏信号检测的直接组合；不能把公式本身写成 ICLR 理论贡献。可能的新贡献只能是：**该二维边界能否跨模型、跨局部方法、跨分辨率，同时预测医学 VLM 的 FP 与 FN。**

### 0.2 `logmeanexp - log M` / 分割细化证据守恒是否原创？

严格结论是：

- 若每个局部分数真的是某个互斥位置假设的 likelihood ratio（似然比），整图证据必须是先验加权平均 `sum_r pi_r L_r`；均匀先验时，其对数就是 `logsumexp(log L_r) - log M`。
- 这由全概率公式直接得到，和 average likelihood ratio、Bayes model averaging、mixture e-value 同源，**数学不原创**。
- 它有一个比手工减 `sqrt(2 log M)` 更优雅的性质：候选被无信息地复制或细分时同步拆分先验质量，整图证据不变；在 null 下若每个 `L_r` 的期望为 1，则整图证据的期望也恒为 1，不随候选数膨胀。
- 但 VLM margin、attention、crop 后的 yes-logit 通常不是 likelihood ratio。直接对这些分数做 logmeanexp 没有证据守恒保证，甚至可能仍然随 `M` 漂移。

因此它目前是一个**强而简洁的 baseline / 方法缝隙**，不是已完成的 Oral idea。只有发现并证明“局部 VLM 方法普遍违反分割一致性，而校准成局部 e-value 后的先验质量边际化能在 matched operating point 下统一修复分辨率依赖的 FP，同时保留小病灶 recall”，才可能成为主会方法；达到 Oral 仍需要跨方法的统一相图，而不只是一个聚合公式。

---

## 1. 最小 selector--verifier 模型

### 1.1 变量的直观含义

一张图有 `M` 个候选区域。对于第 `i` 个区域：

- `A_i`：selector 分数，负责决定“看哪里”；
- `B_i`：verifier 分数，负责决定“这个区域是否真的支持 claim”；
- `J = argmax_i A_i`：最终被选中的区域；
- 最终局部证据为 `B_J`。

在没有该病灶时，假设不同区域先从最简单的独立模型开始：

\[
\begin{pmatrix}A_i\\B_i\end{pmatrix}
\overset{iid}{\sim}
\mathcal N\!\left(
\begin{pmatrix}0\\0\end{pmatrix},
\begin{pmatrix}1&\rho\\\rho&1\end{pmatrix}
\right).
\]

`rho` 表示同一区域在两次判断中的共享噪声。若同一 attention / logit 既用于选区又用于验证，`rho` 会很高；若 verifier 真正独立，`rho=0`。

在有且仅有一个真实区域（令其为 `i=1`）时，只改变该区域的均值：

\[
(A_1,B_1)\sim
\mathcal N((a,b),\Sigma_\rho).
\]

- `a` 是病灶对“找到哪里”的 selection signal；
- `b` 是病灶对“是否成立”的 validation signal。

允许 `a != b` 很重要：一个 saliency map 可能找到异常纹理，但语言 decoder 未必把它解释成正确病种；反之，一个 verifier 可能会识别给定 crop，但 selector 根本找不到它。

### 1.2 模型边界

独立区域与高斯只是可解的最小模型，不是胸片 patch 的真实假设。重叠 patch 强相关、不同位置异方差、多个病灶并存时，后面的解析阈值不能直接用作保证。正式实验使用 image-level empirical null，不估一个无保证的 `M_eff`。

---

## 2. Null inflation：没有病灶也会出现“最佳区域”

令 `Z_(M)=max_i A_i`。联合高斯的条件均值为

\[
\mathbb E[B_i\mid A_i]=\rho A_i.
\]

由于 `J` 完全由所有 `A_i` 决定，故

\[
\boxed{\mathbb E_0[B_J]=\rho\,\mathbb E[Z_{(M)}]}.
\]

当 `M` 较大时，定义

\[
u_M=\Phi^{-1}(1-1/M)
\approx \sqrt{2\log M}-
\frac{\log\log M+\log(4\pi)}{2\sqrt{2\log M}},
\]

则 `E[Z_(M)]` 的首项为 `u_M`，所以

\[
\mathbb E_0[B_J]\approx \rho\sqrt{2\log M}.
\]

直观解释：`M` 越大，selector 越容易挑到一个偶然高值；只要 verifier 与 selector 共享噪声，第二次判断会保留其中 `rho` 的比例。

### 2.1 精确的 null 分布

记 `sigma=sqrt(1-rho^2)`。对 `|rho|<1`，有

\[
\Pr_0(B_J\le t)
=M\int_{-\infty}^{\infty}
\phi(z)\Phi(z)^{M-1}
\Phi\!\left(\frac{t-\rho z}{\sigma}\right)dz.
\]

因此 level-`alpha` 的正确阈值 `q_{M,rho,alpha}` 是上式分布的 `1-alpha` 分位数，而不是固定阈值。其大样本首项为

\[
q_{M,\rho,\alpha}
=\rho u_M+\sqrt{1-\rho^2}\,z_{1-\alpha}+o(1).
\]

所以，如果仍使用与 `M=1` 相同的固定阈值且 `rho>0`，null FP 会随 `M` 上升；正确校准必须支付约 `rho u_M` 的验证税。

特殊情形：

- `rho=0`：选区不抬高 verifier，阈值仍为 `z_(1-alpha)`；
- `rho=1`：`A=B`，退化成普通 maximum scan，其阈值直接是 `max_i A_i` 的分位数；
- 所有 patch 完全相关：新增候选不提供新赢家，独立模型的 `sqrt(2 log M)` 不再适用。

这些结论都是经典 winner's curse / post-selection inference 的背景，不是新数学。

---

## 3. 真区域被选中的概率

真实区域的 selector 均值为 `a`，其余 `M-1` 个区域均值为零。真实区域被选中的精确概率为

\[
p_{\mathrm{sel}}(M,a)
=\Pr(A_1>\max_{i>1}A_i)
=\int \phi(z-a)\Phi(z)^{M-1}dz.
\]

当 `M` 增长、`a=u_M+c` 时，其他区域的最大值在 `u_M` 附近集中，而真实区域仍有单位方差，因此

\[
p_{\mathrm{sel}}(M,u_M+c)\to\Phi(c).
\]

这给出第一道相变：

- `a-u_M -> -infinity`：找到真区域的概率趋近 0；
- `a-u_M -> +infinity`：找到真区域的概率趋近 1；
- `a=u_M+O(1)`：处于非平凡过渡区。

所以增加候选不是免费提高分辨率；它同时把真区域必须战胜的最大噪声从常数推到约 `sqrt(2 log M)`。

---

## 4. 校正后的 power 与二维相变

### 4.1 精确 power

仍令真实区域为 `1`，并使用正确的 null 阈值 `q=q_(M,rho,alpha)`。power 分为“选对区域”与“选错区域”两项：

\[
\begin{aligned}
\Pr_1(B_J>q)
=&\int \phi(z-a)\Phi(z)^{M-1}
\bar\Phi\!\left(
\frac{q-b-\rho(z-a)}{\sigma}
\right)dz\\
&+(M-1)\int \phi(z)\Phi(z)^{M-2}\Phi(z-a)
\bar\Phi\!\left(
\frac{q-\rho z}{\sigma}
\right)dz.
\end{aligned}
\]

第一行是真区域被选中且通过 verifier；第二行是某个 null 区域被选中后仍越阈值。这个公式允许 `a` 与 `b` 不同，因此比“max score 越大越好”更符合局部 VLM 的真实管线。

### 4.2 最清楚的独立 verifier 情形

若 `rho=0`，选区与验证噪声独立，`q=z_(1-alpha)`，精确化简为

\[
\boxed{
\mathrm{Power}
=\alpha+p_{\mathrm{sel}}(M,a)
\left[
\bar\Phi(z_{1-\alpha}-b)-\alpha
\right].
}
\]

这条式子的背景很直观：

- 以 `p_sel` 的概率找到病灶；
- 找到后，verifier 的检测率从 null 的 `alpha` 提升为 `barPhi(z_(1-alpha)-b)`；
- 找不到时，只剩 false-positive 水平。

因此，即使 verifier 很强（`b` 大），selector 找不到病灶时也没有用；即使 selector 完美，`b` 接近零时也没有用。

### 4.3 相关 verifier 的两门相变

当 `0<=rho<1` 时，令

\[
a=u_M+c,\qquad b=\rho u_M+d.
\]

在极限中，power 进入一个非平凡区域：

\[
\mathrm{Power}
\to
\alpha\Phi(-c)
+\int_{-c}^{\infty}\phi(z)
\bar\Phi\!\left(
z_{1-\alpha}-\frac{d+\rho z}{\sqrt{1-\rho^2}}
\right)dz.
\]

不要求读者记住这个积分；它表达的是两个坐标：

\[
\boxed{
\text{selection coordinate}=a-u_M,
\qquad
\text{validation coordinate}=b-\rho u_M.
}
\]

要得到稳定 power，两者都必须为正并随样本增强：

- `a` 不超过 `u_M`：真病灶淹没在许多候选的最大噪声里；
- `b` 不超过 `rho u_M`：即使偶尔找到病灶，也无法越过 selection--reuse 造成的 null 验证税；
- 两者都超过边界：才进入可利用的局部证据区。

这就是当前候选最简洁的 phase law。它不是把一个 Gaussian max 换名字，而是把“能否找到”与“找到后能否证明”分开，并允许二者在 VLM 内部不同。

### 4.4 一个重要反例

若 `A=B`，则 `rho=1` 且真区域的两种均值必须一致（`a=b`）。此时二维边界退化为普通 scan 的一维边界 `a approx u_M`。所以只有实证证明不同局部方法存在 `a != b` 或不同 `rho`，二维理论才比经典 scan 多提供解释力；否则它只是冗余记号。

---

## 5. Small-lesion scaling

设视觉网格有 `N` 个 patch，真实病灶覆盖其中 `k` 个，每个病灶 patch 在 selector 上产生均值偏移 `delta_A`，在 verifier 上产生 `delta_B`。若正确局部窗口使用标准化求和，则

\[
a\approx\delta_A\sqrt{k},
\qquad
b\approx\delta_B\sqrt{k}.
\]

大小为 `k` 的非重叠候选位置数量约为 `M_k approx N/k`。于是两门边界变为

\[
\boxed{
\delta_A\sqrt{k}
\gtrsim \sqrt{2\log(N/k)},
}
\]

\[
\boxed{
\delta_B\sqrt{k}
\gtrsim \rho\sqrt{2\log(N/k)}.
}
\]

全图平均的信号强度却只有

\[
S_{\mathrm{global}}\approx \delta_A k/\sqrt N.
\]

因此存在经典的 sparse-local regime：

\[
\frac{\sqrt{2\log(N/k)}}{\sqrt{k}}
\lesssim\delta_A
\ll\frac{\sqrt N}{k},
\]

其中全局平均看不到病灶，局部 scan 能看到。该区间在 `k log(N/k) << N` 时非空。

本项目新增的可检验预测不是这条经典 scan 边界，而是第二个 verifier 坐标：

> 相同病灶面积与相同 selector 命中率下，`rho` 更高的局部增强方法应有更高 null threshold；若没有更强的 `delta_B` 补偿，它反而更容易把局部响应变成 false-positive commitment。

这可以同时解释“小病灶 FN”与“局部增强 FP”，但必须用真实病例级错误确认，不能只画内部 score 曲线。

---

## 6. 有限样本、相关 patch 下怎样测试

### 6.1 不使用无保证的 `M_eff`

相关 patch 的 maximum 分布不能由一个 effective sample size 唯一决定。相同 effective rank 的协方差可以有不同的坐标最大值尾部。因此：

- `sqrt(2 log M_eff)` 只允许作描述性拟合；
- 不把它当 FP 保证；
- 不因解析曲线不合就事后换 `M_eff` 定义。

### 6.2 直接测三个可观察量

对每个预先冻结的 `M / scale / method / model` 配置：

1. **null 验证税**：在 image-disjoint `0/3` development 图上完整运行 selector 与 verifier，估计 `q_M^(0)(alpha)`；
2. **真实命中率**：在有 bbox 的 `3/3` positives 上报告被选区域与 bbox 的预注册 overlap，得到 `p_sel(M,k)`；
3. **命中后的验证率**：只按预注册 overlap 定义分层，估计
   `P(B_J > q_M^(0) | selected overlaps bbox)`。

在独立 confirmation 上，三者预测最终 detection：

\[
\widehat{\mathrm{Power}}_M
=\widehat p_{\mathrm{sel},M}\widehat v_{\mathrm{hit},M}
+(1-\widehat p_{\mathrm{sel},M})\widehat v_{\mathrm{miss},M}.
\]

这个分解本身是全概率公式，不是贡献。真正的检验是：只用 development 冻结的三个量，能否跨 `M`、病灶面积、模型和方法预测 confirmation 的病例级 FP/FN，而不重新拟合。

### 6.3 Phase-law GO/NO-GO

第一门必须同时满足：

1. `M` 增大时，null 的 selected verifier 分位数 `q_M^(0)` 在至少两个模型和多数局部方法上显著增长；
2. 增长幅度与在 null 上测得的 selector--verifier coupling 同向，并能 out-of-sample 预测最终 FP，而不只是内部 margin；
3. 小病灶的 bbox hit rate 随 `M` 与区域分辨率呈预测的竞争边界，而非所有面积一起变化；
4. 两坐标 `(selection hit, B_J-q_M)` 比 final margin、病灶面积、方法 identity 的强 baseline 增加至少 `0.02` AUROC，image-bootstrap CI 下界大于 0。

任一失败就不能声称二维 phase law。特别是，若 `A=B` 型的一维 scan 已解释全部结果，二维故事关闭。

---

## 7. Partition-Invariant / Evidence-Conserving aggregation

### 7.1 为什么先验加权平均是正确对象

设阳性 claim 的病灶位置是一个未知 latent variable `R`，其先验概率为 `pi_r`。对固定位置假设 `R=r`，定义局部 likelihood ratio：

\[
L_r(x)=\frac{p(x\mid Y=1,R=r)}{p(x\mid Y=0)}.
\]

由于阳性模型对未知位置做边际化，整图 likelihood ratio 必须是

\[
\boxed{
L_{\mathrm{image}}(x)=\sum_r\pi_r L_r(x).
}
\]

若位置先验均匀，`pi_r=1/M`，其对数为

\[
\boxed{
\log L_{\mathrm{image}}
=\operatorname{logsumexp}_r(\log L_r)-\log M.
}

这里的 `-log M` 不是手调 multiple-testing penalty，而是每个位置只有 `1/M` 先验质量的必然结果。

### 7.2 分割细化一致性

假设一个粗位置 `r` 被细分成子位置 `r1,...,rq`，同步拆分先验：

\[
\sum_j\pi_{rj}=\pi_r.
\]

若粗位置的条件模型确实等于子模型的条件混合，即

\[
L_r=\sum_j\frac{\pi_{rj}}{\pi_r}L_{rj},
\]

则

\[
\pi_rL_r=\sum_j\pi_{rj}L_{rj},
\]

所以整图证据完全不变。一个候选被无信息地复制成 `q` 份、每份权重改为原来的 `1/q`，也是这个性质的特例。

这给出一个简单的 characterization：若聚合器满足

1. 单一候选时输出其证据；
2. 交换候选顺序不改变结果；
3. 用一个条件混合替换其子候选时结果不变；

那么聚合器只能是先验加权线性和 `sum pi_r L_r`。证明只是反复把所有候选合并成一个条件混合。因此这是概率论的 coherence 公理，不是新定理。

### 7.3 Null evidence 不随候选数膨胀

真正的 likelihood ratio 在 null 下满足 `E_0[L_r]=1`。无论区域间怎样相关，线性期望给出

\[
\mathbb E_0[L_{\mathrm{image}}]
=\sum_r\pi_r\mathbb E_0[L_r]=1.
\]

于是 Markov 不等式给出

\[
\Pr_0(L_{\mathrm{image}}\ge1/\alpha)\le\alpha.
\]

这说明候选增多不会自动创造**期望意义上的**证据。它不等于说方差或有限阈值分布完全相同；相关结构仍可改变分布形状。若只需要有限样本安全性，可以把每个局部分数预先校准成 e-value `E_r`（null 下期望不超过 1），再用 `sum pi_r E_r`；任意图内依赖下它仍是 e-value。

mixture e-value、Bayes factor averaging 和 average likelihood ratio 都是标准工具。参见 Chan & Walther 的 scan 与 average likelihood ratio 比较；不能把上述期望等式当原创理论。

### 7.4 它没有消灭 search cost，只把成本放回正确位置

若只有一个真实位置 `r*`，先验质量约为 `pi_(r*)=1/M`，强信号时

\[
\log L_{\mathrm{image}}
\approx \log L_{r*}-\log M.
\]

因此真区域仍需提供约 `log M` 的 local log-evidence。若局部观测是均值偏移 `delta`、病灶覆盖 `k` 个 patch，则真区域 log-likelihood ratio 的典型强度约为

\[
\log L_{r*}\approx\delta^2k/2.
\]

得到边界

\[
\delta^2k/2\gtrsim\log M_k
\quad\Longleftrightarrow\quad
\delta\sqrt{k}\gtrsim\sqrt{2\log M_k}.
\]

它与经典 scan boundary 一致。换言之，证据守恒不是“免费搜索”；它只是不允许算法把未支付的搜索机会伪装成证据。

---

## 8. 为什么不能直接对 VLM logits 做 logmeanexp

这是该方法最重要的真实性门槛。

### 8.1 VLM margin 通常不是 likelihood ratio

decoder 的 yes/no logit 可能同时包含：

- 语言先验；
- 全图病种先验；
- prompt 诱导的肯定偏置；
- crop/resize 的 OOD 响应；
- 已在不同区域中重复使用的相同全局上下文。

所以 `exp(margin_r)` 通常不满足 `E_0 exp(margin_r)=1`，也不对应互斥位置模型。对它做 `logmeanexp - log M` 只是一个 soft pooling heuristic，没有证据守恒保证。

### 8.2 Overlapping windows 不是空间 partition

多尺度、重叠窗口是一个候选 support dictionary，不是互斥位置分割。简单给每个窗口相同 `1/M` 权重会随字典设计改变先验。合法做法是先冻结一个连续的“病灶位置、尺度、形状”先验，再把离散候选看作该先验的 quadrature；增加采样密度时权重随覆盖体积缩小。

### 8.3 多病灶与弥散病变是反例

“恰好一个未知位置”的混合模型不适合：

- 多发结节、双侧积液等多个病灶；
- 水肿等弥散证据；
- 一个局部异常需和全局解剖共同判断的 claim。

可扩展成 latent support / point-process alternative，但方法和理论会明显变复杂。第一轮应把结论限定为 focal、single-support findings，并将 diffuse finding 作为预注册反例；不能用一个位置混合解释所有医学 claim。

### 8.4 数据依赖 attention 权重会重新引入选择偏差

若用 `A_r` 在测试图上产生权重 `w_r(A)`，再计算 `sum w_r(A) E_r(B)`，即使每个 `E_r` 单独是 e-value，也一般有

\[
\mathbb E_0\sum_rw_r(A)E_r(B)>1

\]

当 `A` 与 `B` 相关时尤其如此。安全条件只有三类：

1. 权重是测试前冻结的物理/解剖先验 `pi_r`；
2. selector 与 verifier 在 null 下条件独立，并以合法 cross-fitting 构造；
3. 直接建模 `(A,B)` 的联合 likelihood ratio，再对位置边际化。

普通 attention-weighted pooling 不自动属于其中任何一类。

---

## 9. 最小可测算法：先不命名

目标不是立即宣称新方法，而是测试“证据守恒”是否比 max/top-k 更符合 VLM 局部证据。

### 9.1 Calibration-only 版本

对一个预先指定 claim：

1. 冻结一组互斥原子区域，或冻结连续位置/尺度先验的离散 quadrature；
2. 用同一个冻结局部响应得到每个区域的分数 `s_r`；
3. 仅在 image-disjoint `0/3` development 图上，按 finding 与位置 nuisance stratum 将 `s_r` 转为 super-uniform rank p-value；
4. 用一个预先冻结的 decreasing p-to-e 映射，例如
   `E_r = gamma * p_r^(gamma-1)`, `0<gamma<1`，得到 null 期望不超过 1 的局部 e-value；
5. 使用与区域面积/尺度先验一致的固定权重，计算
   `E_image = sum_r pi_r E_r`；
6. CE 中只有 `E_image >= 1/alpha` 才允许局部模块把 baseline commitment 改成 positive；否则回退 baseline，而不是自动输出 No；OE 中保持 claim 数固定，只作一换一。

这个版本不训练模型权重，但依赖 labeled null calibration；它应称 calibration-only / training-free，而不是 zero-data。

### 9.2 为什么这是最小实现

- 不使用 max，所以没有 hard selector--verifier reuse；
- 权重总和固定为 1，增加离散分辨率时同步拆分先验质量；
- 每个局部分数先被规范为 null-valid evidence，避免把 raw response 当 LR；
- 任意图内 patch 相关仍允许 `E_0[E_image]<=1`；
- 与 raw max、top-k、普通 mean、`logmeanexp(raw margin)-log M` 的差别可直接消融。

### 9.3 最低成本致死实验

使用现有 Huatuo patch-score artifact，不需要新 GPU：

- `M in {16,64,256/361}`，相同物理 ROI family 的嵌套细分；
- development vote-0 只做 p-to-e calibration；confirmation vote-0 测 FP / e-value exceedance；
- vote-3 bbox positives 按病灶面积分层测 recall；
- 聚合器：raw max、penalized scan、mean、top-5%、raw logmeanexp、校准后的先验质量 e-mixture；
- 强制 matched threshold / matched positive rate，避免 e-mixture 因保守而“少说即获益”。

首个 GO 门：

1. raw max/top-k 的 confirmation FP 或正向 margin 随 `M` 显著增长；
2. e-mixture 的 null exceedance 不随 `M` 恶化，且跨 development/confirmation 保持；
3. 在 matched FP 下，focal 3/3 positives 的 recall 比强 base
   `final margin + mean + max + top5` 提高至少 1pp，并在至少两个模型同向；
4. 对 bbox 小病灶的增益大于大病灶，interaction CI 排除 0；diffuse finding 不损失超过 1pp。

第一条失败，说明本数据里没有 resolution/search inflation，关闭整条主线。前三条任一失败，证据守恒聚合只是一个安全但无用的平均器，不晋级方法。

---

## 10. 原创性与 ICLR Oral 严格 verdict

| 对象 | 数学/文献状态 | 当前判断 |
|---|---|---|
| `max` 的 `sqrt(2 log M)` 膨胀 | 经典极值统计 | 非贡献 |
| selector--verifier 的 `rho E[max A]` | 经典高斯选择后偏差 | 非贡献 |
| `a-u_M` 与 `b-rho u_M` 两门边界 | 经典结果的直接联合；表述清晰 | 只可作机制框架 |
| 稀疏病灶 `delta sqrt(k)` 边界 | 经典 scan / sparse detection | 非贡献 |
| `sum pi_r L_r` / `logsumexp-logM` | Bayes 边际化、average LR | 非贡献 |
| mixture e-value 在任意依赖下均值有效 | 标准 e-value closure | 非贡献 |
| 分割细化时同步拆 prior mass | 概率 coherence | 非贡献 |
| **VLM 的局部增强同时受 selection 与 validation 两门限制** | 尚需实证 | 若跨模型/方法成立，有机制新意 |
| **分辨率/ontology 增大导致病例级 FP，而 evidence-conserving aggregation 在 matched recall 下消除** | 尚需实证；BCEA 占据选后校准，但未完全等价于 partition coherence | 有窄方法缝隙 |
| **同一相图统一预测医学小病灶 FN 和通用 VLM local-enhancement FP** | 未验证 | 唯一可能接近 Oral 的版本 |

### 最终审查

**现在不足以完成 ICLR Oral 论文。**

优点是故事一句话可以讲清：

> 局部视觉方法不应把“搜索到的最大响应”当证据；正确的整图证据应对未知位置做先验质量守恒的边际化，而一个病灶只有同时越过定位边界与验证边界才可被安全纠正。

但审稿人也能一句话指出风险：

> 这是把经典 scan/winner's curse 与 average likelihood ratio/e-value 应用到 VLM。

要反驳这个评价，实验必须给出不是经典公式能自动保证的 VLM-specific 规律：

1. 两个坐标能跨局部算法预测真实 FP/FN，明显优于 final margin、面积和方法 identity；
2. 分割/分辨率改变在像素语义不变时系统改变 VLM commitment；
3. evidence-conserving aggregation 在 matched positive rate、fixed-K OE 下同时降低 FP 且不增加 omission；
4. 医学 focal/diffuse 分层与自然图像 small/large object 均落在同一相图；
5. 至少两类模型、三类局部方法复现。

若只得到“平均 LR 比 max 稳定”，更适合作为 strong baseline 或统计修正，不够 ICLR Oral。若 null inflation 第一门不成立，应立即关闭，不能再用 conformal、e-value 或新名字挽救。

---

## 11. 参考与碰撞边界

1. Chan, Walther. *Detection with the Scan and the Average Likelihood Ratio*. 2013. <https://arxiv.org/abs/1107.4344>
2. Walther. *Optimal and Fast Detection of Spatial Clusters with Scan Statistics*. 2010. <https://arxiv.org/abs/1002.4770>
3. Walther, Perry. *Calibrating the Scan Statistic: Finite Sample Performance vs. Asymptotics*. 2022. <https://arxiv.org/abs/2008.06136>
4. Fithian, Sun, Taylor. *Optimal Inference After Model Selection*. 2014. <https://arxiv.org/abs/1410.2597>
5. Xu et al. *Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable Vision-Language Models*. 2026. <https://arxiv.org/abs/2606.16667>
6. Park et al. *SECOND*. ICML 2025. <https://arxiv.org/abs/2506.08391>
7. An et al. *Assembly of Global and Local Attention*. CVPR 2025. <https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html>
8. Zhao et al. *Vision-Guided Attention*. CVPR 2026. <https://arxiv.org/abs/2511.20032>


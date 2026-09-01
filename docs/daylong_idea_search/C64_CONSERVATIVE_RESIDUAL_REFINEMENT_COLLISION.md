# C64 — Conservative Residual Refinement（CRR）独立审计

> 日期：2026-08-13  
> 裁决：**原始 variable-length 版本 NO-GO；不进入 GPU。** 它的守恒式是经典 wavelet/multigrid 的低频替换，系统上是 centered crop feature fusion；而且“feature mean 守恒”不推出冻结 decoder 的函数守恒。

## 1. 公式的精确数学身份

令 ROI 的 coarse token 为 `v`，crop 产生 children `f_1,...,f_m`，非负权重满足 `sum w_j=1`。CRR 定义

\[
\mu_f=\sum_jw_jf_j,
\qquad
f'_j=f_j-\mu_f+v.
\]

于是

\[
\sum_jw_jf'_j=v.
\]

定义 restriction `R f = sum_j w_j f_j`，constant prolongation `P v=(v,...,v)`，则

\[
f'=Pv+(I-PR)f,
\qquad RP=I,
\qquad Rf'=v.
\]

这给出三个正确性质：

1. crop 的共同平移 `f_j -> f_j+a` 被完全消除；
2. children 间差异不变：`f'_i-f'_j=f_i-f_j`；
3. `f'` 是满足 weighted mean 为 `v` 的所有 token sets 中，离原始 `f` 最近的一个：

\[
f'=\arg\min_{g:Rg=v}\sum_jw_j\|g_j-f_j\|_2^2.
\]

第三条是带线性等式约束的标准最小二乘投影，不是新定理。

## 2. Wavelet lifting / multigrid 碰撞

### Wavelet / Laplacian pyramid

Haar wavelet 把 children 分成低频平均与零均值细节，再用

```text
child = lowpass + detail
```

重建。CRR 精确执行“保留 crop 的 detail coefficients，把 lowpass coefficient 换成 coarse VLM token `v`”。一般 lifting scheme 通过 predict/update 构造可逆 coarse/detail 分解，并用 vanishing moments 保证 detail 不携带低阶矩；CRR 是最简单的一阶矩版本。

因此“只注入细节、保持低频不变”是 wavelet/multiresolution analysis 的经典基本单元。

### Conservative multigrid prolongation

multigrid 中 fine state 常写为

\[
x_f=Px_c+d,qquad Rd=0,qquad RP=I.
\]

CRR 就是把 crop feature 投影到 `ker R` 得到 fine detail，再加到 coarse prolongation 上。`restriction(prolongation)=identity` 与守恒插值都是成熟结构。

所以 wavelet/multigrid 提供了很好的解释语言，但不能作为 C64 的数学原创性。

## 3. 与 LLaVA-HR、TokenPacker 的关系

### LLaVA-HR / Mixture-of-Resolution Adaptation

[Feast Your Eyes: Mixture-of-Resolution Adaptation, arXiv:2403.03003](https://arxiv.org/abs/2403.03003) 已使用 low/high-resolution 两条视觉路径，并通过 MR-Adapters 将高分辨率信息嵌入低分辨率路径，目标正是低 token 长度下保留细粒度信息。

C64 的差别是 training-free、显式保持 first moment；但“coarse carrier + high-resolution detail”这一系统抽象已经被占据。

### TokenPacker

[TokenPacker, arXiv:2407.02392 / IJCV 2025](https://arxiv.org/abs/2407.02392) 的 coarse-to-fine projector：先把低分辨率 feature 作为 point query，再用 high-resolution multi-level local regions 作为 keys/values，通过 region-to-point injection 更新 coarse token，同时保持压缩后的 token 数。它已经直接实现“coarse token 吸收相应局部的 fine-grained information”。

- TokenPacker 是 learned/query-dependent fine-to-coarse injection；
- C64 是 deterministic、mean-preserving coarse-to-fine expansion。

二者不完全同式，但若把 C64 修正为固定 token 数，就自然变成

\[
g_i=v_i+\sum_j a_{ij}(f_{ij}-\mu_i),
\]

即“对 TokenPacker/region cross-attention 的 values 做 centering”。这只是 centered coarse-fine feature fusion，而不是新基本计算。

## 4. 原始版本最致命的问题：mean 守恒不等于 decoder 守恒

C64 用 `m` 个 children 替换一个 `v`。即使 crop 没有任何 detail，所有 `f'_j=v`，标准 decoder 看到的仍是 `m` 个 `v` 而不是一个 `v`：

- softmax denominator 中有 `m` 份 key；
- visual/text attention mass 改变；
- causal decoder 中 children 的可见前缀不同；
- 后续文字 position/RoPE 改变。

因此一般有

\[
F([\ldots,v,\ldots])
\ne
F([\ldots,v,\ldots,v,\ldots]).
\]

换句话说，CRR 只守恒输入 embedding 的算术平均，却没有守恒生成计算。它无法保证“不产生 crop criterion shift”。要修复必须：

1. 同时引入 C63 的 measure/proportional attention；并处理 causal visual order；或
2. 用 TokenPacker 式固定长度 cross-attention 将 details 压回 coarse slots。

前者已碰撞 ToMe/quadrature attention且不适配原生 causal VLM，后者正是 centered feature fusion。

## 5. softmax 下 detail 如何生效

在线性 key/value projection 后写成

\[
K_j=K_0+\Delta K_j,quad V_j=V_0+\Delta V_j,quad
\sum_jw_j\Delta K_j=\sum_jw_j\Delta V_j=0.
\]

children-only cross-attention 输出为

\[
V_0+
\frac{\sum_jw_je^{q^T\Delta K_j}\Delta V_j}
{\sum_jw_je^{q^T\Delta K_j}}.
\]

零均值只消除均匀平均的一阶 common mode；detail 通过 query-key 与 value residual 的相关项进入。这个解释很干净，但它也是对 centered attention pooling 的 Taylor/协方差解释，不提供 label correctness：crop artifact、边界、resize 高频同样位于 zero-mean residual 中。

## 6. 与本地证据的关系

支持动机的结果：

- 小病灶面积与正确 margin 在 Huatuo/Hulu 多 split 相关，说明 coarse full view 可能遗漏局部细节；
- crop/context-removed 阴性 FP 从 full `8.1%` 升到约 `62.9–71.0%`，说明 crop 带来很强 common criterion shift。

反对直接晋级的结果：

- full → crop AUROC 仅 `+0.0035`，CI `[-0.0623,+0.0730]`；响应主要是工作点平移，而非新增病例排序；
- `full+crop` 的 post-hoc 增量未在 fresh、统一 selector 下确认；
- full-image patch mean/max/top-k/multiscale scan 的增量门失败；
- 现有缓存只有 full-image patch scores 或 full/crop 最终 margins，没有 matched crop child embeddings，无法从 cache 恢复 `f_j-mean(f)`。

所以“高分辨率 zero-mean residual 中仍有 label 增量”是尚未直接测试的假设，但不是已有正结果。

## 7. 为什么不运行原始 ≤16 canary

一个合法的 canary 必须先过 zero-detail identity：把所有 detail 置零后，输出应回到 native full。原始 variable-length C64 按架构必然不满足这一点，因此任何准确率变化都混合 token multiplicity、position shift 与 residual detail。

若未来只把它当 TokenPacker/LLaVA-HR 的消融，可构造固定长度版本：每个 coarse ROI slot 用 crop children 的 centered cross-attention residual 更新，token 数和 position 全不变。冻结 8 个小病灶阳性和 8 个 finding-matched 阴性，比较：

1. native full；
2. raw fine-feature injection；
3. centered residual injection；
4. spatially shuffled centered residual；
5. norm-matched random zero-mean residual。

必要门：

- zero-detail arm 与 full margin 差 `<1e-4`；
- 8 个阳性中至少 6 个 centered residual 相对 full 提高正确 margin，且显著超过 shuffled/random；
- 8 个阴性中不超过 1 个变成 FP；
- centered arm 至少消除 raw injection 的 80% 阴性 common shift；
- 效果不能仅由 residual norm 预测。

但即使全部通过，也只说明“mean-centering 是 coarse/fine projector 的有效安全修正”，不足以把经典 lifting + feature fusion 升为 ICLR Oral 主线。因此当前不占 baseline GPU。

## 8. 最终裁决

| 问题 | 判断 |
|---|---|
| weighted mean 是否严格保持 | 是 |
| 数学对象是否新 | 否：affine projection / Haar detail substitution / conservative prolongation |
| 是否区别于普通 raw concat | 是：严格消除 crop common translation |
| 是否区别于 coarse/fine fusion 大类 | 否 |
| 与 LLaVA-HR / TokenPacker exact 相同 | 不完全；剩余差异主要是一行 centering constraint |
| 是否保证生成计算不变 | 否；variable length、softmax、causal order、RoPE 均破坏 |
| 是否有未测 empirical hypothesis | 有：crop zero-mean detail 的 conditional label value |
| 该 hypothesis 是否足以救 Oral 新颖性 | 否 |
| 是否当前运行 GPU | **否** |

结论：CRR 是一个合理、可能有效的 multiresolution projector 消融，但不是新的 hallucination mitigation 基本单元。


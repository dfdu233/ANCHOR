# C70 — XRV Local Counterexample Householder：严格代数与接口审计

日期：2026-08-13  
资源边界：只读已有 XRV/VLM 缓存并使用 CPU；未占用 GPU，未改变 baseline 队列。

## 裁决

**NO-GO as a new Householder / mirror intervention.**

新的正信号是真实的：冻结 XRV 的 18 维病理 logit 向量，相对 development 中同病种的阳性、
阴性局部邻居所形成的边界，在 `VLM margin + 目标病种 XRV scalar` 之外仍有病例级增量：

| 模型 | kNN boundary macro-AUROC 增量 | image-bootstrap 95% CI |
|---|---:|---:|
| Huatuo | +0.0261 | [+0.0140, +0.0383] |
| Hulu | +0.0098 | [+0.0032, +0.0166] |

但把这套几何写成 Householder 镜像不会得到第二种证据，也不能天然控制冻结 VLM。它有两个
严格原因：

1. 对最近的阴性/阳性锚点，Householder 的有符号坐标**精确等于平方距离差的缩放**；
2. XRV logit 不在 VLM 的因果计算路径中，镜像 XRV 向量本身不会改变 VLM 的任何 token。

任何让镜像产生输出变化的 bridge，必然重新落入 image counterfactual、activation steering、
expert-guided decoding 或 post-hoc verification。因此本候选不能作为“全新直接缓解原语”进入 GPU。

## 1. 正确保留的自然现象

令冻结 XRV 对一张胸片输出

\[
z(x)\in\mathbb R^{18},
\]

18 个坐标对应 XRV 的 18 个病理概念。对目标 finding `c`，分别从 development 中取三位最近
阴性与三位最近阳性邻居，定义

\[
b_c(z)=\overline d(z,N_c^-)-\overline d(z,N_c^+).
\]

这里正值表示病例在完整 18 维“共病谱”中更接近阳性病例；这与只看 `c` 的单个 XRV logit
不同。finding 内随机打乱标签后增量约为零，因此该信号不是 kNN 维数本身制造的。

这个结果应保留为：**多病种联合几何含有单病种 posterior 丢掉的信息。** 它不是目前方法。

## 2. Householder 镜像为何是同一个对象

取离当前病例最近的阴性锚点 `a0` 和阳性锚点 `a1`，令

\[
n={a_1-a_0\over\|a_1-a_0\|},\qquad
m={a_0+a_1\over2}.
\]

穿过中点 `m`、法向为 `n` 的垂直平分面把两锚点分开。病例在这条法向上的有符号坐标是

\[
s(z)=(z-m)^\top n.
\]

对应的 Householder 反射为

\[
H(z)=z-2s(z)n.
\]

它满足三个漂亮但标准的性质：

\[
H(H(z))=z,
\]

即镜像两次回到原点；并且交换到两锚点的距离：

\[
\|H(z)-a_0\|=\|z-a_1\|,
\quad
\|H(z)-a_1\|=\|z-a_0\|.
\]

决定性等式是

\[
\|z-a_0\|^2-\|z-a_1\|^2
=2\|a_1-a_0\|s(z).
\]

因此 `s(z)` 不是一种新的 counterfactual evidence；它就是最近正负反例平方距离差换了坐标。
本地数值误差均在 `1.5e-14` 以内，验证了 involution、锚点距离交换和上述恒等式。

## 3. CPU 致死实验

在相同 development/confirmation、相同 finding one-hot、VLM margin 与 XRV scalar 基础上，
比较 kNN boundary、Householder 坐标以及二者联合：

| 模型 | scalar base | +kNN boundary | +Householder | +二者 | 二者相对 kNN 95% CI |
|---|---:|---:|---:|---:|---:|
| Huatuo | 0.8264 | **0.8525** | 0.8442 | 0.8460 | [-0.0130, +0.0001] |
| Hulu | 0.8708 | **0.8806** | 0.8788 | 0.8788 | [-0.0058, +0.0022] |

Householder 不仅没有在 kNN boundary 之外增加信息，点估计还更低。原因不是超参数，而是最近
两锚点坐标丢掉了三邻居平均所表达的局部密度；“反射”并未创造新病例证据。

## 4. 为什么它不能直接干预冻结 VLM

当前系统的因果图是

```text
image ──> frozen VLM ──> text
   └────> frozen XRV ──> z, H(z)
```

`H(z)` 位于旁路。若不再建立一条边回到 VLM，则

\[
P_\text{VLM}(y\mid x,H(z))=P_\text{VLM}(y\mid x),
\]

因为 VLM 根本没有读取 `H(z)`。建立 bridge 只有四种实际位置：

| bridge | 实际方法族 | 直接碰撞 |
|---|---|---|
| 由 `z→H(z)` 生成/编辑第二张图 | visual counterfactual / two-pass CD | VCD、RVCD、CIPHER、CounterVHD |
| 把锚点法向翻译成 VLM hidden direction | activation projection/steering | VTI、CIPHER、MIDSTEER / Householder editing |
| 用 boundary 改 next-token / claim logits | expert fusion / guidance | MARINE、CGD、Expert-CFG、LEAD、CoFE |
| 生成后按 boundary 删除或改 claim | detector/verifier/veto | CoEV、CounterVHD、本地 C47/C65 |

这不是“还没想到 clever bridge”；在不训练共享 codebook 的异构模型中，任何非平凡影响都必须
通过图像、VLM 内部状态、决策分布或输出之一进入。Householder 只规定如何在 XRV 空间变换，
不提供跨模型语义映射。

## 5. 2024–2026 精确碰撞邻域

- [RVCD, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.430/) 已用检测器识别幻觉
  对象，再用显式正/负概念图像的 VLM logits 调节解码；“外部专家 + 反例几何 + VLM”若经
  reference images 实现，直接进入该接口。
- [MARINE, ICML 2025](https://proceedings.mlr.press/v267/zhao25j.html) 已用开源视觉模型的
  对象级信息引导 LVLM，并支持多视觉模型；普通专家信息注入不新。
- [Expert-CFG, ICCV 2025](https://arxiv.org/abs/2507.09209) 已将专家高亮经 classifier-free
  guidance 写入医学 VLM。
- [CIPHER, CVPR 2026](https://hamidreza-dastmalchi.github.io/cipher-cvpr2026/) 从视觉反事实
  构造 hallucination subspace，并在推理时正交投影 hidden states；把局部锚点法向接入 VLM
  hidden state 属于其近邻。
- [RVCD](https://aclanthology.org/2025.findings-acl.430/) 与
  [CounterVHD](https://arxiv.org/abs/2606.28520) 分别覆盖外部正负视觉反例的解码与医学
  counterfactual grounding verifier。
- 2026 的 MIDSTEER/通用 activation-editing 工作已明确把 Householder reflection 作为仿射
  steering operator；Householder 本身不能作为数学新颖性。

## 6. 最小结论与下一合法放宽

可继续研究的不是“XRV Householder decoding”，而是这个更窄的新现象：

> claim truth depends on a patient's location in a joint disease manifold, not only the
> target disease posterior.

但在当前排除 `fusion / veto / rerank / prompt / training` 的约束下，它没有新的直接缓解接口。
若要把这个正信号转成方法，必须至少放宽一项：

1. **允许 `<1%` bridge training**：学习 XRV 局部切空间到 VLM visual-token 切空间的等距映射；
   然后与普通线性 adapter、CIPHER 投影、LEAD fusion 等量对照；或
2. **允许决策级 expert guidance**：直接用 kNN boundary 作强 baseline，但诚实定位为新的
   joint-disease counterexample score，而非新的 Householder 数学。

在用户当前严格条件下，C70 判为 **NO-GO；不排 GPU**。正信号保留，不被“方法失败”抹掉。

## 7. 产物

- 脚本：`anchor/corrected_sgta/audit_xrv_householder_reflection_v1.py`
- 结果：`corrected_runs/daylong_idea_search_v1/xrv_householder_reflection_v1/result.json`
- 上游：`corrected_runs/daylong_idea_search_v1/xrv_counterexample_geometry_v1/result.json`


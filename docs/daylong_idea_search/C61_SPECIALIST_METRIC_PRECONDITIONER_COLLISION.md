# C61 — Small-Specialist Metric / Preconditioner / Kernel 审计

审计日期：2026-08-13  
范围：只做公式硬门、2023--2026 碰撞和本地证据核对；不排 GPU。  
裁决：**没有公式级唯一候选，整个候选族 strict NO-GO。**

## 1. 冻结问题

小医学专家不能输出病种结论，也不能参与 veto、exchange、RAG、logit fusion、
CAM mask、crop、foveation、rerank 或 verification。唯一允许的信息是每个 patch 的
专家表征

\[
S=(s_1,\ldots,s_n),
\]

并希望它只定义视觉 token 的 Mahalanobis metric、preconditioner 或 attention
kernel，让大 VLM 在一次前向中更好地组织原有视觉证据。

本轮要求的硬条件是：所得运算不能只是 attention bias、Q/K adapter、feature fusion
或普通 token mixing。

## 2. 全局 Mahalanobis metric 必然可吸收到 Q/K

设某层 query、key、value 为 \(Q,K,V\)，专家产生输入相关的半正定矩阵
\(M_S\succeq0\)。候选 attention 为

\[
\operatorname{Att}_{M_S}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QM_SK^\top}{\sqrt d}
\right)V.
\]

由于任意半正定矩阵均存在平方根

\[
M_S=P_SP_S^\top,
\]

于是

\[
QM_SK^\top
=
(QP_S)(KP_S)^\top.
\]

所以 specialist Mahalanobis attention 精确等于普通 dot-product attention，只是
对 Q/K 使用同一个输入相关线性 adapter。它没有新的 attention 对象。

更一般地，若 \(M_S\) 不要求对称或正定，任意矩阵都可分解为
\(M_S=A_SB_S^\top\)，仍有

\[
QM_SK^\top=(QA_S)(KB_S)^\top.
\]

因此放弃 Mahalanobis 性质也只会变成两套动态 Q/K adapter。

## 3. pair-dependent metric 必然是 attention bias

为了避开全局因子分解，可以让每一对 token 使用不同矩阵 \(M_{ij}(S)\)：

\[
\ell_{ij}^{S}=q_i^\top M_{ij}(S)k_j.
\]

但相对于原 attention logit

\[
\ell_{ij}^{0}=q_i^\top k_j,
\]

定义

\[
b_{ij}(S,Q,K)=\ell_{ij}^{S}-\ell_{ij}^{0},
\]

便得到恒等式

\[
\ell_{ij}^{S}=\ell_{ij}^{0}+b_{ij}.
\]

无论 \(M_{ij}\) 如何复杂、是否由小专家动态产生，只要最终仅为每个 query-key pair
给出一个标量，它在执行层面就是 additive attention bias。把它叫 curvature、
geodesic、clinical kernel 或 local metric 不改变该代数。

## 4. multiplicative specialist kernel 仍是 bias 或 feature fusion

设专家定义正 kernel

\[
\kappa_S(i,j)>0.
\]

若把它乘到原 softmax kernel，

\[
A_{ij}^{S}
\propto
\exp(\ell_{ij}^{0})\kappa_S(i,j),
\]

则

\[
A_{ij}^{S}
\propto
\exp\left(\ell_{ij}^{0}+\log\kappa_S(i,j)\right),
\]

仍是 attention bias。

若 \(\kappa_S\) 是正定 Gram kernel，存在特征映射 \(\phi_S\)，使

\[
\kappa_S(i,j)
=
\langle\phi_S(s_i),\phi_S(s_j)\rangle.
\]

base kernel 与 specialist kernel 的乘积对应 tensor-product feature map。因此其
另一种实现只是把 VLM feature 与 specialist feature 放入联合空间，即 feature
fusion。两种解释均被硬约束排除。

## 5. preconditioner 放在其他位置也没有逃逸

### 5.1 对 token feature 做预条件

\[
\widetilde V=VP_S
\]

是输入相关线性 feature adapter；若 \(P_S\) 来自专家 covariance whitening 或
coloring，则属于 whitening、CORAL 或 covariance alignment。

### 5.2 对 value 做预条件

\[
Y=A(VP_S)=(AV)P_S
\]

是 attention 输出后的 feature transform，不改变 token 选择机制。

### 5.3 对 token 轴做 graph preconditioning

设专家 patch graph 的 Laplacian 为 \(L_S\)，常见更新

\[
\widetilde V=(I+\lambda L_S)^{-1}V
\]

是一个固定于当前图像的 token mixing matrix。它等价于 graph diffusion 或一层
expert-conditioned attention/value aggregation；仍是 feature fusion，而且可能把小病灶
与正常邻域平滑在一起。

### 5.4 natural-gradient / proximal 更新

若用专家 metric \(G_S\) 对某个解码损失 \(\mathcal L\) 做一步更新，

\[
V^+
=
V-\eta G_S^{-1}\nabla_V\mathcal L,
\]

则这是 preconditioned test-time optimization。若 \(\mathcal L\) 是 claim 验证或
生成能量，就回到 verification/energy guidance；若没有 \(\mathcal L\)，metric
本身没有规定应朝哪个临床方向移动。

## 6. 跨空间接口还有一个不可消除的 translator

专家 feature \(s_i\in\mathbb R^p\) 与 VLM query/key
\(q_i,k_i\in\mathbb R^d\) 通常不在同一坐标系。若要把专家 covariance 变成
\(d\times d\) metric，必须有 translator \(B:\mathbb R^d\to\mathbb R^p\)，例如

\[
M_S=B^\top\Sigma_S^{-1}B.
\]

- 全局学习 \(B\)：训练过的 cross-model adapter；
- 每图用 patch 对做 least squares / CCA / Procrustes：输入相关 feature alignment；
- 不使用 \(B\)，只在 token-pair 轴使用专家距离：回到 pairwise attention bias。

所以 small specialist 不是无代价地提供一个 VLM metric；跨模型坐标对齐本身就落入
adapter/fusion 家族。

## 7. 2023--2026 直接碰撞

### Elliptical Attention, NeurIPS 2024

该工作直接把标准 attention 改成

\[
H
=
\operatorname{softmax}
\left(
\frac{QMK^\top}{\sqrt D}
\right)V,
\]

使用 Mahalanobis geometry 将球形邻域变为椭球邻域，并给出无需额外可学习参数的
坐标重要性估计以及 kernel-regression 方差分析。它精确覆盖本候选最自然的公式，
而且已经声称 robustness 与防 representation collapse。

https://papers.nips.cc/paper_files/paper/2024/file/c63908a3e946af0e7978c23737229137-Paper-Conference.pdf

### DARKFormer, ICLR 2026 DeLTa workshop

该工作使用

\[
q^\top\Sigma k,\qquad \Sigma=M^\top M,
\]

构造 data-aware Mahalanobis softmax kernel，并从 importance sampling 与 anisotropic
query/key geometry 解释其作用。它覆盖数据依赖 kernel geometry 这一叙事。

https://openreview.net/pdf?id=dN3My5i8sl

### Whitened Self-Attention / Preconditioned Attention, 2025--2026

Whitened Self-Attention 用 covariance whitening 修正 token correlation；
Preconditioned Attention 把 conditioning matrix 直接加入 attention head 以改善
condition number。二者进一步压缩了 covariance/preconditioning 的新颖性空间。

https://openreview.net/forum?id=XQ0VTUIhEJ

https://arxiv.org/abs/2603.27153

### VLM hallucination 邻域

ARCD、Vision-Guided Attention 与 SPIN 已直接通过外部区域、visual semantic
confidence 或 image-guided head signal 调整视觉 attention。即使 specialist metric
来源不同，只要最后变成 patch-pair bias 或 attention weighting，仍落在该系统家族。

https://ojs.aaai.org/index.php/AAAI/article/download/37620/41582

https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_Tell_Model_Where_to_Look_Mitigating_Hallucinations_in_MLLMs_by_CVPR_2026_paper.pdf

https://aclanthology.org/2025.emnlp-main.631/

## 8. 本地证据约束

- C41 Polar Projector Rectification 已检查视觉 projector 几何：Huatuo finding
  directions 没有系统落在弱奇异方向，Hulu 的 7/7 finding directions 反而被明显
  放大，因此没有“专家 metric 应纠正统一压缩”的本地前提。
- C46 小专家条件增量只在 Huatuo 强，在 Hulu 未过联合门；不存在跨模型稳定的
  specialist geometry 正信号。
- C54 specialist-as-encoder 已证明跨模型 side input 需要 shared codebook 或 adapter。
- C59 CAM-Sinkhorn 已证明把专家空间关系投影到 attention 最终就是 patchwise logit
  bias。

所以即使忽略碰撞，也没有本地已确认现象自然要求一次 Mahalanobis
preconditioning。

## 9. 公式硬门结论

| 专家几何的作用位置 | 精确等价物 | 决定 |
|---|---|---|
| 全局 \(M_S\) 进入 \(Q M_S K^\top\) | dynamic Q/K adapter | NO-GO |
| pair-specific \(M_{ij}\) | additive attention bias | NO-GO |
| positive specialist kernel 相乘 | log-kernel bias | NO-GO |
| PSD kernel lift | tensor-product feature fusion | NO-GO |
| token/value preconditioner | dynamic feature adapter | NO-GO |
| graph Laplacian inverse | token diffusion/mixing | NO-GO |
| natural-gradient step | test-time optimization/energy guidance | NO-GO |

在当前约束下不存在可以同时满足下列条件的候选：

1. 真正使用 small specialist 的 metric；
2. 改变大 VLM 的视觉计算；
3. 不化为 attention modulation；
4. 不化为 feature adapter/fusion；
5. training-free；
6. 一次低时延前向。

因此不设计 CPU 致死实验，也不进入 GPU。下一候选必须让小模型提供的对象不是
输出值、空间权重、feature 坐标或 pairwise similarity；否则都会被上述三分法吸收。

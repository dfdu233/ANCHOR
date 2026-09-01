# C59 — Specialist Broyden Fast-Weight 与 CAM-Sinkhorn 审计

## 总结

严格 NO-GO；未占用 GPU，也不进入效果实验。

两个候选都满足形式上的优雅性质，但这些性质恰好把它们唯一地确定为已有原语：

1. specialist state 作为 secant constraint 的最小 Broyden 更新，等价于条件残差叠加；在当前样本上是 specialist 硬替换，加入步长后是凸 expert fusion。
2. specialist CAM 作为列边际的最小 KL 投影，等价于 Sinkhorn/IPFP attention reweighting；在 attention logit 空间中就是逐 patch 加偏置，属于 soft mask。

因此它们没有绕开已关闭的 expert fusion、test-time editing 和 attention-mask 路线。

## A. Specialist Broyden / delta-rule 临时更新

设当前局部视觉到语言映射为

\[
y=Wk,
\]

其中 \(k\) 是当前图像的视觉状态，\(W\) 是冻结 VLM 的局部线性映射；小模型给出临床状态 \(s\)。要求用最小改动满足 secant constraint：

\[
\min_{\widetilde W}\|\widetilde W-W\|_F^2
\quad
\text{s.t.}\quad \widetilde Wk=s.
\]

唯一解是 good-Broyden / rank-one least-change update：

\[
W^+
=
W+\frac{(s-Wk)k^\top}{k^\top k}.
\]

### 三个要求

- Identity-if-agree：若 \(s=Wk\)，更新为零。
- Idempotent：第一次更新后 \(W^+k=s\)，再次施加同一约束不再改变。
- Orthogonal-input preservation：若 \(k^\top q=0\)，则 \(W^+q=Wq\)。

但对任意输入 \(q\)，有

\[
W^+q
=
Wq+
\frac{k^\top q}{k^\top k}(s-Wk).
\]

这正是一个以 key 相似度为门控的 conditional residual stack。特别地，对当前样本 \(q=k\)：

\[
W^+k=s.
\]

也就是说，精确 secant 更新在目标样本上不是“融合证据”，而是把 VLM 状态完整替换成 specialist state。若加入步长 \(\beta\)：

\[
W_\beta^+k=(1-\beta)Wk+\beta s,
\]

则精确退化为 convex expert fusion。

若用协方差加权距离

\[
\min_{\widetilde W}
\operatorname{tr}\!\left[
(\widetilde W-W)C(\widetilde W-W)^\top
\right],
\]

解为

\[
W^+
=
W+
(s-Wk)
\frac{(C^{-1}k)^\top}{k^\top C^{-1}k},
\]

即 ROME 类受限最小二乘更新；所谓“保持正交输入”只是把正交性换到 \(C^{-1}\) 度量。

对非线性模型做一阶展开后，最小参数改动为

\[
\delta
=
J^\top(JJ^\top)^\dagger
\bigl[s-F_\theta(k)\bigr],
\]

属于标准 min-norm test-time model editing。若在 activation 上做，则又回到残差注入。

### OE 中的额外问题

specialist logits 与 VLM hidden state 没有天然共享坐标。任何把二者对齐的 translator 都会成为训练过的 adapter / stacking module；把 specialist 文本嵌入直接写入，则是显式 expert guidance。固定 finding ontology 也不能自然扩展到开放 claim。

### 直接碰撞

- ROME, NeurIPS 2022：rank-one least-change model editing  
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html
- Linear Transformers Are Secretly Fast Weight Programmers, ICML 2021  
  https://proceedings.mlr.press/v139/schlag21a.html
- Parallelizing Linear Transformers with the Delta Rule, 2024  
  https://arxiv.org/abs/2406.06484
- Gated Delta Networks, 2024  
  https://arxiv.org/abs/2412.06464
- M3Bench：medical VLM model editing, 2026  
  https://arxiv.org/abs/2607.05310

结论：公式级 NO-GO。

## B. Specialist-CAM 作为边际的 KL-Sinkhorn 投影

设 decoder 对视觉 token 的 attention 子矩阵为

\[
A\in\mathbb R_+^{T\times P}.
\]

保留每个生成位置原有的总视觉质量

\[
r=A\mathbf 1,
\]

并令 specialist CAM \(\mu\in\Delta^P\) 指定目标 patch 列边际

\[
c=M\mu,\qquad M=\mathbf 1^\top r.
\]

最小改动投影为

\[
B^*
=
\arg\min_{B\ge 0}\operatorname{KL}(B\|A)
\quad
\text{s.t.}\quad
B\mathbf 1=r,\quad B^\top\mathbf 1=c.
\]

KKT 条件给出唯一的矩阵缩放形式

\[
B^*=\operatorname{diag}(u)A\operatorname{diag}(v),
\]

其中 \(u,v\) 由 Sinkhorn/IPFP 迭代求得。逐行写为

\[
B_{tp}^*
=
r_t
\frac{A_{tp}v_p}{\sum_j A_{tj}v_j}.
\]

若原 attention 为 \(A_{tp}\propto \exp(\ell_{tp})\)，则

\[
B_{tp}^*
\propto
\exp(\ell_{tp}+\log v_p).
\]

因此 Sinkhorn 的全部作用，是计算一个满足 CAM 列边际的逐 patch attention-logit bias：

- \(v_p=0\)：hard mask；
- \(v_p>0\)：soft mask / reweighting。

Identity-if-agree、idempotence 和最小 KL 都是 I-projection 的标准性质，并不产生新的 mitigation 原语。

### OE 中的额外问题

- claim-specific CAM \(\mu_c\) 需要先知道 claim \(c\)，与开放生成顺序冲突；
- 对 ontology 中全部 claim 逐一求 CAM，本质上是 specialist gating，多次前向；
- 聚合 CAM 会丢失 claim identity；
- CAM 表示模型归因位置，不等同于临床支持，尤其不能区分病灶证据与共现捷径。

### 直接碰撞

- Sinkformers, AISTATS 2022：Sinkhorn-normalized attention  
  https://proceedings.mlr.press/v151/sander22a.html
- PLOT, 2022：用 Sinkhorn/OT 对齐视觉语言局部表示  
  https://arxiv.org/abs/2210.01253
- Attention Prompting on Image, 2024：视觉提示重加权 attention  
  https://arxiv.org/abs/2409.17143
- SPIN, EMNLP 2025：image-guided head suppression  
  https://aclanthology.org/2025.emnlp-main.631/
- Image Token Attention Guided Decoding, NAACL 2025  
  https://aclanthology.org/2025.naacl-long.75/
- Scalpel, WACV 2026：OT 对齐 attention activation manifolds  
  https://openaccess.thecvf.com/content/WACV2026/html/Shi_Scalpel_Fine-Grained_Alignment_of_Attention_Activation_Manifolds_via_Mixture_Gaussian_WACV_2026_paper.html

结论：公式级 NO-GO。

## 为什么不进入 CPU/GPU 实验

本轮失败不是“效果可能不好”，而是候选在代数上已经等于已知且已关闭的原语：

- 把 specialist value 写入临时权重，不能逃离 expert fusion；exact secant 是硬替换，partial secant 是凸融合。
- 把 specialist map 写成 attention 边际，不能逃离 attention mask；KL projection 只是求出最小的 patchwise logit bias。

运行效果实验只会重复测试这些旧家族，不能验证一个新的机制。

## 对下一候选的约束

后续候选必须同时避免：

1. 把 specialist 输出当作目标 value；
2. 把 specialist heatmap 当作目标 attention；
3. 把已有读出做线性、凸或核相似度融合；
4. 仅用最小范数、正交保持或 KL 投影给旧操作换数学表述。

真正的新原语需要来自一个病例级、可验证、非输出值也非 attention 目标的计算约束。

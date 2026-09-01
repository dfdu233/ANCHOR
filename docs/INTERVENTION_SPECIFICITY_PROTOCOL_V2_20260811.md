# When Response Is Not Evidence：Intervention Specificity Protocol v2

日期：2026-08-11  
状态：**冻结为机制存在性审计；NCD-as-mitigation 暂停**

## 1. 数值矛盾核查与正式更正

`evidence_admission_feasibility_v1/huatuo.json` 的 `macro_auroc_delta=-0.1091` 是真实且必须
报告的 NO-GO，但它比较的不是 `visual_mean:21` 与最终层：

| 口径 | 非最终候选 | 最终对照 | Confirmation | 差值/结果 |
|---|---|---|---:|---:|
| evidence-admission blind selection | `claim:7` | `claim:28` | 0.6452 vs 0.7543 AUROC | **−0.1091**，pooled CI [−0.1651,−0.0789] |
| static visual family | `visual_mean:21` | `visual_mean:28` | 0.7433 vs 0.7158 AUROC | +0.0275 |
| visual vs final claim readout | `visual_mean:21` | `claim:28` | 0.7433 vs 0.7543 AUROC | −0.0110 |
| frozen equal fusion | visual-21 + final margin | final margin | 0.7167 vs 0.6875 BAcc | +2.9167pp，CI [+0.5936,+5.1930]pp |

所以 `−0.1091` 与 `+2.92pp` 不是同一统计量，不构成算术矛盾：较弱的单独 predictor 仍可能
与最终 predictor 有不同误差而产生融合增益。但是原文的科学叙事确实过强，原因是：

1. 正式 evidence-admission 路线明确失败；
2. 融合的预注册门是 `≥3pp`，实际以 `0.0833pp` 未通过；
3. ROI-vs-background AUC 仅 0.5449，未证明局部临床证据；
4. 只有 Huatuo 的提示性融合结果，未跨模型。

正式更正：`visual_mean:21` 只保留为 **gate-failed suggestive observation**，不能再作为 NCD
或任何 mitigation 的正立论支柱。

## 2. 论文问题重新冻结

旧问题：“能否用 NCD 从中间层提取正确临床证据？”——暂停，因为尚未证明存在可提取证据。

新问题：

> **一次 hallucination intervention 的输出响应，有多少是 image-wide / claim-wide 的非特异
> 工作点移动，有多少是真正携带标签信息的 patient×claim interaction？这个 interaction 的
> 信息量能否预测该 intervention 的可修复上限？**

主贡献类型改为 **New Measurement / Mechanism Boundary**，不是先验假定有效的 decoding method。
NCD 只在 interaction 存在性通过后作为自然推论出现。

## 3. 从一维 NCD 升级为双向特异性分解

对方法 `a`，构造完整的图像×病种响应矩阵：

\[
\Delta^{a}_{ic}=m^{a}_{ic}-m^{0}_{ic}.
\]

更完整的结构模型是：

\[
\Delta^{a}_{ic}=\mu_a+u^{a}_i+v^{a}_c+h^{a}_{ic}+\epsilon^{a}_{ic}.
\]

- \(\mu_a\)：方法整体让模型更肯定/更否定；
- \(u_i^a\)：该图像上所有病种共同移动，例如“整张图显得更异常”；
- \(v_c^a\)：某个病种或词汇在所有图像上被方法稳定推高；
- \(h_{ic}^a\)：唯一可能成为临床证据的 patient×claim interaction。

旧 NCD 只减 \(u_i\)，无法消除 `pleural effusion` 等 claim 自身的全局 token/base-rate 响应；
因此正式版本必须同时消除行效应和列效应。

均值版本定义：

\[
P_N=I_N-\frac1N\mathbf 1\mathbf 1^\top,\qquad
P_C=I_C-\frac1C\mathbf 1\mathbf 1^\top,
\]

\[
H^a=P_N\Delta^aP_C.
\]

部署时使用 robust median-polish：训练折估计冻结的 claim effect \(\hat v_c\)，新图像从全部
ontology claims 估计 image effect \(\hat u_i\)，再得到：

\[
\hat h_{ic}=\Delta_{ic}-\hat\mu-\hat u_i-\hat v_c.
\]

暂称该测量为 **Intervention Specificity Decomposition（ISD）**；只有通过 admission gate 后，
才把 \(m'_{ic}=m^0_{ic}+\lambda\hat h_{ic}\) 称为 Interaction-Gated Decoding。

## 4. 数学结论

### 4.1 唯一交互投影定理

令

\[
\mathcal N=\{u\mathbf1_C^\top+\mathbf1_Nv^\top\}
\]

为所有 image-additive 与 claim-additive 非特异响应组成的 nuisance 子空间，令

\[
\mathcal H=\{H:H\mathbf1_C=0,\ \mathbf1_N^\top H=0\}
\]

为 patient×claim interaction 子空间。则 \(\mathcal H=\mathcal N^\perp\)，且

\[
P_N\Delta P_C
\]

是 \(\Delta\) 到 \(\mathcal H\) 的唯一最小 Frobenius 距离正交投影。因此所有纯 image-wide、
claim-wide 以及任意加性域漂移均被精确消去。

### 4.2 特异性能量分解

正交性给出：

\[
\|\Delta\|_F^2
=\|\Pi_{\mathcal N}\Delta\|_F^2+\|P_N\Delta P_C\|_F^2.
\]

可定义 response specificity ratio：

\[
\mathrm{RSR}(a)=
\frac{\|P_N\Delta^aP_C\|_F^2}{\|\Delta^a\|_F^2}.
\]

但 RSR 只衡量“特异”，不衡量“正确”；随机噪声也可有高 RSR，必须再做标签方向性检验。

### 4.3 可修复性的信息上界

令 \(Z=(m^0,c)\) 表示 baseline margin 和 claim 身份，\(H\) 为 ISD residual。增加 \(H\) 后的
最优 Bayes accuracy 增益满足：

\[
\mathrm{Acc}^*(Y\mid Z,H)-\mathrm{Acc}^*(Y\mid Z)
\le
\sqrt{\frac12 I(Y;H\mid Z)}.
\]

证明由 Bayes decision 的增益不超过条件分布 total variation，再用 Pinsker 不等式和 Jensen
不等式得到。特别地，当

\[
I(Y;H\mid m^0,c)=0
\]

时，任何只依赖该 residual 的 post-hoc decoder 都不可能提高最优准确率。这给出了比“先做 NCD
看分数”更严格的 admission law：**先证明 residual 含有 baseline 之外的条件标签信息。**

## 5. 唯一前置实验：Specificity Existence Gate

### 5.1 数据与拆分

- VinDr 320 张唯一图像；每张图对固定 8 findings 全部评分，得到 2,560 claims/model。
- 采样：每张图随机指定一个 anchor finding；`8 findings × 4 reader-vote bins × 10 images`。
- 模型：Huatuo、Hulu。
- 5-fold image-disjoint cross-fitting；所有阈值、claim effects 和 \(\lambda\) 只在训练折估计。
- 真值：原始三位 reader 的完整 vote vector；不需要额外医生 review。

### 5.2 干预

先测四类已有且机制不同的 intervention，不发明新模块：

1. VCD clean−corrupted image；
2. LET/intermediate−final response；
3. SPIN/head-suppressed−native；
4. style/source-render−canonical。

每种方法都保存完整 \(\Delta^a\)，同时构造：

- raw response；
- image-only centering（旧 NCD）；
- two-way ISD；
- common-mode-only reconstruction；
- row/column-preserving shuffled interaction；
- random/norm-matched control。

### 5.3 `h>0` 的正式定义

在 held-out folds 比较两个冻结校准器：

\[
M_0:Y\sim(m^0,c),\qquad
M_1:Y\sim(m^0,c,\hat h^a).
\]

定义 held-out clinical specificity information proxy：

\[
\widehat{\mathrm{CSI}}_a=
\mathrm{NLL}(M_0)-\mathrm{NLL}(M_1).
\]

只有同时满足以下三条，才判定存在可提取 \(h\)：

1. `CSI > 0`，image-bootstrap 95% CI 排除 0；
2. 相对 row/column-preserving shuffled interaction，conditional AUROC 至少高 0.05，CI 排除 0；
3. 在 baseline errors 上，\(y_{ic}\hat h_{ic}>0\) 的比例显著超过 matched null，且 matched
   correct cases 的翻转伤害不超过 1pp。

必须至少一个 intervention 在 **Huatuo 和 Hulu 两模型**均通过；否则 `h-existence` 判失败。

### 5.4 方法进入门

只有 h-existence 通过才运行：

\[
m'_{ic}=m^0_{ic}+\lambda\hat h_{ic}.
\]

Interaction-Gated Decoding 还必须：

- 比 raw intervention 和旧一维 NCD 至少高 1pp BAcc，95% CI 排除 0；
- 相对 native 同时报告 FP/FN，不能靠 polarity shift；
- reader-vote Brier 相对改善至少 5%；
- fixed-K listing 下 claim precision 提升且 recall 不下降。

## 6. 决策表

| Specificity gate | Decoding gate | 科学结论 | 决定 |
|---|---|---|---|
| FAIL | 不运行 | 所测 interventions 的响应可由边际工作点/噪声解释，没有可提取临床 interaction | 停止 NCD/ISD mitigation；保留严格负结果 |
| PASS | FAIL | 存在条件标签信息，但简单线性送回 decoder 不安全 | 做机制/测量论文，不包装缓解算法 |
| PASS | PASS | interaction specificity 预测并实现无 coverage 交换的错误修复 | 才进入 ICLR 方法主线 |

## 7. 当前审稿结论

### 7.1 最新碰撞风险

| 相邻工作 | 已覆盖 | 仍未覆盖的 delta |
|---|---|---|
| [System-Mediated Attention Imbalances, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1940/) | 干预会系统性改变 yes-rate，且效果依赖任务/架构 | 没有把每个 intervention 分解成 image、claim、patient×claim interaction，也没有条件信息上界 |
| [Looked but didn't see, 2026](https://pubmed.ncbi.nlm.nih.gov/42369483/) | “是否看见”必须配 matched false-alarm baseline | 负对照用于信号检测，不用于跨 claims 的 response decomposition 或解码 admission |
| [Evaluating Diagnostic Robustness Under Perturbations, 2026](https://arxiv.org/abs/2608.04885) | 医学 VLM 视觉/文本扰动与 lesion-removal negative control | 评估预测翻转和 overcommitment，没有识别 intervention response 中的 patient×claim interaction |
| [How Many Counterfactuals Does It Take?, 2026](https://arxiv.org/abs/2606.08777) | counterfactual causal influence 及采样复杂度 | 不控制 claim-wide/image-wide response，也不连接 conditional information 与 correction upper bound |

因此不能把“使用负对照”本身写成创新；当前未检索到的窄贡献是：**对 hallucination mitigation
response 做双向 interaction decomposition，并检验其 conditional information 是否预测或上界
实际可修复增益。** 该 delta 仍需实验成立，检索缺失也不等于证明首创。

### 7.2 判决

- **原 NCD-as-method：Reject and Pivot。** 不是因为 `−0.109` 与 `+2.92pp` 数学矛盾，而是
  它错误地把 gate-failed 中间层提示当成默认 evidence source，且一维 median DID 新颖性不足。
- **Intervention Specificity-as-problem：Accept with major revisions, pending h-existence gate。**
  新意不在 two-way demeaning，而在检验一个领域默认假设：现有 mitigation 的 response 是否真的
  含有 patient×claim conditional information，以及该信息是否给出可修复上限。
- 当前仍不是 ICLR-ready；320×8×2 之前不再设计新 decoder。

最可能达到 ICLR 的故事是：

> Hallucination mitigators are evaluated by how much they change an answer, but response decomposes into
> method-, image-, claim-, and patient×claim effects. Only the interaction term contains admissible clinical
> evidence, and its conditional information bounds the attainable correction gain.

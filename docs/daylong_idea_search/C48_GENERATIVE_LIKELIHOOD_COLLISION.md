# C48 — Generative Image Likelihood / One-Step Reconstruction：公式级碰撞审计

日期：2026-08-13  
目标：判断 diffusion likelihood、consistency/flow 一步近似、MAE reconstruction residual 能否构成一个新的、低时延、training-free 医学 VLM 幻觉缓解原语。  
结论：**严格 NO-GO；不下载生成模型、不运行 GPU。**

## 1. 候选原本想解决什么

开放式医学 VLM 可能生成一个图像并不支持的阳性 finding。候选想用冻结的图像生成/重建模型回答：

> “在存在 claim `c` 的条件下，当前图像是否比不存在 `c` 时更容易被生成或重建？”

若答案为是，则把它当作独立于语言 decoder 的病例级证据。这个信息来源方向是合理的；问题在于它是否形成了新的算法对象。

## 2. 不同名称下的共同数学对象

设 `x` 是当前医学图像，`c` 是一个 finding。任何条件生成模型最终都在比较

\[
B(x,c)=\log p(x\mid c)-\log p(x\mid \neg c).
\]

`B` 是正、负 claim 对当前图像的生成似然比。加入先验后，它就是 Bayes 分类器的 log-odds。以下几种实现只是在估计同一个 `B`：

| 表面实现 | 实际计算 | 与 `B` 的关系 |
|---|---|---|
| diffusion classifier | 比较正、负文本条件下的去噪误差 | 用 diffusion ELBO / score matching 近似条件 log-likelihood |
| 单时间步 diffusion | 只在一个或少量噪声时刻比较误差 | `B` 的低成本 Monte-Carlo 近似 |
| consistency model | 一步映射回干净图，再比较重建误差 | 用一步生成映射定义未归一化 conditional energy |
| flow / rectified flow | 比较条件流的重建误差或路径能量 | 仍是 conditional density / energy 比较；若精确算 likelihood 还需 Jacobian/divergence |
| conditional MAE | 比较 `c` 与 `not c` 条件下 masked reconstruction error | 用重建误差充当 `-log p(x|c)` |

Diffusion Classifier 的典型分数是

\[
\widehat L_c(x)
=-
\mathbb E_{t,\epsilon}
\left\|\epsilon-\epsilon_\theta(x_t,c)\right\|_2^2,
\qquad
\widehat B=\widehat L_c-\widehat L_{\neg c}.
\]

把多步 denoiser 换成一步 consistency/flow map `F_c`，得到

\[
\widehat B_{\text{one-step}}
=
\|x-F_{\neg c}(x_t)\|^2
-
\|x-F_c(x_t)\|^2.
\]

它改变的是 `B` 的数值估计器和计算量，而不是判断原则。因此如果贡献仅为“一步近似更快”，它是 **faster Diffusion Classifier**，不满足新算法原语要求。

## 3. MAE residual 也没有留下独特出口

无条件 MAE 只能给出

\[
A(x)=\|M\odot(x-\hat x)\|^2,
\]

它表示“图像看起来异常”，不能区分肺炎、积液、气胸等具体 finding。用它纠正 claim 会把所有异常共享为一个 common-mode signal。

一旦让 MAE 接收 claim 条件并计算

\[
A(x,\neg c)-A(x,c),
\]

它就重新变成上面的 conditional reconstruction classifier。Ano-swinMAE、MAEDiff、MAEDAY 等已覆盖医学异常的重建残差；把文本条件加入并不能产生与生成式分类不同的可辨识量。

## 4. 与已有工作的公式级关系

| 工作 | 已占据的核心 |
|---|---|
| Li et al., **Your Diffusion Model is Secretly a Zero-Shot Classifier**, ICCV 2023 | 直接用正、负文本条件的 diffusion 去噪误差近似图像条件似然并分类；相同随机噪声/时间步降方差 |
| **Classification Diffusion Models**, NeurIPS 2024 | 通过 noise-level classification / density ratio 在单次前向中得到生成式分类信号，直接覆盖“single-pass likelihood” |
| Hierarchical Prompting for Diffusion Classifiers, ACCV 2024 | 自适应类与时间步以加速 diffusion classification |
| **DEEM**, ICLR 2025 | 用 diffusion generative feedback 改善 VLM 视觉感知并降低 hallucination，覆盖“diffusion 作为 VLM 的第二只眼”叙事 |
| ESREAL, Cycle Consistency as Reward, CycleCap | 把回答/描述生成回图像，以图像重建一致性训练或纠正 hallucination |
| CIPHER | 用 diffusion 反事实编辑识别并投影 hallucination subspace；覆盖 diffusion counterfactual mitigation 邻域 |
| Ano-swinMAE, MAEDAY, MAEDiff | 使用 MAE/diffusion reconstruction residual 做医学异常检测 |

因此：

1. **多步 diffusion likelihood** 是已有 Diffusion Classifier；
2. **一步 consistency/flow** 是相同统计量的加速近似；
3. **无条件 MAE residual** 不具有 claim identity；
4. **有条件 MAE residual** 又退化为 generative classifier；
5. **回答到图像的 cycle reconstruction** 已被 ESREAL/CycleReward/CycleCap/DeGF 覆盖。

## 5. 本地证据并不支持继续付出 GPU 成本

本地 C30 先用冻结 BioMedCLIP 测试“独立视觉模型能否在 VLM final margin 之外提供病例级增量”：

| 模型 | final-margin macro-AUROC | + independent visual score | 增量 | image-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Huatuo | .7667 | .8084 | +.0417 | [.0239, .0596] |
| Hulu | .8606 | .8634 | +.0027 | [-.0028, .0084] |

Huatuo 有增量，但 Hulu 明确未通过跨模型门。更早的 Bayes reciprocal gradient 也被证明不能判断“当前图像是否支持 claim”：梯度只说明往哪个方向改图会增加 claim 概率；要恢复当前支持值必须做路径积分，最终仍回到 `B(x,c)`。

当前本地模型缓存没有可直接使用的 CXR text-conditioned diffusion、consistency、flow 或 conditional MAE checkpoint。按照预注册规则，在廉价跨模型增量门失败、且公式新颖性已经关闭后，不为它下载权重或占用 GPU。

## 6. Go / No-Go

| 要求 | 结果 |
|---|---|
| 引入病例级新信息 | 生成模型可能做到，但本地廉价代理只在一个较弱 VLM 上有增量 |
| 相对 Diffusion Classifier 的新数学对象 | **失败** |
| 一步近似是否改变可辨识量 | **否，只改变估计成本/偏差** |
| claim-specific 且非生成分类的 MAE 信号 | **不存在** |
| 跨模型 L0 门 | **失败** |
| 是否值得 GPU | **否** |

最终裁决：**C48 NO-GO。** 只有未来出现一个不是 conditional likelihood / reconstruction energy、并且能证明 `I(Y;u | s,c)>0` 的生成模型内部 invariant 时才允许重开；“更快的 diffusion classifier”不构成重开理由。

## 7. 参考文献

- Li et al. [Your Diffusion Model is Secretly a Zero-Shot Classifier](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.html), ICCV 2023.
- [Classification Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/13183a224208671a6fc33ba1aa661ec4-Abstract-Conference.html), NeurIPS 2024.
- [DEEM: Diffusion Models Serve as the Eyes of Large Language Models for Image Perception](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a8399aace3dfa6dfb8b635117748c561-Abstract-Conference.html), ICLR 2025.
- Rashmi et al. [Ano-swinMAE](https://proceedings.mlr.press/v250/rashmi24a.html), MIDL 2024.


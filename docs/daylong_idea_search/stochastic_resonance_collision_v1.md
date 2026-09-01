# Clinical Stochastic Resonance：碰撞、数学与致死实验审计 v1

更新时间：2026-08-12。

## 结论先行

**关闭为 ICLR 主方法，不占用当前 GPU。** 原始候选是：对同一医学图像逐渐增加噪声，
真实弱病灶会在某个中等噪声强度下“被抬过阈值”，而语言先验造成的假阳性不会；据此
利用整条 noise-response curve 同时恢复 FN、抑制 FP。

它有直观物理图景，但当前同时触发三条硬性淘汰：

1. **方法级直接碰撞。** VTI 已用多次噪声图像的视觉特征平均降低 VLM 幻觉；VAP 已把
   “有益视觉噪声”做成 training-free 幻觉缓解；最新 stochastic-resonance latent ensemble
   又直接做了 training-free、architecture-agnostic 的扰动—对齐—聚合。
2. **原始区分预测在最简单阈值模型中不成立。** 单个阈下真信号的检出概率随噪声单调
   上升，并非倒 U；出现倒 U 的是扣除了 false alarm 后的**群体判别力**。因此“真病灶
   有峰而 FP 没峰”不是 stochastic resonance 的数学推论。
3. **本地证据的先验方向不利。** 医学 VCD 明显降低 BAcc且 parse rate 下降；风格翻转
   很低；局部 multiscale scan 在强基线之上只有 `+0.0040` AUROC、CI 跨 0。它们不严格
   否定多强度噪声曲线，但没有提供值得优先消耗 GPU 的正现象。

这个候选最多保留为一个**诊断性负对照**：“噪声曲线测到的是局部决策边界曲率，还是
病例特异临床证据？”如果以后有空闲算力，可跑文末 64 例协议；即使通过，也只能恢复为
机制测量候选，不能直接恢复为新缓解算法。

## 1. 研究问题与最小数学背景

对图像 `x` 和临床 claim `c`，记

\[
m_c(x)=z_{\mathrm{present}}(x,c)-z_{\mathrm{absent}}(x,c)
\]

为模型的连续极性 margin。`m>0` 表示模型更倾向“存在”，`m<0` 表示更倾向“不存在”。
加入标准高斯噪声 `epsilon ~ N(0,I)` 后，观测

\[
\mu_c(\sigma)=\mathbb E_\epsilon[m_c(x+\sigma\epsilon)],\qquad
q_c(\sigma)=\Pr_\epsilon[m_c(x+\sigma\epsilon)>0].
\]

`mu` 是平均连续证据，`q` 是重复加噪时回答 Yes 的频率。原始设想希望真实弱病灶的
`mu` 或 `q` 在某个非零 `sigma` 达到峰值，而无病灶图像不会。

### 1.1 经典阈值模型真正预测什么

考虑最有利于 stochastic resonance 的玩具模型：无病灶时信号为 `0`，弱病灶时信号为
`a`，模型只有信号超过阈值 `theta` 才回答阳性，并且 `0<a<theta`。加入标准差为
`sigma` 的高斯噪声后：

\[
\operatorname{TPR}(\sigma)=Q\!\left(\frac{\theta-a}{\sigma}\right),\qquad
\operatorname{FPR}(\sigma)=Q\!\left(\frac{\theta}{\sigma}\right),
\]

其中 `Q(t)` 是标准正态变量超过 `t` 的概率。

关键点是：**TPR 和 FPR 都随噪声单调上升。** 非单调的是 Youden 判别力

\[
J(\sigma)=\operatorname{TPR}(\sigma)-\operatorname{FPR}(\sigma).
\]

它在 `sigma -> 0` 和 `sigma -> infinity` 时都趋近 0，并可能在下式给出的非零噪声处
达到最大值：

\[
\sigma_*^2=
\frac{a(2\theta-a)}{2\log\!\left(\theta/(\theta-a)\right)}.
\]

所以合格的 SR 检验必须证明**同一个预先冻结的噪声强度提高了正负样本的总体分离**，
而不能展示几张 FN 在某个随机种子下变成 TP。尤其不能先按 clean output 选 FN/FP：
`q(0)` 对 FN 已是 0、对 FP 已是 1，“FN 只能上升而 FP 不能再上升”会制造一个边界
导致的伪非对称。

### 1.2 噪声曲线通常测的是曲率，不自动是临床证据

定义高斯平滑后的 margin

\[
u_c(x,t)=\mathbb E[m_c(x+\sqrt{2t}\epsilon)].
\]

在通常的光滑条件下，它满足 heat equation

\[
\partial_t u_c(x,t)=\Delta_x u_c(x,t),
\]

即噪声强度刚开始增加时，margin 的变化由输入附近决策面的 Laplacian（局部曲率）决定。
Stein identity 还给出

\[
\nabla_x\mathbb E[m_c(x+\sigma\epsilon)]
=\frac{1}{\sigma}\mathbb E[\epsilon\,m_c(x+\sigma\epsilon)].
\]

因此多次加噪本质上也是一种无梯度的局部敏感度/梯度估计。一个峰可以来自病灶，也可以
来自压缩纹理、边缘、投影器 aliasing 或错误语言工作点；仅凭曲线形状不能给它“临床
证据”语义。

### 1.3 信息论边界

独立加噪形成 Markov 链 `Y -> X -> X+noise`。由 data-processing inequality，

\[
I(Y;X+\text{noise})\le I(Y;X).
\]

所以噪声不能创造原图中不存在的临床信息，也不能优于以原图为输入的 Bayes 最优诊断器。
它能改善一个固定 VLM，只可能因为模型的读出或阈值并不最优，噪声平均改变了其工作点或
平滑了错误决策边界。这种改善仍有实用价值，但不能叙述成“噪声放大了新的医学证据”。

更强的不识别边界也很直接：若一个真实弱病灶样本和一个视觉伪影样本在所有所测
`sigma` 下诱导完全相同的 margin 分布，则任何只读取 noise-response curve 的规则都对
二者给出同一答案，必然至少错一个。没有独立定位器、标签或新观测，曲线无法普遍区分
“被抬过阈值的真信号”和“被抬过阈值的伪信号”。

## 2. 文献碰撞

| 工作 | 已经覆盖的对象 | 与候选的关系 | 裁决 |
|---|---|---|---|
| [Stochastic Resonance, RMP 1998](https://doi.org/10.1103/RevModPhys.70.223) | 阈值/势垒、弱输入、噪声以及非零最优噪声 | 物理和数学母题 | 不能把 SR 本身当新理论 |
| [What Is Stochastic Resonance?, PLOS 2009](https://doi.org/10.1371/journal.pcbi.1000348) | 区分“noise benefit”与严格 SR，强调性能量必须随噪声在非零点改善 | 约束实验命名 | 只有单样本翻转不能称 SR |
| [Stochastic Resonance Improves Detection of Low Contrast Images, 2025](https://arxiv.org/abs/2502.14442) | ANN 中降低图像对比形成阈下信号，再用受控噪声恢复分类 | 几乎相同的弱视觉信号现象；仅模型/医学场景不同 | 现象级强碰撞 |
| [Randomized Smoothing, ICML 2019](https://proceedings.mlr.press/v97/cohen19c.html) | 对高斯扰动后的分类分布做 Monte Carlo 聚合并给出鲁棒性保证 | 多噪声视图平均的标准基线 | 聚合/平滑非新 |
| [VCD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_Decoding_CVPR_2024_paper.html) | 将高斯噪声退化图像作为负分支，与原图 logits 对比 | 相同图像干预，不同读出 | 必须直接比较 |
| [VTI, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b4008025c2182bfe16fcc8566ee14d64-Abstract-Conference.html) | 多次噪声/遮罩扰动后平均视觉特征；发现平均可减幻觉，而单次噪声往往加重幻觉；再学习稳定方向 | 与“多噪声采样—聚合—减幻觉”最直接碰撞 | 方法级直接碰撞 |
| [NoiseBoost, 2024](https://arxiv.org/abs/2405.20081) | 训练时向 projected visual tokens 加 Gaussian noise，以正则化视觉/语言依赖 | 不是 training-free，但“噪声改善幻觉”的论点已占 | 邻近碰撞 |
| [Poison as Cure / VAP, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/5f868c3e0050f13ec82d3694df531de1-Paper-Conference.pdf) | 在输入端优化有益视觉扰动，training-free；覆盖 8 个 VLM，并报告随噪声强度变化的动态 | “有益视觉噪声缓解幻觉”已成为正式方法 | 直接方法碰撞 |
| [Stochastic Resonance of Latent Ensembles, 2025 preprint / ICLR 2026 submission](https://arxiv.org/abs/2510.03224) | training-free 地对输入做微小变换、反向对齐 latent 后聚合；跨分类与 dense prediction | 同一跨领域概念、同一 test-time aggregation 范式 | 通用性主张被强占 |
| [SECOND, ICML 2025](https://proceedings.mlr.press/v267/park25c.html) | 逐级选择并整合多尺度局部 patch，再做 contrastive decoding | 若进一步限定“只在病灶处加噪”，会进入其局部选择—增强邻域 | 局部版本也非空白 |

未检索到完全等价的“用整条噪声强度曲线区分医学弱病灶与语言先验 FP”的论文；但这个
剩余差异是一个**医学诊断性测量**，不是足以越过 VTI/VAP/latent ensemble 的方法创新。

## 3. 与本地已完成结果的相容性

| 本地证据 | 数值 | 对本候选的影响 |
|---|---:|---|
| LLaVA-Med CXR-VisHal 的 VCD | BAcc `44.71%`，Yes rate `91.87%`，parse `79.10%` | 单一退化分支没有产生可靠医学分离，构成反证但不等于多强度曲线实验 |
| 风格/DG 统一确认 | 翻转率 `<5%`；style drift error AUROC `0.425`，clean margin `0.798` | 语义保持扰动的响应不比原 margin 更能识别错误 |
| 小病灶自然现象 | area–margin 在两模型确认相关 `0.323/0.415` | 支持“存在弱信号群体”，是本候选唯一真实动机 |
| 面积预测 miss | Huatuo `+0.083` AUROC，Hulu `+0.015` 且 CI 跨 0 | 弱病灶难度跨模型存在，但不足以成为通用错误探测器 |
| supervised multiscale patch scan | AUROC `0.7376 -> 0.7416`，`+0.0040`，CI `[-0.0194,0.0273]` | 局部视觉场没有在强聚合基线上显示增量，降低“局部噪声能恢复证据”的先验概率 |
| lesion delete–relocate | 删除效应 CI 跨 0；搬运较原图 overshoot `+0.185` | 图像响应容易被人工干预全局推动，不能把响应幅度当证据 |

这些结果不构成多噪声强度曲线的形式性证伪，但共同表明：若仅看到输出变化或某个噪声点
改善，最可能的简单解释仍是 decision-boundary/criterion shift，而不是临床信号共振。

## 4. 若将来要做：64 例不自欺致死实验

该实验仅在 GPU 空闲时作为诊断执行；当前不进入队列。

### 4.1 固定样本与模型

- 数据：VinDr-CXR，同一 `Nodule/Mass` finding，patient/image-disjoint。
- 64 张图，不按模型 clean 对错或 margin 选择：
  - 16 张 `3/3` 阳性、bbox area 最小四分位；
  - 16 张 `3/3` 阳性、bbox area 最大四分位；
  - 32 张 `0/3` 阴性。
- 首个模型：HuatuoGPT-Vision-7B；只有全部门通过才在 Hulu 复现。
- 问题模板、Yes/No token、图像预处理和 clean score 与现有 VinDr CE 完全一致。

### 4.2 干预

在 `[0,1]` 像素域加入零均值 Gaussian noise 并裁剪回合法范围：

\[
\sigma\in\{0,0.005,0.01,0.02,0.04,0.08\}.
\]

每个非零强度固定 8 个 seed；clean 只跑一次，共
`64 * (1 + 5*8) = 2,624` 次 claim forward。所有样本使用同一组 seed，保存完整
present/absent logits，不能只保存最终 Yes/No。噪声尺度不得根据 64 例结果改写。

若全图 pilot 通过，再增加两种等面积干预：仅 bbox 加噪、同尺寸随机健康区域加噪。
不能一开始加入这些分支，否则 64 例会同时承担尺度、位置和方法选择而失去确认意义。

### 4.3 统计量

对每个图像计算 `mu_i(sigma)`、`q_i(sigma)` 和 margin variance。主要群体统计量是

\[
J(\sigma)=\Pr(m>0\mid Y=1)-\Pr(m>0\mid Y=0),
\]

而不是 FN 的正向翻转率。另报告：

1. 以 8 个 noisy margins 的均值为 score 的 AUROC、BAcc、NLL；
2. `small positive`、`large positive` 的 TPR 与 0/3 阴性的 FPR；
3. interior-peak contrast：最佳中间强度相对 clean 和 `sigma=0.08` 的较小差值；
4. curve features 在 `clean margin + area` 之上的 image-bootstrap 增量 AUROC/NLL；
5. 答案解析率、Yes rate、图像 SSIM/LPIPS 和 cap-hit。

最佳强度的选择必须包含在 bootstrap/permutation 的最大值统计量中，不能先选最好
`sigma` 再当作固定假设做普通置信区间。

### 4.4 严格 Go/No-Go

64 例只允许触发扩量，不能形成论文结论。必须同时满足：

1. 存在非零 interior optimum：`J(sigma*)` 同时高于 clean 与 `sigma=0.08` 至少
   `0.04`，selection-aware 95% CI 下界 `>0`；
2. 在同一 `sigma*`，small-positive TPR 增加至少 `10pp`，而 0/3 FPR 增加不超过
   `2pp`；不能靠统一提高 Yes rate 获益；
3. 以 clean margin 和 bbox area 为强基线，curve features 额外增加 AUROC 至少 `0.02`
   且 NLL improvement CI 下界 `>0`；
4. large-positive/clear case BAcc 下降不超过 `1pp`，parse rate 不下降；
5. 结果不是 1–2 个 noise seed 驱动，leave-one-seed-out 方向一致。

通过后才扩至至少 256 张独立图、两模型、三个 findings，并加入 VCD、VTI-style feature
averaging、普通 randomized smoothing、VAP 和 matched-compute test-time augmentation。
若任一门失败，永久关闭，不改噪声分布、尺度网格或事后样本定义续命。

### 4.5 即使通过，能声称什么

可声称：在冻结医学 VLM 中，预先定义的弱病灶群体存在一个非零噪声下的判别力峰，并且
noise-response curve 对 clean margin 有增量预测价值。

仍不能声称：噪声创造了临床证据、普遍区分所有 hallucination、或提出了新的 stochastic
resonance 数学。要升级为方法，还必须证明曲线给出的病例选择能在 fixed operating point
下同时减少 FP 和 FN，并显著超过 VTI/VAP/随机平滑；当前没有这样的证据。

## 5. 最终科研裁决

| 维度 | 分数（0–3） | 依据 |
|---|---:|---|
| Importance | 2 | 小病灶漏诊与医学幻觉重要，且本地有跨模型 area–margin 现象 |
| Mechanistic value | 1 | 原始“真病灶有峰、FP 无峰”不由 SR 理论推出；曲率/工作点是同样简单的解释 |
| Novelty space | 0 | VTI、VAP、latent ensemble 与低对比 ANN-SR 已覆盖机制、干预或主张 |
| Executability | 3 | 2,624 次 forward 即可做 64 例 pilot，现有 VinDr、bbox 和 CE scorer 均可复用 |

**硬门 G2（无直接机制碰撞）失败，因此不按加权平均挽救。** 这条路线不是“实验还没跑
所以待定”，而是“作为新缓解方法已被文献碰撞关闭”；64 例实验只在需要建立医学噪声
敏感性边界时才值得运行。

对当前主线最有用的结论不是“noise can help”，而是：

> 独立噪声是对原图信息的 garbling；多噪声响应首先是模型决策面的光谱/曲率测量。
> 在没有独立定位或验证信号时，response curve 无法把弱真证据与易激发伪影区分开。

这与本项目已经反复确认的 `response is not evidence` 原则一致，但不构成一个新的 ICLR
Oral 级算法。

# 真实新观测能否取代同图反复解码：Active Sensing 文献碰撞与致死实验

日期：2026-08-12  
范围：2023--2026 active perception、value of information、sequential diagnosis、医学成像与 VLM；只使用论文原文、官方 proceedings/project page 或作者代码仓库。2026 年只有 arXiv 的工作均明确视为未审稿预印本。

## 结论先行

> **“第二个真实视角优于对同一张图反复解码”是一个重要、自然且可证伪的研究问题，但现在还不是一个可声称原创的算法。**

原因很清楚：

1. 同图 crop/zoom/search 已被 V*、ZoomEye、AdaptVision，尤其是 2026 年 BCEA 直接覆盖；
2. 通过移动相机获得真实新视角已被 embodied active perception 覆盖，Explore until Confident 已包含 conformal 停止；
3. 按信息增益选择真实医学测量，已是 active MRI 和 cost-effective diagnosis 的中心问题；
4. 多视角胸片融合本身也已有多年工作，CVPR 2025 MLRG 已同时使用当前多视角和纵向信息。

剩余的、可能有主会价值的缝隙不是“再做一个 zoom 方法”，而是：

> **把 VLM 幻觉拆成计算不足与观测不足，并在匹配代价下实证刻画二者的不可替代边界：什么时候应该继续想，什么时候必须再看一次。**

这条主张只有在“真实新观测稳定胜过所有匹配算力的同观测推理”，且跨医学、embodied/视频或 WSI 等至少两个领域复现后，才有 ICLR 主会机制论文的可能。当前没有正结果，也没有逐视角临床真值，因此不是 oral-ready idea。

---

## 1. 先冻结术语：三种“再看一次”不是同一件事

令病人真实状态为 \(Y\)，当前图像为 \(X\)。

| 操作 | 得到什么 | 是否增加关于病人的新信息 | 代表工作 |
|---|---|---:|---|
| 同图重算 | beam、self-consistency、更多 token、换 prompt | 否；输入仍是 \(X\) | 常规 test-time compute |
| 同图再编码 | crop、zoom、mask、图像金字塔 | 严格说没有新采集；可能让有限模型更容易读出 \(X\) 中已有细节 | V*、ZoomEye、AdaptVision、BCEA |
| 真实新观测 | lateral view、prior study、新 MRI 序列、新 WSI 区域/倍率、机器人换视点 | 可以；得到 \(A\)，信息集从 \(X\) 变为 \((X,A)\) | Explore-EQA、active MRI、AdaptivePath |

这一区分解释了本仓库已有负结果：风格变化、FedDG、同图内部层融合都只改变读取方式，没有增加真实临床证据；它们失败不能推出“新采集也无效”。反过来，第二张同 study 图有效，也不能证明某个 decoder 能凭单图创造该信息。

---

## 2. 文献碰撞：最接近的工作已经做到了哪里

### 2.1 固定图像内的主动搜索：高度拥挤，不能作为我们的新颖性

1. **V\***（CVPR 2024）让 MLLM 先提出搜索目标，再在高分辨率图像中做局部视觉搜索并维护 visual working memory；它已经把 hallucination 改善纳入评测。论文与[官方代码](https://github.com/penghao-wu/vstar)均公开。[论文](https://arxiv.org/abs/2312.14135)
2. **ZoomEye**（EMNLP 2025）用树形图像探索实现 training-free、model-agnostic zoom，并带停止规则；本质仍是从同一像素阵列选择更易读的区域。[论文](https://arxiv.org/abs/2411.16044)，[官方代码](https://github.com/om-ai-lab/ZoomEye)
3. **Visual Sketchpad**（NeurIPS 2024）允许模型调用检测、分割、绘图等视觉工具进行中间视觉推理，说明“视觉工具作为 scratchpad”也不是空白。[论文](https://arxiv.org/abs/2406.09403)，[官方代码](https://github.com/Yushi-Hu/VisualSketchpad)
4. **AdaptVision**（CVPR 2026）先读取低分辨率图，再用训练得到的策略决定是否请求 bbox crop，并把准确率与视觉 token 成本联合优化。[论文](https://arxiv.org/abs/2512.03794)，[官方代码](https://github.com/AdaptVision/AdaptVision)
5. **BCEA**（2026 arXiv，未确认正式会议）最接近本项目：它把 answer/abstain 扩为 answer/abstain/acquire，对边界 claim 自适应 crop/zoom/干预，并把完整采集策略重新纳入 conformal calibration，给出有限样本 selective hallucination-risk 控制；论文还证明 acquisition 是否提高固定风险下 coverage 与 ROC 是否改善直接相关，并讨论 Blackwell 信息序。[论文](https://arxiv.org/abs/2606.16667)

因此，以下表述均不可再主张为原创：training-free zoom、按不确定性决定是否 crop、预算化视觉重检、answer/abstain/acquire 三态、采集后的 conformal 校准，以及“更清晰局部图像可以降低 hallucination”。

### 2.2 真实新视角：机器人和 active vision 已占据基本范式

1. **Explore until Confident**（RSS 2024）让机器人移动以获得新 RGB/depth 视角，用语义地图与 VLM 引导探索，并用 conformal prediction 决定何时停止。这已经覆盖“真实新视角 + 顺序探索 + 校准停止”。[论文](https://arxiv.org/abs/2403.15941)，[官方代码](https://github.com/Stanford-ILIAD/explore-eqa)
2. **AVIS**（NeurIPS 2023）让 LLM 动态调用视觉工具、网页搜索和图像搜索以获取缺失证据；它拥有“自主视觉信息搜寻”的广义叙事，但主要增加外部知识，而非重新拍摄患者。[论文](https://arxiv.org/abs/2306.08129)，[官方代码](https://github.com/google-research/google-research/tree/master/avis)
3. **ActiveVision: An Exam for Active Observers**（2026 arXiv）开始专门评测模型能否重复观察并主动收集视觉信息，且报告前沿模型在主动观察能力上仍明显不足。[论文](https://arxiv.org/abs/2607.16165)
4. **Starve to Perceive**（2026 arXiv）甚至从训练角度限制一次性视觉带宽，迫使模型学会主动搜索；“让搜索成为必要能力”的训练叙事也已出现。[论文](https://arxiv.org/abs/2605.18603)

所以“把 VLM 变成主动观察者”本身不新。可剩下的区别只能是：**把同观测计算预算与真实观测预算放到同一条 hallucination risk frontier 上，并实证证明二者何时不可替代。**

### 2.3 医学里的新证据采集：信息价值与顺序决策也不是空白

1. **Active MRI Acquisition with Diffusion Guided Bayesian Experimental Design**（2025 arXiv）逐步选择真实 k-space 测量，以期望信息增益（EIG）最大化下一次采集的价值；这已直接覆盖“用信息论选择下一个医学测量”。[论文](https://arxiv.org/abs/2506.16237)
2. **TRUST-MRI**（2026 arXiv）用 transformer token uncertainty 选择 k-space 线，进一步占据“内部不确定性引导医学采集”。[论文](https://arxiv.org/abs/2603.21806)，[官方代码](https://github.com/levayz/TRUST-MRI)
3. **Deep Reinforcement Learning for Cost-Effective Medical Diagnosis**（ICLR 2023）把实验室检查选择写成带成本的顺序决策，并学习 cost--F1 Pareto frontier；“何时继续检查、何时诊断”是已建立问题。[论文](https://arxiv.org/abs/2302.10261)，[官方代码](https://github.com/Zheng321/Deep-Reinforcement-Learning-for-Cost-Effective-Medical-Diagnosis)
4. **AdaptivePath**（2026 arXiv）在 whole-slide pathology 中顺序选择新的位置与倍率，再由解释、审议和仲裁模块给出结论，是医学图像上的真实 active perception。[论文](https://arxiv.org/abs/2608.08648)
5. **ClinSeekAgent**（2026 arXiv）能从原始 EHR、网页及医学影像工具中动态搜集多模态证据；它说明“临床 agent 主动找证据”的广义故事已经出现，但其停止规则还不是严格 value-of-information 策略。[论文](https://arxiv.org/abs/2605.20176)，[官方代码](https://github.com/UCSC-VLAA/ClinSeekAgent)
6. **MedRAX**（ICML 2025）让胸片 agent 动态调用多种专用分析工具；它没有获得新的患者图像，但已占据医学工具协同这条工程路线。[论文](https://arxiv.org/abs/2502.02673)，[官方代码](https://github.com/bowang-lab/MedRAX)

### 2.4 多视角胸片：第二张图有用不是新发现

- **MLRG**（CVPR 2025）在胸片报告生成中同时融合 current multi-view spatial information 与 longitudinal temporal information。[官方 CVF 论文页](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Enhanced_Contrastive_Learning_with_Multi-view_Longitudinal_Data_for_Chest_X-ray_CVPR_2025_paper.html)
- **EVOKE** 把多视角对比学习与患者知识用于胸片报告生成。[论文](https://arxiv.org/abs/2411.10224)
- 更早的多视角胸片融合也已存在，例如 [Hashir et al., 2019](https://arxiv.org/abs/1907.09085)。

所以不能声称“首次证明两张胸片优于一张”。必须问更尖锐的问题：在**相同推理成本、相同输出长度、相同 claim 数**下，真实第二观测能否纠正同图所有解码方法都无法纠正的错误，以及能否提前预测何时值得采集。

---

## 3. 明确的不可申报新颖性

以下任一项单独拿出来都不足以构成 ICLR 贡献：

1. active perception / visual search；
2. 对同一图像 crop、zoom、mask 或图像金字塔；
3. 用 entropy、margin 或 uncertainty 决定是否再看；
4. 用 mutual information / expected information gain 选择下一个测量；
5. 带检查成本的 sequential diagnosis；
6. 多视角 CXR fusion；
7. Blackwell order、data-processing inequality 或“后处理不能创造信息”；
8. answer/abstain/acquire 与 acquisition-aware conformal calibration；
9. 临床 agent 调用检索、检测器或 EHR 工具；
10. “第二张图通常更准”这一平均性能结论。

---

## 4. 可保留的数学核心：计算误差与观测误差的正交分解

### 4.1 必要背景

考虑一个二元临床 claim，例如“存在胸腔积液”。真实标签是 \(Y\in\{0,1\}\)。当前图像是 \(X\)，额外真实观测是 \(A\)，例如同一患者的 lateral view。

- \(p_X=\mathbb E[Y\mid X]\)：如果只允许看 \(X\)，理论上最好的概率判断；
- \(p_{X,A}=\mathbb E[Y\mid X,A]\)：看到额外观测后，理论上最好的判断；
- \(q_C\)：现实 VLM 在只看 \(X\) 时，经过某种额外计算 \(C\) 后输出的概率。

用 Brier loss \((Y-q)^2\) 衡量概率预测误差。条件期望在平方损失下相当于把真值投影到“目前能看到的信息”上，因此有 Pythagorean 分解：

\[
\underbrace{\mathbb E[(Y-q_C)^2]-\mathbb E[(Y-p_{X,A})^2]}_{\text{现实模型距丰富观测最优解的总差距}}
=
\underbrace{\mathbb E[(q_C-p_X)^2]}_{\text{计算/解码不足}}
+
\underbrace{\mathbb E[(p_{X,A}-p_X)^2]}_{\text{观测不足}}.
\]

直观上：

- 同图 beam、CoT、self-consistency 可以减少左边第一项，即模型没有充分利用已看到的图；
- 真实新 view 才可能减少第二项，即原图本来就缺的信息；
- 两者不能用同一个“准确率提升”混在一起解释。

若使用 log loss，额外观测带来的理论收益恰好是条件互信息：

\[
I(Y;A\mid X),
\]

即“已经知道 \(X\) 后，\(A\) 还告诉了多少关于 \(Y\) 的新信息”。

**诚实边界：上述分解是条件期望/信息论的标准结论，不可作为原创定理。** 它的价值是为实验提供不可混淆的两个量。

### 4.2 真正值得冲主会的定理目标：固定覆盖下的观测预算下界

开放式医学回答可以看成从 \(K\) 个候选 claims 中输出固定 \(s\) 个，固定 \(s\) 是为了禁止“少说即少错”。令真实阳性集合 \(S\) 与输出集合 \(\hat S\) 都有 \(s\) 个元素；此时一次 false positive 必然伴随一次 omission。记 hallucinated claims 数为

\[
H=|\hat S\setminus S|.
\]

若 \(S\) 在所有大小为 \(s\) 的集合中均匀，且模型以概率至少 \(1-\varepsilon\) 达到 \(H\le h\)，一个 list-Fano 型下界要求：

\[
I(S;X)
\ge
(1-\varepsilon)
\log\frac{\binom Ks}
{\sum_{j=0}^{h}\binom sj\binom{K-s}{j}}
-h_2(\varepsilon).
\]

分母表示：对一个固定答案，最多有多少个真实集合与它相差不超过 \(h\) 个 claims。其含义是：候选 ontology 越大、要求列出的阳性越多，要同时压低 hallucination 和 omission 所需的视觉信息越多。只对同一图像增加计算不会增加 \(I(S;X)\)；真实新观测最多增加 \(I(S;A\mid X)\)。

这条基础形式仍源自经典 Fano/list-decoding，不能单独当 ICLR 新理论。**潜在主会贡献**必须进一步做到：

1. 扩展到非均匀、相关、带不确定态的临床 claim 集；
2. 给出可从 paired real acquisitions 估计的有限样本下界；
3. 证明并验证一个 phase law：在固定输出 claim 数下，ontology 扩张提高 hallucination floor，而真实观测的条件信息量使 phase boundary 可预测地右移；
4. 该规律跨 CXR、WSI/视频或 embodied VQA 成立，并能预测“继续算”和“再观察”谁更值。

如果做不到这些，数学只是漂亮的背景说明，不是论文贡献。

---

## 5. 一个可部署但尚未足够新颖的算法：Acquire-or-Compute

核心动作只有三个：`ANSWER`、`COMPUTE`、`ACQUIRE`。

1. 冻结 VLM，先用当前观测生成原子 claims 与概率；
2. 用少量校准集分别估计两条曲线：
   - 再增加一次同图推理的预期 proper-score 改善 \(G_C\)；
   - 获得一个真实新观测的预期改善 \(G_A\)；
3. 对每个可用动作按单位成本比较保守下界：

\[
a^*=\arg\max_{a\in\{C,A\}}
\frac{\operatorname{LCB}(G_a)}{\operatorname{cost}(a)};
\]

4. 若两个动作的保守收益都不为正，则回答；否则执行最值动作；
5. 新观测只能是临床上真正增加信息的 acquisition，例如 lateral view、prior、MRI sequence、新 WSI field 或机器人换位；同图 crop 必须单列为 re-encoding baseline，不能冒充新观测；
6. 最终固定 positive claim 数，或采用一换一 claim exchange，防止靠缩短报告获得幻觉下降。

该方法不训练或修改 VLM 权重，只做离线 calibration，具备通用、简洁和可部署性。但在当前文献下，`value/cost` 顺序选择本身不足以新；论文价值必须来自上一节的 observation--computation phase law，而不是这五行策略。

---

## 6. 本地可立即执行的致死实验

### 6.1 已准备 substrate

现有入口：[iuxray_observation_complementarity_v1.py](/home/dbw/ANCHOR/anchor/corrected_sgta/iuxray_observation_complementarity_v1.py)

- IU-Xray 本地有 2,955 studies、6,092 images，其中 2,790 studies 恰有两图；
- 已冻结 256 个 balanced、study-disjoint binary claims：128 Yes / 128 No，每个 study 只保留一个 claim；
- manifests：
  - pilot 64：`corrected_runs/daylong_idea_search_v1/iuxray_observation_pilot64_v1`；
  - full 256：`corrected_runs/daylong_idea_search_v1/iuxray_observation_v1`；
- analyzer 已实现 view0、view1、mean-margin fusion、max-abs、oracle、wrong-study view permutation 和 5,000 次 bootstrap；
- 本轮文献审计没有运行 GPU，也没有干预 baseline 队列。

最关键限制：标签来自同一 study 的共享报告，不是医生逐 view 标注的 `visible / refuted / unobservable`。因此即使通过，只能叫 study-level complementarity proxy，不能称为 view-specific hallucination 修复。

### 6.2 配对条件

同一批 study、同一问题、同一输出预算比较：

| 条件 | 目的 |
|---|---|
| A. view0 greedy | 单观测基线 |
| B. view0 上 beam / self-consistency / 已有最佳同图 decoding，匹配 FLOPs | 计算预算上限 |
| C. view0 + 真实同 study view1 | 观测预算 |
| D. view0 + wrong-study view1 | 排除“多一张图/多 token”的 placebo |
| E. 两视角 oracle | 判断是否真的存在互补 headroom，不作为方法成绩 |

### 6.3 冻结 fatal gate

第一轮只允许 Huatuo；任一核心门失败，关闭 active sensing 主线并恢复 baseline，不换融合权重救结果。

必须同时满足：

1. 真实第二 view 相对**最佳匹配 FLOPs 的同图推理**，BAcc 至少 `+3pp`，或 Brier 相对改善至少 `5%`，study-bootstrap 95% CI 排除 0；
2. 相对 wrong-study placebo 至少 `+2pp`，permutation `p<=0.05`；
3. two-view oracle 相对 view0 至少 `+5pp`，证明不是融合器无 headroom；
4. 增益主要发生在 view0 错误/低置信病例，且不是 Yes-rate、长度、parse rate 或阈值漂移；
5. Huatuo 全过后才用完全相同 manifest 在 Hulu 复现；第二模型也必须同向且 CI 排除 0。

### 6.4 即使 pilot 通过，正式论文还缺的致命真值

要把结果写成“降低医学 VLM hallucination”，必须获得独立逐视角 truth：

- 至少 100 个 paired studies；
- 至少 3 类 findings；
- 每类至少约 30 个 view-exclusive / view-unobservable cells；
- 标签区分 `supported / refuted / unobservable`，不能把“另一个 view 没框”当 negative；
- 标签不能来自共享 report、模型 classifier 或 LLM judge。

仓库现有 MIMIC 只有 44 个 paired studies；Tam boxes 的 354 studies 中没有成对 labelled studies；IU-Xray 只有共享 study report。**没有新逐视角真值时，本方向最多是 acquisition complementarity benchmark，不足以支持临床幻觉主张。**

---

## 7. Go / No-Go 与 ICLR 判断

| 结果 | 允许的结论 | 不允许的结论 | 决定 |
|---|---|---|---|
| 第二 view 不超匹配 compute 或不超 wrong-study | 当前多视角 substrate 没有可用 observation advantage | 不能说主动采集普遍无效 | 关闭该主线 |
| IU 通过、第二模型失败 | 模型特异 complementarity | 不能称通用规律 | 降级为诊断 |
| 两模型通过但无逐视角 truth | shared-report proxy 下新观测有互补 | 不能称 view-specific hallucination mitigation | 做数据/测量短文或继续取标注 |
| 两模型 + 独立逐视角 truth 通过 | 原观测不足确实造成一类不可由同图推理修复的错误 | 仍不能声称所有 hallucination 都如此 | 才允许扩到跨域 phase law |
| 跨至少两个领域复现 phase law，且 Acquire-or-Compute 优于 always-acquire 与 always-compute frontier | 有 ICLR 主会机制/决策论文潜力 | oral 仍取决于理论与规模 | 候选 GO |

### 最终评分

- 问题重要性：高；它改变了 mitigation 的动作空间，从“怎样解码”转为“什么时候需要新证据”。
- 当前方法新颖性：低到中；active perception、VOI、conformal acquisition、multi-view 均已有强近邻。
- 当前实证支撑：无，只有可运行 manifest；逐视角真值缺失。
- 数学潜力：中；标准信息论分解不新，固定覆盖 claim-set phase law 若能给有限样本新结论并跨域验证，才可能升格。
- 当前 ICLR oral readiness：**NO-GO**。

最诚实且最高效的选择是：先跑现有 256-study matched-compute fatal gate。通过后，把课题定义为 **Observation--Computation Frontier for Hallucination**；失败后永久剪枝，不再用 crop/zoom/第二 view 换名续命。

---

## 8. 核验过的主要参考文献与代码

1. Wu & Xie. [V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs](https://arxiv.org/abs/2312.14135). CVPR 2024. [Code](https://github.com/penghao-wu/vstar).
2. Shen et al. [ZoomEye: Enhancing Multimodal LLMs with Human-Like Zooming Capabilities through Tree-Based Image Exploration](https://arxiv.org/abs/2411.16044). EMNLP 2025. [Code](https://github.com/om-ai-lab/ZoomEye).
3. Hu et al. [Visual Sketchpad: Sketching as a Visual Chain of Thought for Multimodal Language Models](https://arxiv.org/abs/2406.09403). NeurIPS 2024. [Code](https://github.com/Yushi-Hu/VisualSketchpad).
4. Lin et al. [AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition](https://arxiv.org/abs/2512.03794). CVPR 2026. [Code](https://github.com/AdaptVision/AdaptVision).
5. Xu et al. [Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable Vision-Language Models](https://arxiv.org/abs/2606.16667). 2026 preprint.
6. Ren et al. [Explore until Confident: Efficient Exploration for Embodied Question Answering](https://arxiv.org/abs/2403.15941). RSS 2024. [Code](https://github.com/Stanford-ILIAD/explore-eqa).
7. Hu et al. [AVIS: Autonomous Visual Information Seeking with Large Language Model Agent](https://arxiv.org/abs/2306.08129). NeurIPS 2023. [Code](https://github.com/google-research/google-research/tree/master/avis).
8. Iollo et al. [Active MRI Acquisition with Diffusion Guided Bayesian Experimental Design](https://arxiv.org/abs/2506.16237). 2025 preprint.
9. Yu et al. [Deep Reinforcement Learning for Cost-Effective Medical Diagnosis](https://arxiv.org/abs/2302.10261). ICLR 2023. [Code](https://github.com/Zheng321/Deep-Reinforcement-Learning-for-Cost-Effective-Medical-Diagnosis).
10. Wu et al. [ClinSeekAgent: Automating Multimodal Evidence Seeking for Agentic Clinical Reasoning](https://arxiv.org/abs/2605.20176). 2026 preprint. [Code](https://github.com/UCSC-VLAA/ClinSeekAgent).
11. Fallahpour et al. [MedRAX: Medical Reasoning Agent for Chest X-ray](https://arxiv.org/abs/2502.02673). ICML 2025. [Code](https://github.com/bowang-lab/MedRAX).
12. Liu et al. [Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Enhanced_Contrastive_Learning_with_Multi-view_Longitudinal_Data_for_Chest_X-ray_CVPR_2025_paper.html). CVPR 2025.


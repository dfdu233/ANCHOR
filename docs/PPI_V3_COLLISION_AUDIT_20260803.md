# PPI v3 最新文献碰撞与新颖性上限审计

## 摘要

本审计回答三个问题：

1. PPI 的随机 provenance--claim 关联是否只是 Colored-MNIST、Waterbirds 和人工 shortcut 实验的医学复刻？
2. `互补 assignment + exact-parent matched children + reader-disagreement safety + natural medical validation` 的组合是否已被 VLM、backdoor 或医学 shortcut 文献实质覆盖？
3. 若要达到 ICLR 强接收乃至 oral 的研究站位，PPI 还必须提供什么机制证据？

结论是：**PPI v3 的因果设计明显强于普通 Colored-MNIST，但“模型会把与标签相关的空视觉线索学成预测线索”本身已经被 shortcut、fine-tuned VLM 与 backdoor 文献覆盖。** 最直接的碰撞是 RaVL（NeurIPS 2024）：它人为构造视觉特征—文本属性相关性，微调同一 VLM，并筛选确实学到关联的 children；TrojVLM（ECCV 2024）、VLOOD（ICLR 2025）与 Shadowcast（NeurIPS 2024）进一步证明视觉触发或少量污染可以控制生成式 VLM 的文本输出。医学侧，ShorT（Nature Communications 2023）已经通过受控改变年龄—疾病关联与 balanced 对照验证 CXR shortcut，RoentMod（npj Digital Medicine 2026）又用经医生验证的反事实 CXR 识别并缓解临床 shortcut。

因此，v3 目前最多支持一个严谨的 **medical generative shortcut model organism**。它尚不足以单独支撑“新机制”或 ICLR oral。可提升的核心不是再增加数据集，而是证明一个比 backdoor 更具体的机制命题：**医学适配没有简单地让 trigger 劫持输出，而是把可复用的来源坐标绑定成 claim-specific prior；该 prior 与视觉 likelihood 相加，在 reader-disagreement/弱证据附近跨越 commitment boundary，并且真实医学来源指纹通过同一隐变量—读出通路产生错误。**

## 1. 检索范围与纳入标准

检索截至 2026-08-03，分为四条证据链：

- 经典 randomized spurious feature、Colored-MNIST、background shortcut；
- VLM fine-tuning 中的 image-feature--text-attribute 关联、counterfactual augmentation；
- 生成式 VLM backdoor/data poisoning；
- 医学 acquisition/site shortcut、反事实医学影像及 reader disagreement。

只纳入官方会议论文页、PMLR、ACL Anthology、期刊正文或 arXiv 原稿。综述和二手网页只用于发现关键词，不用于核心判断。

## 2. 碰撞版图

### 2.1 经典 shortcut 文献已经覆盖“改变相关性便改变模型依赖”

IRM 用 Colored-MNIST 将颜色与标签的相关性设为随 environment 改变的人工变量，以检验 invariant predictor [1]；Sagawa 等进一步指出 majority/minority 比例和 spurious signal-to-noise ratio 决定 shortcut 学习强度 [2]。Noise or Signal 则通过 foreground/background 分离、随机或对抗背景替换表明背景本身足以驱动预测 [3]。这些工作已经建立了 PPI 的最低层事实：神经网络会利用与标签相关、但不属于目标语义的视觉坐标。

PPI 比这些工作的识别更干净：它保持完整样本、回答文本、训练顺序和 token mass 不变，只互换 shell assignment；`child-plus` 与 `child-minus` 提供反号预测，而不是只比较 biased 与 balanced dataset。但是这一优势主要是 **experimental identification strength**，不是新的学习现象。

### 2.2 RaVL 已经非常接近“VLM 在微调中获得视觉线索—文本属性绑定”

RaVL 人工规定 image feature 与 textual attribute 的 pair，在 MNIST/FashionMNIST 中使用 red rectangle，在 COCO 中使用真实 region attribute；随后采样 image-text fine-tuning data，使该视觉特征与 caption 中的类别属性达到指定相关强度，再从同一 pretrained CLIP 初始化微调多个随机种子 [4]。它共生成 620 个 fine-tuning datasets、1860 个候选 runs，并保留确实表现出目标关联的 654 个设置。它还在 PubMedCLIP/Object-CXR 上报告了 metal clips 与 cardiomegaly 的自然医学关联，但这部分是 off-the-shelf checkpoint 的相关性发现，不是受控医学 acquisition。

RaVL 与 PPI 的差异仍然重要：RaVL 没有互补反号 children，没有 parent-subtracted crossover，没有固定完整 example/text set，也没有生成式 claim commitment、reader-vote safety 或自然来源 fingerprint 的机制桥接。但审稿人完全可以据此指出：**“fine-tuning assigns textual semantics to a visually irrelevant cue” 已被 NeurIPS 2024 直接证明。**

ICML 2024 的 Amend to Alignment 同样把 VLM prompt tuning 的 OOD 失败归因于文本表示与 spurious image feature 的错误对齐 [5]；NeurIPS 2024 的 CounterAnimal 则显示 CLIP 会依赖更贴近其预训练数据的真实背景 shortcut [6]。因此，PPI 不能把“VLM 也会 shortcut”写成贡献。

### 2.3 Backdoor 文献构成比 Colored-MNIST 更危险的概念碰撞

TrojVLM 在 image captioning 和 VQA 中给图像加入 trigger，并训练 VLM 在保留原图语义的同时插入 predetermined target text [7]；VLOOD 进一步表明，即便没有原始训练数据，也能用 OOD samples 给生成式 VLM 注入 trigger-to-text 映射 [8]；Shadowcast 则只用少量 visually indistinguishable poison samples 操纵 VLM 的 label 与开放式 persuasion outputs [9]。RoCLIP 还把防御目标明确表述为打破 poisoned image-caption association [10]。

在没有机制结果时，苛刻审稿人会把 PPI 描述为：

> 一个非恶意、全样本、multi-label 的 visual backdoor，在 CXR padding 区放置 A/B trigger，并用训练分布使 trigger 与 clinical claims 相关。

“不改标签”“shell 临床为空”“A/B 频率平衡”能排除传统 label poisoning 和临床像素污染，却不能单独排除 backdoor-level novelty collision。PPI 必须证明 cue 的作用是 **evidence-conditioned prior**，而不是更隐蔽的 target trigger。

### 2.4 Counterfactual training 与医学 shortcut 也已覆盖大部分外围模块

COCO-Counterfactuals 用 minimal image-text changes 生成 paired counterfactual data，并将其用于 VLM training augmentation 以提高 OOD generalization [11]。医学中，ShorT 通过随机 subsampling 构造 biased 与 balanced CXR datasets，并以五个 replicates、不同 gradient scaling 干预 age encoding，证明受控增强 age--effusion 关联会增强 shortcut 与 fairness degradation，而平衡关联会消除这一模式 [12]。这已经覆盖“医学数据 + 受控关联 + balanced mitigation + representation intervention”的组合。

自然 acquisition shortcut 更早已被反复确认：Zech 等发现 CXR 模型能高精度识别 hospital system/department，并随机构疾病率调整 pneumonia prediction [13]；Compton 等在四个 CXR datasets、九个 labels 的组合中显示 disease--hospital 关联和 hospital-specific artifacts 可使 pooled training 的 worst-group performance 下降 [14]。Boland 等还用 Prediction Depth 和 KL divergence 定位人工医学 image shortcut 在哪些层出现 [15]。

2026 年的工作继续抬高门槛。RoentMod 生成经 radiologist review 的 pathology counterfactual CXR，证明 off-target pathology 可作为 shortcut，并用反事实训练缓解它 [16]；Vigneshwaran 等通过 causal generative model 从 activation 中 counterfactually remove site，量化多中心 Parkinson MRI classifier 的 site shortcut [17]。所以“反事实医学验证”“医生审查”“site/acquisition cue”均不能独立成为 PPI 的主新颖性。

### 2.5 Reader disagreement 是临床价值增量，但不是足够的机制增量

CheXthought 已利用大规模 multi-reader annotations 预测 human--human 与 human--AI disagreement，并将其用于不确定性沟通 [18]；Moll 等也以临床图像/文本 perturbations 和四位 radiologists 评估 medical VLM 的 causal attribution 与 confidence calibration [19]。PPI 将 reader-vote bins 作为 shortcut harm 的 effect modifier 是合理且有价值的，但仅仅分层报告 0/3、1/3、2/3、3/3 不足以构成机制贡献。

它真正能产生新知识的方式是检验：**provenance prior 是否在证据接近决策边界时导致过度承诺，而在清晰证据下被 likelihood 压过。** 这需要显式的 evidence-by-provenance interaction，而不是只报告四个 subgroup metrics。

## 3. 组合新颖性矩阵

| 设计维度 | 已有最强覆盖 | PPI v3 的剩余增量 |
|---|---|---|
| 人工 cue--label/attribute 关联 | Colored-MNIST；RaVL | 固定同一完整 image-text set，只互换 shell assignment |
| exact pretrained parent → children | RaVL、TrojVLM、VLOOD 通常从明确 base fine-tune | parent-subtracted matched triplets 和运行级 randomization inference |
| complementary sign reversal | 检索语料通常比较 correlated/balanced 或 poisoned/clean | `plus/minus` 的 claim-wise antisymmetric crossover 是最干净的设计增量 |
| 生成式 VLM 输出被视觉 cue 控制 | TrojVLM、VLOOD、Shadowcast | 不是固定 target string，而是多 claim 的概率性 clinical commitment |
| 医学受控 shortcut | ShorT；RoentMod | 医学 VLM 生成、完整 clinical pixels 不变、claim-level hallucination |
| 自然 site/acquisition validation | Zech；Compton；RaVL medical case | 将自然 source occurrence fingerprint 与受控机制相连接，而非只做相关性复现 |
| reader disagreement safety | CheXthought；medical VLM perturbation studies | 把 disagreement 作为 prior-vs-evidence 的效应修饰变量 |
| 内部机制 | ShorT 干预 attribute encoding；Boland 定位 shortcut layer；近期工作开始从 LoRA update 中识别 spurious factors [20] | 目前 v3 只有行为 crossover；内部 acquisition/binding/commitment 通路尚未建立 |

矩阵的判断是：**组合不是逐项重复，但目前的新颖性主要来自更严谨的组合与临床 outcome，而不是一个已经成立的新机制。**

## 4. 新颖性上限

### 4.1 按 v3 原样完成

- 能可信证明：在 exact-parent controlled continuation 中，临床为空的视觉 provenance coordinate 可获得 claim-specific predictive meaning，并在外部 CXR 上改变 claim commitment。
- 不能声称：首次证明 VLM 学习 spurious visual-text association；首次证明 trigger 控制生成式 VLM；首次证明医学模型使用 acquisition/site shortcut。
- 审稿定位：强 model-organism / causal audit；对 MIDL、ML4H 或 robustness track 有竞争力；对 ICLR main 属于 borderline-to-weak-accept 上限，**不具 oral 站位**。
- 主要拒稿理由：`medicalized RaVL/Colored-MNIST + benign backdoor`。

### 4.2 加入内部机制，但没有自然机制桥接

若能定位 source coordinate 在哪里可解码、在哪里与 claim 绑定，并通过 activation/weight intervention 消除 crossover，同时保持 clinical evidence，则可将论文从“现象复现”提升为“acquisition mechanism”。这可以达到 ICLR main 的合理强度，但仍可能被质疑 artificial shell 是否代表真实 provenance。

### 4.3 达到 oral ceiling 所需的命题

oral 级别不能是“医学 VLM 也会 shortcut”，而应是一个可跨领域表达、在医学中被精确验证的规律：

> **Domain adaptation converts a reusable provenance representation into a task prior by changing its downstream binding rather than its visual encoding; this prior causes errors primarily when task evidence is weak, and the same binding mechanism predicts natural source-conditioned errors.**

要支撑这句话，至少需要：第二个 exact-parent family、跨 shell family/真实 source 的机制复现、完整 causal mediation、reader-evidence interaction、自然 source subspace intervention，以及不靠缩短/拒答的 OE claim 结果。

## 5. 必须增加的机制证据

### 5.1 把“cue 被看见”与“cue 被绑定成 claim prior”分开

每个 layer/checkpoint 同时估计：

1. `availability A_l`：A/B 是否可从视觉或 multimodal representation 解码；
2. `binding B_{l,c}`：在控制 clinical evidence 后，source coordinate 对 claim logit 的方向性 coupling；
3. `commitment C_c`：该 coupling 是否穿过 absent/uncertain/present 或 definite boundary。

必要预测是：parent、plus、minus、balanced 对 A/B 的 availability 可以相近，但 `B_plus ≈ -B_minus`，`B_balanced ≈ 0`。如果 plus/minus 只是学出两个不同的 shell detectors，论文仍停留在 backdoor 描述。

### 5.2 做跨 child 的 causal path patching

在严格相同临床 tensor 下：

- patch A 与 B 的 shell-token representation，确定最早产生 `D_mic` 的层；
- 把 plus 的早层 A--B activation difference patch 到 minus；若输出符号仍由 minus 决定，说明视觉 source code 共享、反号来自 downstream binding；
- 在 binding 层只投影/消融 provenance-to-claim component，并恢复 norm；要求 crossover 消失，而 shell identity probe、clinical claim identity、clear 3/3 performance 保持；
- 用 random direction、same-norm direction、unrelated claim direction 和 sham shell 做控制。

这类结果才能把“相关 cue 被模型用了”提升为“适配重写了 source-to-claim readout”。

### 5.3 建立 prior 而非 trigger 的行为定律

显式拟合或检验：

`claim_log_odds = visual_evidence + provenance_offset + interaction`。

一个 prior-like mechanism 应表现为：

- A/B 主要移动 log-odds intercept，而不是改变 claim identity；
- 最大 threshold crossing 出现在独立视觉证据靠近边界的病例；
- 0/3 false positive 和 1/3、2/3 overcommitment 增加，但强 3/3 evidence 不被普遍反转；
- cue effect 随 admitted visual evidence 连续变化，而不是像 backdoor 一样无条件输出固定 target text。

必须预注册 reader vote 只是 evidence proxy，并增加独立 clarity/evidence measure。否则“在 disagreement cases 更敏感”也可能只是这些病例本身 logit margin 更小。

### 5.4 审计 weight-update mechanism

比较 matched LoRA updates：

- `ΔW_plus - ΔW_minus` 是否集中在少数层/低秩 subspace；
- 该 antisymmetric update 是否与 source activation direction 和 claim readout directions 的 outer-product 结构一致；
- 跨 seed 的 subspace/coupling 是否稳定，而 balanced 与 sham 不存在；
- 只训练 vision、projector、cross-attention/decoder LoRA 的 module-restriction 实验，定位最小充分模块。

近期工作已开始直接从 naive LoRA update 中识别并去除 fine-tuning 获得的 spurious factors [20]；PPI 若只做输出 crossover，会落后于这一机制门槛。

### 5.5 将 artificial organism 与 natural medical model 通过同一机制连接

自然验证不能止于 `natural source occurrence g` 与 output gap 相关。最低要求是：

1. 在真实 medical checkpoint 中解码 source/site/provenance coordinate；
2. 估计 source-coordinate--claim coupling matrix，并检验是否与 source-only occurrence fingerprint 对齐；
3. 对该 coordinate 做 causal erase/patch，要求 source-conditioned claim error 降低，而 pathology evidence 与 clear-case accuracy 保持；
4. 使用 independent source-held-out dataset 或第二模型复现；
5. 用 controlled children 验证同一分析 pipeline 确实能恢复已知 randomized `g`，再将其迁移到 natural checkpoint。

如果 controlled shell 和 natural source 只在行为上“都影响答案”，不能称为同一机制。

### 5.6 在 OE 中证明是 claim prior，不是长度或 positivity bias

固定输出 claim 数或 matched coverage，分别报告：

- 与 `g_c` 对齐的 claim substitutions；
- 不相关 claims 的 sham effect；
- polarity、certainty、location、severity 的独立变化；
- 报告长度、positive count、拒答、hedging。

真正的 claim-specific binding 应只移动与 randomized fingerprint 相符的 claims；若所有 positive claims 一起增加，它更像 global answer-style/positivity shift。

## 6. 决策建议

1. **保留 plus/minus/balanced 三臂和 exact-parent 设计。** 这是 v3 相对 RaVL、ShorT 和普通 backdoor 最扎实的识别优势。
2. **重写主问题。** 不再问“空 provenance cue 能否被学到”，而问“医学适配把既有 provenance representation 绑定为 evidence prior 的哪个环节，以及该 prior 如何与视觉证据竞争”。
3. **把 backdoor 文献放入主 related work，而不是回避。** 明确 PPI 不改标签、不是 attacker-chosen target、作用应受 evidence gating、并追求自然 source mechanism；然后用实验而非措辞证明这些差异。
4. **把 reader disagreement 从 outcome table 提升为机制交互。** 预注册 provenance effect 随独立 evidence margin 的函数，并把 reader votes 作为临床外部坐标。
5. **自然机制桥接是 go/no-go gate。** 若受控 shell 成功、真实 source subspace intervention 失败，则论文应降级为 model-organism shortcut study，不能包装成 medical provenance mechanism。
6. **oral 目标要求第二 exact-parent model organism。** 单一 Qwen2.5-VL child family 即使五个 seeds 成立，也难排除 architecture/LoRA-specific binding。

## References

[1] Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, David Lopez-Paz, “Invariant Risk Minimization,” arXiv:1907.02893, 2019.

[2] Shiori Sagawa, Aditi Raghunathan, Pang Wei Koh, Percy Liang, “An Investigation of Why Overparameterization Exacerbates Spurious Correlations,” ICML, 2020.

[3] Kai Xiao, Logan Engstrom, Andrew Ilyas, Aleksander Mądry, “Noise or Signal: The Role of Image Backgrounds in Object Recognition,” ICLR, 2021.

[4] Maya Varma, Jean-Benoit Delbrouck, Zhihong Chen, Akshay Chaudhari, Curtis Langlotz, “RaVL: Discovering and Mitigating Spurious Correlations in Fine-Tuned Vision-Language Models,” NeurIPS, 2024.

[5] Jie Zhang et al., “Amend to Alignment: Decoupled Prompt Tuning for Mitigating Spurious Correlation in Vision-Language Models,” ICML, 2024.

[6] Qizhou Wang et al., “A Sober Look at the Robustness of CLIPs to Spurious Features,” NeurIPS, 2024.

[7] Weimin Lyu et al., “TrojVLM: Backdoor Attack Against Vision Language Models,” ECCV, 2024.

[8] Weimin Lyu et al., “Backdooring Vision-Language Models with Out-of-Distribution Data,” ICLR, 2025.

[9] Yuancheng Xu et al., “Shadowcast: Stealthy Data Poisoning Attacks Against Vision-Language Models,” NeurIPS, 2024.

[10] Wenhan Yang, Jingdong Gao, Baharan Mirzasoleiman, “Robust Contrastive Language-Image Pretraining against Data Poisoning and Backdoor Attacks,” NeurIPS, 2023.

[11] Tiep Le, Vasudev Lal, Phillip Howard, “COCO-Counterfactuals: Automatically Constructed Counterfactual Examples for Image-Text Pairs,” NeurIPS Datasets and Benchmarks, 2023.

[12] Alexander Brown et al., “Detecting Shortcut Learning for Fair Medical AI Using Shortcut Testing,” Nature Communications, 2023.

[13] John R. Zech et al., “Variable Generalization Performance of a Deep Learning Model to Detect Pneumonia in Chest Radiographs,” PLOS Medicine, 2018.

[14] Rhys Compton, Lily Zhang, Aahlad Puli, Rajesh Ranganath, “When More Is Less: Incorporating Additional Datasets Can Hurt Performance by Introducing Spurious Correlations,” MLHC, 2023.

[15] Christopher Boland et al., “There Are No Shortcuts to Anywhere Worth Going: Identifying Shortcuts in Deep Learning Models for Medical Image Analysis,” MIDL, 2024.

[16] Joseph Paul Cohen et al., “RoentMod: A Synthetic Chest X-Ray Modification Model to Identify and Correct Image Interpretation Model Shortcuts,” npj Digital Medicine, 2026.

[17] Vibujithan Vigneshwaran et al., “Evaluating Shortcut Utilization in Deep Learning Disease Classification through Counterfactual Analysis,” MIDL, 2026.

[18] Sonali Sharma et al., “CheXthought: A Global Multimodal Dataset of Clinical Chain-of-Thought Reasoning and Visual Attention for Chest X-Ray Interpretation,” arXiv:2604.26288, 2026.

[19] Johannes Moll et al., “Evaluating Reasoning Faithfulness in Medical Vision-Language Models Using Multimodal Perturbations,” Machine Learning for Health, 2026.

[20] Ciarán M. Gilligan-Lee et al., “Unsupervised Identification and Removal of Spurious Correlations During Fine-Tuning,” arXiv:2605.27676, 2026.

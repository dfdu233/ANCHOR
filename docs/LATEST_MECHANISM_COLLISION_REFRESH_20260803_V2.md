# Latest mechanism-collision refresh V2: CECD 与 Reader-Grounded Two-Plane

**审计冻结日：** 2026-08-03  
**检索范围：** 截至冻结日可访问的顶会/顶刊论文与 arXiv；只采用论文主页、会议论文页和作者官方代码仓库。  
**审计模式：** outcome-blind。未读取任何本地模型输出、指标或封存结果；未调用模型、GPU 或实验脚本。  
**最终判定：** **KEEP CECD（仅保留严格条件化的行为机制）；KILL Reader-Grounded Two-Plane/RCCP 作为独立 headline 或方法贡献；NO-PIVOT。**

## 1. 冻结问题与判定口径

本轮只回答三个问题：

1. **RQ1 — collision：** 当前“极性 \(\pi\) 与证据清晰度 \(\kappa\) 被分别保留、但 \(\kappa\) 在生成时被抹除”的 two-plane 机制，是否已经被近邻工作覆盖？
2. **RQ2 — fatal alternatives：** 即使观测到 layerwise clarity gap，哪些更简单机制也会产生同样结果？
3. **RQ3 — alternatives：** 是否存在至多两个、比当前主线更高价值且可证伪的新机制方向？

这里将 collision 分为四层，避免把“用了相似技术”误写成“科学问题完全重复”：

| 级别 | 判定 |
|---|---|
| phenomenon collision | 相同可观察失败，如 late-layer override、视觉证据低依赖或过度确定 |
| mechanism collision | 相同中介解释，如 evidence/prior 分解、信息保留但利用失败 |
| intervention collision | 相同干预家族，如 layer-reference decoding、子空间投影、token-level grounding decoding |
| residual delta | 只有对方未覆盖、且能由独立真值与因果控制区分的最窄剩余命题 |

一个方向只有同时满足“重要现象、机制级非重复、独有可证伪预测、可执行因果端点”才可 `PIVOT`。仅换成医疗数据、reader votes 或新符号，不算机制新颖性。

## 2. 结论先行

### 2.1 Reader-Grounded Two-Plane：独立主线已被封闭

该构想的四个组成部分已分别出现直接碰撞：

- **视觉信息仍在、输出却不用：** [Seeing or Knowing?](https://arxiv.org/abs/2607.26326) 证明粗粒度视觉属性可从最终层 image tokens 重建，并将失败定位为 post-perceptual utilization/control，而不是视觉表征简单消失。
- **中层答对、后层被文本覆盖：** [MLLMs Get It Right, Then Get It Wrong](https://arxiv.org/abs/2606.17953) 明确定义 late-layer textual override，并以 CALRD 从中间层恢复被压制的视觉预测。
- **信心与视觉证据组合：** 医学 VQA 的 [CEBaG](https://arxiv.org/abs/2603.21693) 已把 token confidence variation 与 image-vs-text-only evidence magnitude 组成确定性 hallucination detector；[VES-RFT](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VES-RFT_Rewarding_Visual_Evidence_Sensitivity_to_Mitigate_Hallucinations_in_Large_CVPR_2026_paper.html) 则显式奖励图像引起的熵变化。
- **两个 uncertainty/source coordinates：** [VL-Calibration](https://aclanthology.org/2026.acl-long.2074/) 已解耦 visual confidence 与 reasoning confidence。因此，“两个正交平面”本身没有新颖性。
- **正交 evidence/prior/residual editing：** [HulluEdit](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_HulluEdit_Single-Pass_Evidence-Consistent_Subspace_Editing_for_Mitigating_Hallucinations_in_Large_CVPR_2026_paper.html) 已实现 evidence subspace、其正交 anti-prior subspace、residual uncertainty、自适应收缩与 norm restoration。RCCP 的 clarity-only 正交投影不能再作为方法贡献。
- **多读者分歧与不确定性表达：** [CheXthought](https://arxiv.org/abs/2604.26288) 已用大规模多读者数据预测 human-human/human-AI disagreement，并改善 uncertainty communication。

所以，two-plane 唯一尚未被直接覆盖的 delta 是：

> 在独立 reader-vote 真值上，控制 claim polarity、image-use 与 generic context sensitivity 后，reader clarity 是否在特定 support-to-language transition 上选择性下降；并且只去除该 clarity 分量是否能改变确定性，而不改变 claim identity、polarity、数量与覆盖率。

这可作为一个**嵌套的机制测量**，不能再作为论文 headline；只要其中任一限定不成立，整个 two-plane 解释即退化为已有的 late override、visual context sensitivity、confidence-evidence detection 或 subspace editing。

### 2.2 CECD：仍有一个窄而真实的剩余空间

本轮没有检索到完全等价于下述对象的工作：

> 对同一个 reader-grounded clinical claim，两个分别由医生确认不改变证据/命题/言语行为的操作——radiographic rendering 与 clinical wording——在完整 product orbit 上产生超出两个边际及通用 context-sensitivity 指标的 mixed derivative，并选择性预测临床错误。

因此：

- `KEEP` 的是 **clinician-admitted equivalence × equivalence product nonseparability**；
- `KILL` 的是“首次发现 prompt bias / late visual erasure / cross-modal conflict / activation steering”；
- 即便 CECD 成立，steering 也只能作 causal probe，不能作算法 novelty。

### 2.3 不应仓促寻找替代主线

本轮没有第二个候选同时通过 novelty、truth、causality 和 endpoint 四个门。最接近的 reader-threshold aliasing 值得作为替代解释控制，但不能解释 0/3 fabrication 或 3/3 omission，也没有干净的因果操作。因此最终是 `NO-PIVOT`，而不是用一个新名字替换一个已碰撞的 decoder。

## 3. 原始论文与官方代码核验矩阵

### 3.1 最危险的直接碰撞

| 工作（作者；日期/venue） | 原始来源 / 官方代码 | 核心对象 | collision 与剩余 delta |
|---|---|---|---|
| **Seeing or Knowing? Visual Context Sensitivity in Multimodal Large Language Models** — Jiaang Li, Chengzu Li, Zhaochong An, Yifei Yuan, Xi Liu, Serge Belongie, Vésteinn Snæbjarnarson；2026-07-28，arXiv | [paper](https://arxiv.org/abs/2607.26326)；冻结日未在论文页检出官方代码 | 最终层仍含可重建的粗视觉证据；failure 是 architecture-specific context utilization/control；activation patching 与 learned steering vector | **严重机制碰撞。** 直接否定通用“早层有证据、后层表征消失”叙事。剩余只可能是 reader-clarity-specific decline，而非视觉信息一般消失 |
| **MLLMs Get It Right, Then Get It Wrong: Tracing and Correcting Late-Layer Textual Bias** — Xingming Li, Ao Cheng, Qiyao Sun, Xixiang He, Xuanyu Ji, Runke Huang, Qingyong Hu；2026-06-16，IJCAI 2026 | [paper](https://arxiv.org/abs/2606.17953)；冻结日未检出官方代码 | 中间层视觉预测正确，后层向文本偏置移动；CALRD 以 layer-reference decoding 恢复 | **直接 phenomenon/mechanism/intervention collision。** two-plane 必须证明是 polarity-conditioned reader clarity，而非普通 textual override |
| **HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs** — Yangguang Lin, Quan Fang, Yufei Li, Jiachen Sun, Junyu Gao, Jitao Sang；CVPR 2026 | [paper](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_HulluEdit_Single-Pass_Evidence-Consistent_Subspace_Editing_for_Mitigating_Hallucinations_in_Large_CVPR_2026_paper.html)；[code](https://github.com/VioAgnes/HulluEdit) | sample-adaptive evidence/prior/residual decomposition；正交 anti-prior；自适应 contraction；norm restore | **RCCP 方法新颖性死亡。** reader votes 只能改变监督对象，不能使正交投影重新变新 |
| **VES-RFT: Rewarding Visual Evidence Sensitivity to Mitigate Hallucinations in Large VLMs** — Xuehe Hou, Wenshuo Li, Yali Li, Han Shu, Yuan Wang, Xinghao Chen, Shengjin Wang；CVPR 2026 | [paper](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VES-RFT_Rewarding_Visual_Evidence_Sensitivity_to_Mitigate_Hallucinations_in_Large_CVPR_2026_paper.html)；冻结日未检出官方代码 | 原图相对 no-image 的 image-attributable entropy change，作为视觉证据敏感性奖励 | **直接占据 evidence→uncertainty 关系。** reader truth、signed polarity 与 null-OOD 控制是剩余，不是通用思想 |
| **Deterministic Hallucination Detection in Medical VQA via Confidence-Evidence Bayesian Gain** — Mohammad Asadi, Tahoura Nedaee, Jack W. O’Sullivan, Euan Ashley, Ehsan Adeli；2026-03-23/v2 2026-07-17，arXiv | [paper](https://arxiv.org/abs/2603.21693)；[code](https://github.com/masadi-99/CEBaG) | 两次 teacher-forced pass；token confidence variation + image-vs-text-only evidence magnitude；医学 VQA CE/OE | **直接医学 collision。** 一般 `Support–Commitment Gap` 已被占据；剩余须是 signed reader distribution、layerwise conditional clarity 与 identity-preserving causal change |
| **VL-Calibration: Decoupled Confidence Calibration for LVLM Reasoning** — Wenyi Xiao, Xinchi Xu, Leilei Gan；ACL 2026 | [paper](https://aclanthology.org/2026.acl-long.2074/)；[code](https://github.com/Mr-Loevan/VL-Calibration) | 解耦 visual confidence 与 reasoning confidence；视觉 certainty 使用 perturbation KL 与 token entropy | **two-coordinate novelty collision。** 其坐标不同于 polarity/reader clarity，但足以杀死“首次 two-plane”主张 |
| **CheXthought: A global multimodal dataset of clinical chain-of-thought reasoning and visual attention for chest X-ray interpretation** — Sonali Sharma, Jin Long, George Shih, Sarah Eid, Christian Bluethgen, Francine L. Jacobson, Emily B. Tsai, Global Radiology Consortium, Ahmed M. Alaa, Curtis P. Langlotz；2026-04-29/v2 2026-04-30，arXiv | [paper](https://arxiv.org/abs/2604.26288) | 501 radiologists、50,312 multi-read CXRs；human-human/human-AI disagreement；uncertainty communication | **reader-grounded clarity 现象直接相邻。** 剩余仅是 polarity-conditioned layerwise erasure 与严格因果 transition |

### 3.2 医学检测、OE 与 image-use 的必要近邻

| 工作（作者；日期/venue） | 原始来源 / 官方代码 | 它排除的宽泛主张 | 本地协议必须新增的控制 |
|---|---|---|---|
| **Vision-language models for chest radiography do not always need the image** — Mahshad Lotfinia, Sebastian Ziegelmayer, Lisa Adams, Daniel Truhn, Andreas Maier, Soroosh Tayebi Arasteh；2026-06-16/v2 2026-06-19，arXiv | [paper](https://arxiv.org/abs/2606.17710) | 相关遮挡、无关遮挡、same-label image swap 揭示部分系统低图像依赖；confidence 不能自动当 evidence | 在拟合 \(\pi,\kappa\) 前先做 causal image-use/directional admission；任意 null 不能作唯一主证据 |
| **Visual Intervention-based Hallucination Detection for Medical VQA (VIHD)** — Jiayi Chen, Benteng Ma, Zehui Liao, Winston Chong, Yasmeen George, Jianfei Cai；MICCAI 2026 early accept | [paper](https://arxiv.org/abs/2605.20772)；[repo](https://github.com/Jiayi-Chen-AU/VIHD)（冻结日 README-only，声明代码稍后发布） | visually dominant layer selection + targeted visual-token masking + calibrated semantic entropy | layer selection、visual intervention、semantic entropy 都不是 novelty；正式比较须等可执行 release 或透明重实现 |
| **VGS-Decoding: Visual Grounding Score Guided Decoding for Hallucination Mitigation in Medical VLMs** — Govinda Kolli, Adinath Madhavrao Dukre, Behzad Bozorgtabar, Dwarikanath Mahapatra, Imran Razzak；2026-03-19，arXiv | [paper](https://arxiv.org/abs/2603.20314)；论文称 acceptance 后发布代码 | 原图与 distorted-image distribution 的 per-token visual dependency，medical OE adaptive decoding | per-token evidence decoding 已占；RCCP 若保留只能比较 reader calibration、polarity conservation 与 fixed-K OE |
| **Counterfactual Visual Grounding Uncertainty for Medical Hallucination Detection** — Xiao Song et al.；2026-06-26，arXiv | [paper](https://arxiv.org/abs/2606.28520)；[code](https://github.com/Agentic-CliniAI/CounterVHD) | 将任意医学输出抽成实体，以 supporting/counterfactual grounding 和置信度检测 entity-level hallucination | OE atomic claim grounding 不是新颖性；外部 grounding verifier 不能替代独立 reader truth |
| **A Benchmark for Hallucination Detection in VLMs for GI Endoscopy** — Aminu Lawal, Niyoj Oli, Sachin Acharya, Prashnna Gyawali, Maria Carmen Romano, Binod Bhattarai；2026-06-23，MIUA 2026 | [paper](https://arxiv.org/abs/2606.24115) | confident confabulation：采样一致性与 token probability 可在 hallucination 上保持高值 | entropy/self-confidence 只能是 readout/control，不能定义 visual evidence 或 truth |
| **HalluCXR** — Haoyu Wang, Zitong Li；2026-05-19，arXiv | [paper](https://arxiv.org/abs/2605.20469) | 长回答与 fabrication 相关；ensemble 减少 fabrication 可能增加 omission | fixed positive-claim count \(K\)、matched length/coverage、omission 与 refusal 必须共同报告 |

### 3.3 layerwise probing、causal heads 与 prompt presupposition

| 工作（作者；venue） | 原始来源 | collision 意义 |
|---|---|---|
| **HALP: Detecting Hallucinations in Vision-Language Models without Generating a Single Token** — Sai Akhil Kogilathota, Sripadha Vallabha E G, Luzhe Sun, Jiawei Zhou；EACL 2026 | [paper](https://aclanthology.org/2026.eacl-long.287/) | pre-generation probe 已覆盖 visual-only、decoder vision-token、query-token states；最佳层随架构变化。固定“早层最好”的统一叙事不可接受 |
| **VIB-Probe** — Feiran Zhang, Yixin Wu, Zhenghua Wang, Xiaohua Wang, Changze Lv, Xuanjing Huang, Xiaoqing Zheng；ACL 2026 | [paper](https://aclanthology.org/2026.acl-long.1078/)；论文页仅承诺将发布代码 | layer/head probe、VIB nuisance filtering、gradient-guided causal head intervention 已占据；普通 probe+head ablation 不构成新贡献 |
| **Mechanisms of Prompt-Induced Hallucination in Vision–Language Models** — William Rudman, Michal Golovanevsky, Dana Arad, Yonatan Belinkov, Carsten Eickhoff, Ritambhara Singh, Kyle Mahowald；ACL 2026 | [paper](https://aclanthology.org/2026.acl-long.1941/) | false-presupposition prompt 与 model-specific copying heads 已被机制化。Clinical Presupposition Amplification 只有在 proposition/speech act 严格等价、matched length 后才可能剩余；否则直接碰撞 |
| **Perceptual Hallucination in Vision-Language Models** — Taewook Hwang et al.；Findings ACL 2026 | [paper](https://aclanthology.org/2026.findings-acl.1237/) | damaged-image pair 与 activation patching 支持 perception error propagation/amplification；它是 clarity-erasure 的竞争解释，而非次要 baseline |
| **To Agree or To Be Right? The Grounding-Sycophancy Tradeoff in Medical VLMs** — OFM Riaz Rahman Aranya, Kevin Desai；2026-03-23，arXiv | [paper](https://arxiv.org/abs/2603.22623) | 医学 grounding 与 prompt agreement/sycophancy 已形成联合问题；宽泛“医学 presupposition 导致 hallucination”不够新 |

## 4. 官方代码只读审计

本轮只为了核验“方法是否真的这样实现”检查官方仓库，没有运行训练或推理。

| 仓库 | 冻结 commit / 日期 | 核验到的实现事实 | 可复现性限制 |
|---|---|---|---|
| [HulluEdit](https://github.com/VioAgnes/HulluEdit) | `ac0beeba40021578d8fa01543024338ba8138c3e` / 2026-06-14 | `hulluedit/steer.py`：weighted-SVD evidence basis \(U\)；anti-prior basis 投影到 \(U^\perp\)；evidence/prior/residual 分解；adaptive contraction；activation norm restore；fixed/random/gating controls | README 声称 MIT，但审计 checkout 未见独立 `LICENSE` 文件；引用和复现时需注明 |
| [CEBaG](https://github.com/masadi-99/CEBaG) | `736ee6977d302cb0b4d6fb57b2e332fc7e5e4f7b` / 2026-06-26 | image-conditioned 与 text-only teacher-forced passes；token log-prob std；normalized absolute Bayesian gain；MedGemma/LLaVA-Med/Huatuo，含 OE 模式 | ground truth 路径依赖 GREEN score；text-only pass 可能是 OOD，不等同 reader evidence |
| [VL-Calibration](https://github.com/Mr-Loevan/VL-Calibration) | `d38e15869d3a95d5d0c8fa1627d9745e490685d6` | 视觉置信与推理置信分离，perturbation-KL 与 token entropy 进入训练/校准路径 | Apache-2.0；属于训练式近邻而非直接 reader-vote 方法 |
| [CounterVHD](https://github.com/Agentic-CliniAI/CounterVHD) | `c03d17242873d714c523dd05a724f8c29a94bf92` / 2026-06-19 | 实体抽取、supporting/contradictory conversation、counterfactual phrase generation、bbox grounding、logit/sample confidence 与组合 uncertainty 均有代码 | entity extraction/counterfactual generation 可依赖 OpenAI-compatible API；其 verifier 不是独立临床真值 |
| [VIHD](https://github.com/Jiayi-Chen-AU/VIHD) | 冻结日网页只读核验 | 仓库只有 README 层信息，声明 code will be released soon | 不能把摘要声称当作 code-verified baseline，也不能宣称已复现 |

对 Seeing or Knowing、CALRD、VES-RFT、VGS-Decoding 的定向检索未在冻结日找到可归因的作者官方代码链接。因此报告只引用论文层结论，不声称实现已核验。

## 5. Two-Plane 的致命替代解释与一锤定音控制

### A1. generic visual context utilization，而非 clarity erasure

**同样可产生的结果：** 中层 reader-disagreement probe 高、末层低；steering 后更谨慎。  
**更简单解释：** 模型一直保留图像表征，但在回答时不稳定地选择视觉或语言 prior；probe 下降只是 readout basis 改变。  
**决定性控制：** 同层联合拟合 `polarity + reader clarity + WhatIfVis/CALRD-style context-sensitivity score + final-layer reconstruction/causal image gain`。只有 reader clarity 在这些变量之外仍有 held-out 增量，并由 cross-layer transport/causal patching显示真实损失，才支持 erasure。

### A2. perception-limited 或 image-ignoring model

**同样可产生的结果：** 低 reader clarity、过度确定、null 差异小。  
**更简单解释：** 模型从未获得正确方向性的视觉证据，decoder 无信息可“抹除”。  
**决定性控制：** 每个模型/claim 先通过 directional admission：score 随 0/3→3/3 正确移动，且大于 same-support image-swap drift；再做 relevant/irrelevant occlusion 与 same-label swap。未通过的模型标记 `perception-limited`，禁止进入 RCCP。

### A3. textual override 只影响 polarity，不是 clarity

**同样可产生的结果：** 后层 definite-positive 增多、\(\kappa\) readout 下降。  
**更简单解释：** late-layer bias 改变 present/absent margin，表观 certainty 只是 margin 的单调函数。  
**决定性控制：** clarity probe 必须在固定 finding、vote-count stratum、polarity score 和 margin 后仍增加 held-out AUROC/NLL；干预必须改变 certainty calibration，而 claim polarity、identity、positive count 和 location 保持不变。CALRD 若等效，即 two-plane 失败。

### A4. null/perturbation OOD

**同样可产生的结果：** 原图相对 mean-token-null/distorted image 有大 evidence gain。  
**更简单解释：** null 触发不同 token 分布或异常 attention dynamics，而非删除了可见证据。  
**决定性控制：** 主证据采用 in-distribution matched controls：same-support image swap、irrelevant-region occlusion、relevant-region occlusion、norm/position-matched image-token replacement。null 只作 sensitivity analysis，不能定义 \(S\)。

### A5. confidence/entropy 不是 evidence clarity

**同样可产生的结果：** disagreement case 熵更高，hedging 后 Brier 改善。  
**更简单解释：** token entropy 反映措辞多样性、回答长度或生成不稳定，而 hallucination 本身仍可高度自信。  
**决定性控制：** reader distribution 是唯一 clarity target；entropy、self-consistency、CEBaG、VES 和 GREEN 只是 baseline/readout。必须在相同 claim 文本 teacher forcing 下检验，避免 surface-form entropy。

### A6. OE brevity/omission exchange

**同样可产生的结果：** hallucination precision 上升、平均 claim 数下降。  
**更简单解释：** 方法少说、统一阴性、拒答或大量 hedge。  
**决定性控制：** 固定 positive claim 数 \(K\) 的 one-for-one exchange；matched claim coverage、长度、阴性率和 refusal；同时报告 0/3 inclusion、3/3 recall、location/attribute errors。任何 fabrication 降低伴随 omission 增加即失败。

### A7. probe geometry / layer reparameterization

**同样可产生的结果：** 非末层线性 AUROC 高于末层至少 0.05。  
**更简单解释：** 信息仍以非线性或旋转基底编码；linear probe 差异不等于信息被删除。  
**决定性控制：** 相同容量与正则的 linear/nonlinear probes；cross-layer probe transport；CCA/CKA 或 invertible alignment；label permutation、random direction、norm-matched controls；activation patching 同时做 noising/denoising 和 distance-matched control。若 nonlinear/readout alignment 恢复末层信息，只能称 `linear decodability change`。

## 6. 修改后的最小研究合同

如果仍将 two-plane 作为 CECD 的嵌套测量，必须一次性冻结以下五道 gate；不得在结果后改阈值：

1. **Admission gate：** reader-vote directionality 超过 same-support swap drift；relevant occlusion 显著强于 irrelevant occlusion。至少两个模型通过。
2. **Specific-information gate：** 在 polarity、margin、CEBaG、VES proxy、CALRD layer-shift、context sensitivity、finding/model/prompt 后，reader clarity 对 held-out reader unanimity 仍有至少 0.05 AUROC 增量，image-cluster bootstrap 95% CI 排除 0。
3. **Erasure gate：** 最佳非末层相对最终层差值至少 0.05，且 linear/nonlinear/aligned transport 一致支持“信息下降”，而非仅 readout rotation；最佳层允许随架构变化。
4. **Causal specificity gate：** clarity-only intervention 改变 certainty/reader Brier，但 clear-case accuracy 下降不超过 1pp，claim identity、polarity、positive count、location、length 与 norm 不变；优于 CALRD、HulluEdit、temperature scaling、random subspace 与 norm-only controls。
5. **OE gate：** fixed-\(K\) 后 fabricated positive 相对下降至少 20%，3/3 recall 不降，omission 不增；至少在 CE 与 OE/report 两类任务复现。

任一 gate 失败：`KILL two-plane mechanism`，不允许用更多阈值、更多 hedge 或更短答案挽救。

CECD 则采用不同的判定对象：它必须证明 clinician-admitted equivalence×equivalence 的 product mixed derivative 在 reader-grounded clinical error 上超出两个 marginals、generic consistency、context sensitivity、prompt-induced hallucination heads 和普通 layer override；否则同样 `KILL`。

## 7. 至多两个备选机制的筛选

### Candidate 1 — Reader-threshold aliasing / implicit virtual reader

**假设：** 所谓 1/3、2/3 上的过度承诺不是 clarity erasure，而是 VLM 采用了一个稳定的“虚拟读者”决策阈值；条件于 finding 与 vote count，回答取决于哪位 radiologist 赞成/反对，并跨 CE/OE 与模型保持排序。

**独有预测：** `positive_reader_pattern` 在 vote count、finding、clean margin、模型和 prompt 之外，对 held-out commitment/error 至少增加 0.05 AUROC 与 5% NLL；同一 reader alignment 在至少 6/8 findings、两模型、CE→OE 上方向一致。

**为何不 PIVOT：** 它不能解释 0/3 fabrication 与 3/3 omission；reader ID 只是匿名观测变量，liberal/conservative prevalence 可以完全解释 alignment；自然干预又落回 multi-annotator calibration。结论是 **保留为 alternative-explanation control，不是主线。**

### Candidate 2 — Claim-position grounding attrition / commitment hysteresis

**假设：** 在固定 \(K\) 的 OE 中，后续 claim 对图像干预越来越不敏感，先前文本承诺形成自回归锁定。

**为何表面诱人：** 可直接连接长答案、fabrication 与 omission tradeoff，并能通过 teacher-forced identical prefixes 检验 intervention timing。

**为何淘汰：** CEBaG/VGS 已覆盖 token-level evidence dependency，长回答与 omission/fabrication tradeoff 已有 HalluCXR，generic visual-prompt dilution 与 autoregressive confidence propagation 也是成熟解释。除非出现一个不由 token position、length、claim difficulty、prefix semantics、KV-cache 与 evidence score解释的医学特异因果 invariant，否则这是邻近机制重命名。**KILL，不进入实验主线。**

没有第二个候选通过四道硬门，因此“最多两个”在本轮实际只保留一个低成本 falsifier，零个 pivot。

## 8. Reviewer-style verdict

| 方向 | 重要性 | 机制新颖性 | 可证伪性 | 方法新颖性 | 判定 |
|---|---:|---:|---:|---:|---|
| Reader-Grounded Two-Plane / RCCP 独立主线 | 3/3 | 0.5/3 | 2/3 | 0/3 | **KILL as headline/method** |
| polarity-conditioned reader-clarity erasure，作为嵌套机制测量 | 2.5/3 | 1.5/3 | 2.5/3 | 0.5/3 | **KEEP only behind five hard gates** |
| CECD clinician-admitted product nonseparability | 3/3 | 2/3 | 2.5/3 | 0.5/3 | **CONDITIONAL KEEP** |
| Reader-threshold aliasing | 2/3 | 1.5/3 | 2/3 | 0.5/3 | **CONTROL ONLY** |
| Claim-position grounding attrition | 2/3 | 0.5/3 | 2/3 | 0.5/3 | **KILL** |

### 最终 KEEP / KILL / PIVOT

- **KEEP：** CECD 仅保留“医生独立承认的 equivalence×equivalence product mixed derivative”这一行为机制；reader clarity 只可作为其嵌套、预注册的替代解释检验。
- **KILL：** Reader-Grounded Two-Plane/RCCP 作为独立论文 headline、通用医学幻觉机制或新型投影算法；`Support–Commitment Gap` 的宽泛版本；通用 late-layer erasure、two-coordinate uncertainty 与 orthogonal activation editing 新颖性。
- **PIVOT：** **NO-PIVOT。** 当前没有备选机制在真实临床真值、机制新颖性、因果识别与不牺牲 omission 的 endpoint 上同时胜过窄化后的 CECD。

最可信、也最有研究审美的下一步不是继续发明 decoder 名称，而是进行一个结果无关的硬判决：

```text
clinician-admitted product-only defect exists?
        ├── no  -> credible negative result; stop CECD
        └── yes -> does it survive generic context-sensitivity / late-override controls?
                    ├── no  -> known utilization failure; no paper-level mechanism
                    └── yes -> is rescue joint-cell-specific and OE omission-neutral?
                                ├── no  -> mechanism observation only
                                └── yes -> ICLR-level conditional mechanism story
```

这条路径把新颖性押在一个此前未被严格定义的临床输入机制对象上，而不是押在已经拥挤的 calibration、steering 或 decoding 方法空间。

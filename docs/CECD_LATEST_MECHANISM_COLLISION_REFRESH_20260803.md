# CECD 2025--2026 最新机制级碰撞刷新

**审计日期：** 2026-08-03  
**范围：** 2025--2026 顶会、顶刊与最新 arXiv；只读文献和官方代码审计；不运行 GPU，不查看封存结果。  
**对象：** CECD（两个各自临床等价的图像显示与问题措辞操作，在完整乘积轨道上的不可加响应）及其候选隐藏态干预。

## 1. 结论先行

本轮新文献没有完全复现 CECD 的最窄科学对象，但显著压缩了它可以声称的新颖性。

### 1.1 唯一仍存活的主张

CECD 只应继续检验下述命题：

> Under two independently physician-admitted equivalence operators—one radiographic rendering and one clinical wording—the product-orbit mixed derivative of a fixed reader-grounded claim contains reproducible clinical-error information beyond either marginal, generic consistency metrics, medical prompt-by-modality-shift grids, Treble under both released semantics, and generic multimodal synergy.

中文即：**在医生分别确认“显示方式不改变可见证据、措辞不改变临床命题与言语行为”以后，图像等价变换与语言等价变换的乘积轨道仍产生不能由两个边际效应相加解释的 reader-grounded 临床错误。**

这仍是 `Conditional GO`，而不是已成立贡献。它的新意在**被严格 admission 的输入因子和临床误差对象**，不在四格差分公式，也不在 activation steering。

### 1.2 必须停止的主张

以下叙事已是严格 `NO-GO`：

- “首次发现医学 VLM 对 paraphrase 敏感”——[PSF-Med](https://arxiv.org/abs/2602.21428) 已有医生审计、层级 SAE feature 和因果 clamp。
- “首次研究图像与文本的四格联合变化”——[VB](https://arxiv.org/abs/2603.06680) 已有完整 `2 x 2` image-edit × text-edit 设计；[Medical Context Distorts Decisions](https://arxiv.org/abs/2605.17436) 的官方实现也已有医学 prompt-version × modality-shift 全网格。
- “首次发现后层视觉信号消失、语言先验增强”——[How Vision Becomes Language](https://arxiv.org/abs/2602.15580)、[CausalLens](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.html) 及多项既有工作已经占据。
- “首次发现/干预 cross-modal synergy 或交互 circuit”——PID flow 已量化 synergy；[Dual-Pathway Circuits](https://arxiv.org/abs/2605.13156) 已做 pathway-level interaction decomposition 和因果抑制。
- “正交化、norm restoration、最小投影或单层 steering 是方法创新”——[HulluEdit](https://arxiv.org/abs/2602.22727)、[ONLY](https://arxiv.org/abs/2507.00898)、[DMAS](https://arxiv.org/abs/2602.21704)、[CausalLens](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.html)、[CAI](https://arxiv.org/abs/2606.29847)、[CIPHER](https://arxiv.org/abs/2603.10470) 和 [Activation Steering Decoding](https://aclanthology.org/2025.acl-long.634/) 已把该方法空间占得很满。
- “医学报告中的语义解耦 latent steering”——[SDLS](https://arxiv.org/abs/2602.23676) 已在放射学报告中以 QR 正交化构造语义解耦方向。

因此，`Δh=h11-h10-h01+h00` 及其 additive-subspace projection 只能叫 **factorial interaction probe / interaction-component ablation**，不能作为独立算法贡献。

## 2. 本轮最危险的新碰撞

### 2.1 两轴稳定性已经被直接研究，但尚未形成乘积轨道

[Questioning the Stability of Visual Question Answering](https://arxiv.org/abs/2511.11206) 同时研究小幅 shift/pad/scale/rotation 等视觉扰动与 question rephrasing/multilingual rewrite，并测量两轴 entropy 及其相关/互信息。但其定义分别为 `S_v={(I',Q)}` 与 `S_t={(I,Q')}`；它没有对每个 `(I^a,Q^b)` 联合单元都前向，也没有计算同一命题的 centered mixed derivative。

碰撞结果：CECD 不能再把“两种等价变化共同影响稳定性”当新发现；只剩 **full product composition defect**。

### 2.2 医学 prompt × modality grid 已存在，但因子会改变证据

[Medical Context Distorts Decisions in Clinical VLMs](https://arxiv.org/abs/2605.17436) 在 MIMIC-CXR 上交叉多种 prompt style 与图像/文本 modality shift。其[官方仓库](https://github.com/dsrestrepo/context-distortion-vlms)的 `src/run_experiments.py` 确实循环完整 `prompt version × {None, Image, Text, Only_text, Only_image}` 网格。

代码审计同时发现：`Image` shift 换成相反标签影像，`Text` shift 换报告，因而是 evidence-changing conflict；release 中未检出 ANOVA、factorial interaction 或 mixed derivative 分析。CECD 不能声称“首个医学交叉网格”，但仍可声称 **equivalence × equivalence residual**，前提是医生 admission 先通过。

### 2.3 完整 2×2 已存在，但目标是 XOR 而非等价性

[VB](https://arxiv.org/abs/2603.06680) 构造 exact image-edit × text-edit 四格 family，并以 double flip 为诊断。区别是 VB 的单轴 edit 被设计为改变 gold，double edit 产生期望 XOR；CECD 的两个操作都应保持 gold/support，二阶项的科学零假设是零。

因此，“四格设计”没有新意；**临床等价下不应出现但实际出现的非可分性**才是剩余对象。

### 2.4 医学 paraphrase 机制与干预已被占据

[PSF-Med](https://arxiv.org/abs/2602.21428) 在六个医学 VLM 上研究 meaning-preserving paraphrase，使用医生审计，并在 MedGemma 定位 layer-17 SAE feature；其 delta-only causal patching 与 feature clamp 已给出干预证据。[官方数据页](https://huggingface.co/datasets/saillab/psf-med)当前可访问，但其 README 所列官方代码仓库 `UNHSAILLab/medical-vlm-robustness` 在 2026-08-03 返回 404。

CECD 必须证明 effect 不是 PSF-Med prompt feature 的简单外推：interaction intervention 若同样改善 prompt-only cell，CECD 的机制 specificity 即失败。

### 2.5 activation 方法空间进一步封闭

- [HulluEdit](https://arxiv.org/abs/2602.22727) 用 sample-adaptive weighted-SVD 视觉子空间与其正交 anti-prior 子空间做单次前向编辑，并显式宣称 evidence non-interference；[代码](https://github.com/VioAgnes/HulluEdit)已发布。
- [CausalLens](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.html) 将 heads 分解为 visual/text/system-prompt pathways，以 sensitivity 选择可靠视觉 heads，在中层进行 projection-aligned correction。
- [DMAS](https://arxiv.org/abs/2602.21704) 按语义检索 truthfulness vectors，并结合 visual-perception vectors 动态选择 heads；固定全局 steering direction 已不足以构成方法贡献。
- [ONLY](https://arxiv.org/abs/2507.00898) 以 text-to-visual entropy ratio 选单层/heads，单 query 干预；[代码](https://github.com/zifuwan/ONLY)已发布。
- [CAI](https://arxiv.org/abs/2606.29847) 按 token visual relevance 决定“看哪里”，再以 entropy/depth gate 决定“何时干预”，并给出 KL-minimal reweighting；[代码](https://github.com/Iris1946/CAI)已发布。
- [CIPHER](https://arxiv.org/abs/2603.10470) 从 diffusion-edited counterfactual 学低秩 hallucination subspace，再投影 hidden states。
- [SDLS](https://arxiv.org/abs/2602.23676) 已把 LLM semantic decomposition、QR orthogonalization 与 radiology report latent steering 结合用于 prior-comparison hallucination。

这组工作使“从 hidden state 投影某个不良方向”在 novelty 上完全不成立。CECD 若保留干预，其作用只能是**验证 CECD 这个输入级二阶因果对象**，而不是卖 steering 算法。

### 2.6 patching 的因果解释出现新的硬性威胁

[The Curse of Multiple Mediators](https://arxiv.org/abs/2606.27510) 重新推导 activation patching，证明常用 NIE 同时包含 pure indirect effect 与 mediator-bypass interaction；interaction 随 clean/patched activation distance 增大，且多组件时产生组合爆炸。论文列出的[官方代码链接](https://github.com/sankaranv/mech-interp-mediation-analysis)在 2026-08-03 暂时返回 404。

这不是普通 baseline，而是 CECD 机制语言的有效性约束。只做一次 `h <- h + αΔh` 并观察输出恢复，不能声称定位了纯 causal pathway。正式实验必须同时报告 noising、denoising、activation distance，并尽可能分解 PIE/INT；做不到时只能写“intervention-sensitive association”，不能写“该层中介了效应”。

[Dual-Pathway Circuits](https://arxiv.org/abs/2605.13156) 又表明 pathway interaction 本身已被用于五种 VLM 的 object hallucination circuit 分析，并报告 grounding/hallucination 双路径及 model-specific wiring。CECD 不能声称“首次 interaction-aware circuit”，只能问：**这个 circuit 是否只在 admitted-equivalence joint cell 中被招募。**

## 3. 机制级碰撞矩阵

| 工作 | 共同 phenomenon | 共同 mechanism | 共同 intervention | 精确剩余 delta |
|---|---|---|---|---|
| [MM-R3](https://aclanthology.org/2025.findings-acl.246/) | 图像 restyle 与 question rephrase 破坏一致性 | 主要是行为稳定性 | consistency mitigation | 轴分别处理；无完整 product orbit、reader truth 或二阶项 |
| [Test-Time Consistency](https://arxiv.org/abs/2506.22395) | 等价变体回答不一致 | test-time agreement | pseudo-label update | 无临床 admission；无 image×prompt mixed derivative |
| [Questioning VQA Stability](https://arxiv.org/abs/2511.11206) | 视觉与文本等价扰动双轴稳定性 | 比较两轴 entropy/MI | 无 CECD 式因果干预 | 只比较两个 marginal sets；没有四格联合单元 |
| [VB](https://arxiv.org/abs/2603.06680) | image×text exact 2×2 | double-flip diagnostic | 无 | edits 改变 gold；CECD 要求两个 nuisance 都保持 claim/support |
| [PSF-Med](https://arxiv.org/abs/2602.21428) | 医学等价 paraphrase flip | layer-17 SAE prompt feature | patch/clamp/normalization | 固定图像；无 render×paraphrase 二阶残差 |
| [Medical Context Distorts](https://arxiv.org/abs/2605.17436) | 医学 prompt style × modality shift | text/vision conflict | benchmark/audit | shift 换成相反证据；未估计 equivalence residual |
| [MedVIGIL](https://arxiv.org/abs/2605.07919) | 医学 text/image counterfactual | broken-evidence failure | benchmark | 未完整交叉两个等价操作；ROI corruption 改变证据 |
| [Words or Vision](https://arxiv.org/abs/2503.02199) | 文本压过视觉 | modality dominance | 主要是审计 | 文本添加/损坏语义；非等价 prompt×render |
| [Treble](https://aclanthology.org/2025.findings-emnlp.1000/) | visual/text/cross-modal causal effects | modality NDE/PCA directions | test-time activation intervention | 全局 perturbation direction；非 within-instance admitted product residual |
| [Prompt-Induced Hallucination](https://aclanthology.org/2026.acl-long.1941/) | prompt 驱动错误 | model-specific copying heads | head ablation | false-premise prompt 改 proposition；非 label-neutral 等价措辞 |
| [Med-CP](https://aclanthology.org/2026.eacl-industry.67/) | 医学 prompt override | cross-modal conflict | reflection/SFT | noisy clinical prompt 改信息；非等价语言操作 |
| [How Vision Becomes Language](https://arxiv.org/abs/2602.15580) | 后层视觉独有信息下降 | PID: vision/language unique、redundancy、synergy | attention knockout | generic multimodal flow；未绑定输入等价缺陷或 reader-grounded error |
| [Dual-Pathway Circuits](https://arxiv.org/abs/2605.13156) | grounding 与 hallucination 路径 | pathway-level interaction | targeted suppression | mediator interaction，不是输入-factor mixed derivative |
| [HulluEdit](https://arxiv.org/abs/2602.22727) | evidence 与 prior 竞争 | 正交隐空间分解 | sample-adaptive subspace edit | 占据投影方法；未研究 admitted product orbit |
| [CausalLens](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.html) | visual signal 后层衰减 | visual/text/system head pathways | sensitivity-guided mid-layer correction | 占据 pathway correction；无临床等价二阶对象 |
| [DMAS](https://arxiv.org/abs/2602.21704) | 语义相关 hallucination direction | truthfulness 与 visual heads 分离 | query-adaptive vectors | 占据动态 steering；无二阶等价 residual |
| [VLI](https://aclanthology.org/2026.acl-long.1784/) | 视觉—语言冲突 | counterfactual image causal directions | instance-specific bi-causal steering | anchor/conflict image 非等价 product orbit |
| [CIPHER](https://arxiv.org/abs/2603.10470) | counterfactual hallucination shift | low-rank subspace | hidden-state projection | global learned semantic counterfactual；非临床 nuisance interaction |
| [SDLS](https://arxiv.org/abs/2602.23676) | 医学报告 hallucination | 语义解耦 latent direction | QR-orthogonal steering | prior-comparison 特定轴；占据医学正交 steering，不占 behavioral CECD |
| [Curse of Multiple Mediators](https://arxiv.org/abs/2606.27510) | patching 结果依赖上下文 | PIE + mediator-bypass INT | 方法论诊断 | 不碰撞 behavioral CECD，但限制所有纯中介因果表述 |

## 4. 官方代码核验摘要

在 2026-08-03 对官方论文链接与仓库做了只读核验：

- [Context Distorts 官方仓库](https://github.com/dsrestrepo/context-distortion-vlms)：可访问；本地审计 release commit `d552cf90585e72c309c1aefdb20f23e605fa1370`，确认全网格循环和 evidence-changing shift，未检出 factorial/interaction 分析。
- [Treble 官方仓库](https://github.com/TREE985/Treble-Counterfactual-VLMs)：可访问；本地审计 release commit `f52197e48bd34a54508afbb49da25a26cb74be3f`，包含 `visual_shift`、`text_shift`、`cross_modal_shift`、PCA directions 与 intervention。论文语义和 release-source 语义不一致的本地问题仍在，必须采用双语义 envelope。
- [ONLY](https://github.com/zifuwan/ONLY)、[HulluEdit](https://github.com/VioAgnes/HulluEdit)、[CAI](https://github.com/Iris1946/CAI)、[MedVIGIL](https://github.com/hq0709/MedVIGIL)：仓库当前均可访问。
- PSF-Med 数据页可访问，但其所列代码仓库当前 404；不得声称已检查实现。
- Curse of Multiple Mediators 论文脚注给出代码 URL，但仓库当前 404；方法学结论来自论文正文，不来自代码复现。
- DMAS 与 Dual-Pathway 论文 HTML 未找到官方 GitHub 链接；不得写成 code-verified baseline。

## 5. 修改后的决定性实验

### 5.1 Behavioral gate：这是 CECD 是否存在的唯一入口

在 outcome-blind physician admission 后，对同一 image/claim 构造完整四格：

\[
s_{00},s_{10},s_{01},s_{11},\qquad
\Delta_s=s_{11}-s_{10}-s_{01}+s_{00}.
\]

冻结 dev 上的 readout、prompt families、render families 和 threshold。主检验必须证明 `Δ_s` 在 image-disjoint test 上：

1. 增量预测 reader-grounded clinical error；
2. 优于 clean margin、render main effect、prompt main effect、entropy、answer length；
3. 优于 MM-R3/TTA generic stability、Questioning-VQA 双轴指标、Context-Distorts grid score、Treble 双语义 scores 和 PID-style synergy；
4. 至少两个架构、多数 admitted finding/transform family 成立；
5. CE 之外，在 teacher-forced aligned OE atomic claims 上复现。

仅仅 `||Δh||` 非零没有意义；内部非线性表征完全可以对应四格都正确。

### 5.2 Mechanism gate：证明 joint-cell specificity，而非普通 steering

在 dev 冻结层和 readout 后，比较：

- interaction-component ablation；
- render-main-effect 与 prompt-main-effect matched-energy ablation；
- sign-permuted、image-permuted、same-norm random direction；
- full-orbit averaging 与 render-only/prompt-only averaging；
- Treble proceedings-faithful 与 source-faithful 两个版本；
- 至少一个静态 steering baseline（ASD/CIPHER 类）；
- 至少一个动态或 multimodal baseline（DMAS/CausalLens/HulluEdit/CAI 类，若架构兼容且官方实现可用）；
- PSF-Med prompt normalization/feature control，在兼容模型上执行。

干预只有在错误 joint cell 被选择性修复，同时 clean、render-only、prompt-only、clear-evidence cells 基本不变时，才支持 CECD-specific mechanism。若对所有 prompt-sensitive cases 都有效，则只是通用 prompt correction。

### 5.3 新增 patching validity controls

受 Curse of Multiple Mediators 约束，正式因果表需同时包含：

- denoising 与 noising 两个方向；
- patched activation distance 与 matched-distance control；
- PIE、NIE、INT 的可行分解，或明确降级 causal wording；
- single component 与 grouped patch 的稳定性；
- hook 后 interaction 是否真的被移除，以及 decoder 输出 interaction 是否重新生成；
- exact cancellation 与 per-cell norm restoration 分开报告，因为独立 rescale 通常会重新引入二阶项。

### 5.4 OE 保护性约束

自由生成四格 token 不能直接相减。OE 先固定候选 atomic claims，再在四格 teacher-force 同一 claim span。任何 mitigation 必须固定 positive claim 数 `K`、matched coverage、长度和 refusal rate，并报告 omission、polarity、location、attribute 与 certainty。收益若来自少说、统一阴性、hedge 或拒答，判失败。

## 6. 严格 NO-GO 与论文升级条件

任一项发生即停止 CECD 主线：

1. 医生不承认任一 render/prompt family 的独立等价性；
2. 乘积 mixed derivative 在 held-out test 上不优于两个 marginals + generic stability + Treble 双语义；
3. 只在一个模型、一个 finding、CE Yes/No 或单一 prompt family 成立；
4. causal intervention 对 prompt-only/clear cells 同样起作用，缺乏 joint specificity；
5. patching 结论在 noising/denoising、PIE/INT 或 activation-distance controls 下坍塌；
6. full-orbit averaging、TTA consistency 或现有 dynamic activation baseline 在相同准确率/coverage/延迟约束下匹配增益；
7. OE 改善由缩短、漏报、拒答或 blanket uncertainty 解释。

只有以下链条全部成立才有 ICLR oral 级可能：

```text
independent physician admission
        -> product-only reader-grounded clinical error
        -> incremental prediction beyond every marginal/near-neighbor construct
        -> joint-cell-specific mechanism under patching validity controls
        -> selective causal rescue beating generic activation methods
        -> CE + aligned OE/report replication without omission exchange
```

## 7. Reviewer-style verdict

- **问题重要性：3/3。** 医学 VLM 在不改变临床命题的显示/措辞变化下出现非可分错误，若真实存在，具有可信度与机制意义。
- **机制清晰度：2/3，条件性。** 输入级 mixed derivative 定义清楚，但 hidden-state patching 受 mediator interaction 和 token alignment 威胁。
- **新颖性：1.5/3，严格收缩后。** paraphrase、双轴稳定性、医学 cross-grid、cross-modal circuits、正交 steering 都已有；只剩 clinician-admitted product nonseparability。
- **可执行性：2/3。** 四格前向与现有 CECD 入口可做，但医生 admission、双架构 hooks、OE teacher forcing 与强 baseline 成本高。
- **当前总判定：Conditional GO for phenomenon; NO-GO for standalone mitigation-method novelty。**

当前最优策略不是继续给投影公式改名，而是用最少实验回答一个高价值二元问题：**严格等价条件下的乘积非可分性是否真实、是否独立、是否因果上只属于 joint cell。** 三个答案缺一，CECD 都不应进入论文主线；三个都成立，现有密集的 activation-steering 文献反而能成为强背景，突出 CECD 发现的是一个此前未被定义的临床输入机制对象。

# Evidence-Set Closure / Supervision Collision：一条成立但不足以成为当前主线的审计

> 冻结日期：2026-08-03 UTC  
> 范围：截至冻结日可检索的顶会、顶刊、最新 arXiv/medRxiv 与公开代码；本轮只做文献和 outcome-blind CPU substrate 审计，不运行 GPU，不读取模型 outcome，不修改共享评测。  
> 最终决定：**NO-GO。不能把已经失败的 SISC 扩写成一个跨 view / prior / history / metadata 的 Evidence-Set Closure 主线。**

## 摘要

本审计检验一个看似统一的命题：若目标 claim 的最小充分证据集没有包含在模型输入中，却仍作为确定正监督参与训练，模型会把与 claim 共现的代理变量学成伪证据，并在证据缺失时过度承诺。这个命题在统计意义上正确，也能把单图—全 study 报告、缺失 prior、缺失临床 indication、缺失 DICOM calibration 和缺失 MRI sequence 写成同一形式。然而，检索结果不支持把它作为当前的新机制主线。首先，CXR-ReDonE、Pragmatic RRG、LLM-RG4、MAIRA-2 与 ReMIND 已分别覆盖 prior-reference 清洗、image-uninferable target 清洗、四种 input-context 对应、真实附加输入消融及 missing-sequence hallucination；其中 LLM-RG4 已直接把 input-output mismatch 和 perfectly corresponded input/output 写成中心问题。其次，MIL/MIML 与 partial-label learning 已把 bag label 下放到 instance 所产生的歧义和 false-positive supervision 形式化多年。最后，本地虽然有两个独立的 source-typed substrate——VinDr 的 image-local reader boxes/DICOM metadata 与 SLAKE 的 physician-authored `vqa/kvqa` 类型和 KG triples——但都不能形成“同一 claim、同一语义、source support/refute/unavailable、自然缺源”的因果四格；MIMIC 本地样本和 MedVIGIL 也缺独立 source truth。因此唯一可能的新意只剩“collision dose 导致 unsupported emission 单调上升”的 exact-parent scaling law，但它目前只能依赖人工 label permutation，缺少自然桥，且不是一个新表示机制。按 ICLR oral 标准，当前应剪枝而非扩题。

## 1. 冻结研究问题

本报告只回答三个问题：

1. **RQ1：** 现有医学 RRG/VLM 与一般 weak-supervision 文献是否已实质覆盖“目标所需证据未输入，却仍被当作正监督”？
2. **RQ2：** Evidence-Set Closure 能否给出超越“obvious missing input”的统一、可证伪机制定律？
3. **RQ3：** 本地是否已有至少两种不依赖模型输出、足以支持一天 kill gate 和 exact-parent child 的独立 evidence-source truth？

审计角度不是“能否想出一个方法”，而是“一个强 reviewer 是否会认为它只是把多个已知 missing-input 问题重新命名”。目标读者是决定下一条 ICLR oral 主线的研究团队。

## 2. 检索与核验方法

检索从五个相互对抗的视角展开：

1. radiology target cleaning：`prior reference`, `uninferable report content`, `input-output mismatch`；
2. multi-view / longitudinal / clinical-context RRG：`view-specific`, `lateral`, `prior study`, `indication`；
3. typed quantitative evidence：`measurement hallucination`, `DICOM calibration`, `grounded report`；
4. missing-modality / missing-sequence learning：`modality ablation`, `natural-missing hallucination`, `incomplete multimodal`；
5. weak-supervision theory：`MIL`, `MIML instance annotation`, `partial multi-label`, `false candidate labels`。

每个进入正文的工作均用题名检索完成存在性核验，并用官方 proceedings、作者论文页、arXiv/medRxiv/PubMed 页面或官方仓库核对本文所述贡献。未能确认的工作不进入论证。代码存在性单独核验；“有代码”不等于代码包含论文的 construction-side annotation pipeline。

## 3. 正确形式化：不是一个最小集合，而是一族最小充分集合

令可用 evidence carriers 为

\[
\mathcal E=\{\text{current views},\text{prior study},\text{history/indication},
\text{metadata/calibration},\text{external knowledge}\}.
\]

对 claim \(c\)，不能只写一个“最小支持集” \(M(c)\)。同一 claim 可能被多种替代证据充分支持，例如设备位置可以由像素加 landmarks 支持，也可能由结构化测量记录支持。更严谨的对象是一族 inclusion-minimal sufficient sets：

\[
\mathfrak M(c)=\{M_1(c),\ldots,M_k(c)\}.
\]

给定实际输入 evidence set \(E\)，claim-level closure 定义为：

\[
\operatorname{Closed}(c,E)=
\mathbb 1\left[\exists M\in\mathfrak M(c):M\subseteq E\right].
\]

训练目标若将 \(c\) 标为 definite-positive，但 \(\operatorname{Closed}(c,E)=0\)，就发生 **supervision collision**。对 claim type \(c\) 定义 collision dose：

\[
q_c=P\big(\operatorname{Closed}(c,E)=0\mid y_c=\text{definite-positive}\big).
\]

这个定义带来一个严格但有限的结论：在只观察 \(E\) 的条件下，cross-entropy learner 的 Bayes optimum 是 \(P(y_c\mid E)\)，而不是不可观测的 source truth。若缺失 source 与可见代理相关，模型会学习代理；若代理相同而 source truth 不同，instance-level source support 不可识别。这与 MIML 中由 bag labels 推断 instance labels的歧义同构 [8–11]。

### 3.1 为什么这还不是一个新机制定律

上述结论是**可识别性边界**，不是对 transformer 内部机制的独特预测：

- 它不指定错误发生在 encoder、projector、decoder 或 optimization 的哪一阶段；
- “语言先验”“study-level co-occurrence”“missing modality imputation”和“标注噪声记忆”都能产生相同的 unsupported emission；
- 一个 claim 在统计上预测正确，并不等于获得了临床支持；反之，source 缺失也不保证 claim 为假；
- \(\mathfrak M(c)\) 依赖 claim 语义和临床任务。把 measurement、temporal comparison、view visibility 与 knowledge question 放在一个实验里，会把多个不同 data-generating processes 拼成一张表，而不是发现一个共同神经机制。

因此，只有一个更强的经验命题可能超过 obvious missing input：

> 在固定 parent、输入 token、目标 claim slots、label marginals、steps 与 coverage 后，只提高 structured collision dose \(q_c\)，会使**对应 source 缺失条件**下的 definite emission 单调增加；同剂量 random label noise 不产生相同的 source-selective slope；恢复 source 后性能恢复。

这可证伪，但它是 supervision-dose law，而不是 representation law。没有自然 collision bridge 时，它仍只是一个人工 weak-supervision model organism。

## 4. 领域碰撞：广义命题已经被连续覆盖

### 4.1 从 target cleaning 到 perfectly corresponded input-output

CXR-ReDonE [1]、Pragmatic RRG [2] 与 LLM-RG4 [3] 形成了最直接的递进：CXR-ReDonE 清除当前输入不存在的 prior-reference；Pragmatic RRG 不仅加入 indication，还明确识别 image-uninferable report content 并清洗 ground truth；LLM-RG4 进一步把传统 single-image-to-full-report 写成 input/output mismatch，构建 single/multi-view × no/with-longitudinal 四种“perfectly corresponded input and output”场景。因而“targets must be tailored to available evidence”不是本文可以重新占据的新问题。

| 工作 | 缺失/错配证据 | target 是否修正 | independent source truth | 因果隔离 | 对本题的占位 |
|---|---|---|---|---|---|
| CXR-ReDonE [1] | prior report/study | 删除或重写 prior references | span classifier/GPT rewrite，不是 claim-source 三态 truth | retrain cleaned dataset | 已占 prior cleanup |
| Pragmatic RRG [2] | indication 与 image-uninferable content | 加 indication；清洗 uninferable target | 规则/文本框架，非 per-source support 四格 | 多模型/指标比较 | 已占 pragmatic target observability |
| LLM-RG4 [3] | frontal/lateral/prior/history 四场景 | 构建 context-corresponded outputs | DiscBERT-style input-agnostic annotation，不是 view-level clinical truth | 四 setting 与 mixed training | **直接占据 broad Evidence-Set Closure** |

### 4.2 多视图与多上下文已经从“有用”走向“缺失会幻觉”

KCLVA [4]、MAIRA-2 [5] 与 ReMIND [7] 又从三个方向压缩剩余空间。KCLVA 从报告抽取 view-specific terms 并做 view-specific attention / many-to-many contrastive learning；它仍缺 independent per-image truth，因此没有完成 SISC 的 exact-parent causal proof，但已占据“full study report 与各 view 的文本对齐并非一对一”。MAIRA-2 同时输入 current frontal、optional lateral、prior frontal/report、indication、technique 和 comparison，并对 lateral/prior/context 做训练与推理消融。2026 年的 ReMIND 更直接定义 natural-missing modality hallucination 和 modality-family ablation 后的 counterfactual persistence，并用 modality-aware report-level reranking/correction 缓解。

| 工作 | evidence carrier | 输出约束 | 是否直接研究 source 缺失 hallucination | 代码/模型资源 |
|---|---|---|---|---|
| KCLVA [4] | PA/AP/lateral views | view-specific token weighting | 间接；核心仍是 alignment/NLG | accepted manuscript 可得 |
| MAIRA-2 [5] | lateral、prior image/report、indication、technique、comparison | grounded / ungrounded findings | 做 additional-input ablation，并量化 comparison mentions | HF model/custom code |
| ReMIND [7] | 六类 MRI sequence families + history | source-cue verifier、rerank、drop violating sentences | **直接：natural-missing + counterfactual persistence** | 数据受限，preprint 描述完整 |

这些工作之间的差别很重要：KCLVA 没有独立 source truth，MAIRA-2 主要看整体临床指标，ReMIND 的 violation detector 依赖 sequence cue lexicon；因此“collision dose causally shapes proxy learning”仍未被它们直接证明。但把三者放在一起，已经足以否定“输入缺哪种证据就不应生成哪种 claim”本身具有高新颖性。

### 4.3 Typed evidence 也已有强专门方法

FactCheXcker [6] 把定量 measurement 从通用生成中拆出，通过 query-code-update、专门定位/测量模块和 report update 处理 ETT-to-carina 距离，在 11 个 report generators 上降低 measurement error。它说明一个关键反例：当 claim 有严格计算语义时，最自然的方法是调用对应测量算子，而不是学习一个跨 prior/history/view/metadata 的统一 latent closure score。将 measurement 当作 Evidence-Set Closure 的第二验证类型，reviewer 很容易认为是在 FactCheXcker 上增加抽象术语。

同理，CXR-ReDonE 对 prior、Pragmatic RRG 对 indication、ReMIND 对 MRI sequence 都有各自的最小充分修复。现有证据更支持 **typed tools / typed targets**，不支持这些 failure 共用一个内部 erasure mechanism。

### 4.4 代码与 construction-side 可复现性审计

| 工作 | 官方资源 | 对本审计真正可复用的部分 | 关键缺口 |
|---|---|---|---|
| CXR-ReDonE | [`rajpurkarlab/cxr-redone`](https://github.com/rajpurkarlab/cxr-redone) | prior-span cleanup 与 cleaned retraining | 只处理 prior-reference；无 source-support 四格 |
| Pragmatic RRG | [`ChicagoHAI/llm_radiology`](https://github.com/ChicagoHAI/llm_radiology) | indication input、uninferable text cleaning | 不提供多 source 同-claim causal manifest |
| LLM-RG4 | [`zh-Wang-Med/LLM-RG4`](https://github.com/zh-Wang-Med/llm-rg4) | 四 context 数据、模型、DiscBERT 与完整训练入口 | construction 依赖其对应输出定义；没有 independent claim-source clinical truth |
| MAIRA-2 | [`microsoft/maira-2`](https://huggingface.co/microsoft/maira-2) | 多图/多 section prompt schema、模型 custom code | 受模型许可约束；不发布我们所需的 per-source adjudication |
| FactCheXcker | [`rajpurkarlab/FactCheXcker`](https://github.com/rajpurkarlab/FactCheXcker) | measurement query-code-update 与专门算子 | typed measurement pipeline，不能证明跨 source law |
| KCLVA | accepted manuscript 可公开下载 | view-specific extractor/attention 设计细节 | 截至冻结日未检索到作者公开代码；报告派生 view terms 不是 independent truth |
| ReMIND | preprint、supplement 与方法描述 | modality-family lexicon、natural-missing/counterfactual protocol | clinical data 受限；文中写明 NACC benchmark 待接收后公开，不能作为当前本地 exact-parent 数据 |

这张表揭示了一个重要差别：已有代码足以复现多种“按 available input 清理或补充 target/context”的工程路径，却没有一个公开资源给出跨 source、同 claim 的 `support/refute/unavailable` 真值。这个空缺是真实的 dataset gap，但 dataset gap 本身不等于机制 novelty。

## 5. 与 missing-modality、MIL 和 partial-label theory 的边界

### 5.1 Missing modality：默认假设与本题相反

系统综述 [12] 将 missing-modality learning 分为 modality imputation、representation recovery、architecture-focused robustness 与 model combinations；其统计对象通常是“同一个 downstream target 在部分模态缺失时仍尽可能被预测”。综述统计的大多数方法试图恢复缺失模态或其 representation，而不是输出 `unobservable`。Evidence-Set Closure 的临床立场反而是：当 target 的支持性依赖缺失 carrier 时，不应把从其它模态预测出的高概率当作当前证据。

这个差异有理论价值，但不是自动的新颖性：missing-modality literature 研究 prediction risk，我们研究 evidence support；二者的 loss 不同。若实验最后仍用 accuracy/F1 判断 missing-source 情况，方法会退化回 ordinary missing-modality robustness。只有 independent source-support truth 能把二者区分开。

### 5.2 MIL/MIML：SISC 的数学核心早已存在

MIML 把一个 bag 表示为多个 instances 并赋予多 labels [8]；instance annotation 文献明确研究“只有 bag-level labels 时如何推断每个 instance 的 labels” [9,10]。其典型假设是 bag label set 为 instance label sets 的 union，而不是把 union label 复制成每个 instance 的 positive。Doran 与 Ray [11] 还表明 instance concept 可学习需要覆盖性假设，例如 negative instances 在某些 negative bags 中以非零概率出现。

因此 SISC 的核心不是一种未知学习现象，而是：RRG pipeline 把一个合法 MIML bag 退化成多个错误的 single-instance positives。若没有 claim-view incidence truth，instance label 不可识别；若有 truth，最直接的修复是 MIML/instance annotation 或保持 study bag，而不是提出一个新的 decoder。

### 5.3 Partial-label / partial multi-label：false positive candidate 已被系统研究

ICML 2020 的 progressive identification [13]、ICLR 2024 的 candidate label set pruning [14] 和 NeurIPS 2024 的机制复盘 [15] 都把 false candidate labels、label purification 与 data-centric pruning 作为核心。它们主要是“一组候选里寻找真 label”，而 radiology report 是 structured multi-label generation，不能直接套算法；但 reviewer 会合理地把“unsupported report claim 是 false positive supervision”归入 weak-label disambiguation，而非全新理论对象。

真正未覆盖的交叉点很窄：**临床 claim 的 source observability 由独立证据标注定义，且 structured collision 的 dose–response 不等同于同剂量 random label noise。** 当前本地数据还不能识别这个交叉点。

## 6. 本地 outcome-blind substrate 盘点

### 6.1 Substrate A：VinDr-CXR 的 image-local reader boxes + DICOM metadata

审计对象：`/workspace/vinbigdata/train.csv`，SHA-256 `d3c52e5b96a73b9263e535ee803dde7fd053e4c8c6a50faf6b00f7dd8f5e2a80`。

| 数量 | 结果 |
|---|---:|
| DICOM train / test | 15,000 / 3,000 |
| train annotation rows | 67,914 |
| distinct images / readers / classes | 15,000 / 17 / 15 |
| rows with complete boxes | 36,096 |
| images with any non-`No finding` annotation | 4,394 |
| train DICOM with `Rows/Columns` | 15,000 / 15,000 |
| train DICOM with `PixelSpacing` | 12,848 / 15,000 |
| train DICOM with usable `ViewPosition` | 0 |

这是一种真正 independent、image-local 的医学 source annotation：radiologist boxes/labels 不依赖模型输出，DICOM calibration 也不来自报告生成器。它能定义当前图像 finding support，也能定义 detector-plane coordinate 的 pixel-to-mm transport。

但它不能单独通过 Evidence-Set Closure gate：

- 没有同 study 的 complementary view set 和 `ViewPosition`，不能重开 SISC；
- 缺 box 不能自动标成 `refuted` 或 `unassessable`；
- box × PixelSpacing 得到的是 detector-plane 几何，不等于投影 CXR 中解剖结构的真实物理尺寸；
- 数据没有自然 narrative measurement target。若人工从 box 生成 “X mm” claim，得到的是 synthetic operator test，而不是自然 supervision collision。

### 6.2 Substrate B：SLAKE 的 physician-authored visual/knowledge source type + KG triples

审计对象与 SHA-256：

- `train.json`: `7a9fb81f9d7bb145c3d7e9bc8d3b9016ba18b6539ea5031a53bae8fd3cd991c3`；
- `validation.json`: `32b016440b0c3be11056a78a18eeab46333268407fbb6e6b32f9f4c2debc50f6`；
- `test.json`: `6be8f7b4c5a46cdbc713a5210a25b6ed5aa1fd1574c83cefb4f998131f17c2c3`。

官方 SLAKE 工作说明数据由 experienced physicians 标注，并附结构化 medical knowledge base [16]。本地释放统计为：

| 数量 | 结果 |
|---|---:|
| QA rows / unique images | 14,028 / 642 |
| English `vqa / kvqa` | 6,148 / 885 |
| Chinese `vqa / kvqa` | 6,140 / 855 |
| total `kvqa` rows with triples | 1,740 |
| images containing both `vqa` and `kvqa` questions | 345 |

它提供第二种不依赖模型 outcome 的 source-type annotation：`vqa` 与 `kvqa` 区分视觉问题和 KG/knowledge 问题，且保存 KG triples。

但它同样不满足正式 gate：visual 与 knowledge questions 通常不是同一 proposition；不存在同一 claim 在 `image-only / knowledge-only / both / neither` 四格中的 support/refute/unavailable；不能把不同问题的正确答案差异解释为 source collision。它适合作为 source-type routing sanity check，不适合作为 exact-parent natural bridge。

### 6.3 其它本地材料为什么不能凑成第二条 formal source

| 本地材料 | 可观察到什么 | 缺失什么 | 决定 |
|---|---|---|---|
| MIMIC report subset | 700 image rows、653 studies、44 paired studies；同 study target 100% identical | official view metadata 为 0；无独立 claim-view `visible/refuted/unassessable` | SISC 已 NO-GO |
| MedVIGIL release audit | ROI-only/masked、knowledge rewrite、indication 字段 | 无 claim-level evidence source、source support、prior/history link；rewrite 改 proposition | 不可作 source-erasure truth |
| IU X-ray | multi-image study + shared report | 无 independent per-image finding truth | 只会复制被检验的污染源 |
| VinDr reader votes | image-local polarity/boxes | 无 longitudinal/history/knowledge source pair | 只够单 source |

结论不是“本地没有两种 source annotations”：VinDr 与 SLAKE 的确是两种独立 annotation substrate；结论是**没有两种同时满足同-claim source closure contract 的自然 substrate**。把它们放在同一张结果表，只能展示方法可跨任务运行，不能证明一个共同机制。

## 7. 一天 kill gate：先杀 natural bridge，不碰 GPU

### 7.1 冻结 admission contract

Evidence-Set Closure 只有在一天内同时满足以下条件时才允许 exact-parent training：

1. 至少 **2 个独立 source families**；每个 family 至少 100 个 patient/study-disjoint claim groups；
2. 每个 family 至少 3 个 frozen claim types，每类至少 30 个 source-open cases；
3. 同一 claim wording、polarity、certainty 和 clinical referent 在 source conditions 间完全不变；
4. 每个 source 对 claim 有独立的 `support / refute / unavailable`，并有 ROI/span/metadata pointer；
5. 至少形成 `both-support / A-only / B-only / neither` 四格；不能用 absent box 当 negative，不能用共享 report、模型 classifier 或 LLM judge 定义 truth；
6. 在真实 training construction 中，至少一个 source family 的自然 collision dose \(q_c\ge 0.15\)，且 95% cluster-bootstrap CI 下界大于 0.10；
7. output 可表示 `unobservable` 而不删除 claim slot，输入和输出长度可严格匹配。

按当前本地审计：VinDr 缺同-claim第二 source，SLAKE 缺同-proposition source 四格，MIMIC/MedVIGIL 缺 independent truth。**Gate 已失败；`gpu_authorized=false`。**

### 7.2 允许重开时的 exact-parent children

若未来获得合格数据，只允许以下极简设计。它不混入 decoder trick、retrieval 或 activation steering。

所有 children 使用相同公开 parent checkpoint、LoRA rank、optimizer、steps、seed family、sample count、input token count 与固定 ontology claim slots。缺失 source 用等长 typed `SOURCE_UNAVAILABLE` tokens 占位；每个 claim slot只生成 `supported / refuted / unobservable`，从而固定 coverage 和输出长度。

| child | source content | target assignment | 目的 |
|---|---|---|---|
| `CLOSED-q0` | 自然 available/unavailable | 全部服从独立 source truth | clean control |
| `COLLISION-q` | 与 q0 完全相同 | 将比例 \(q\in\{.15,.30,.45\}\) 的 definite-positive 放到 source-open cases | 估计 structured collision dose |
| `RANDOM-q` | 与 q0 完全相同 | 同剂量、同 claim/label marginals 的随机 assignment noise | 排除普通噪声记忆 |
| `RESTORED-q` | 对 `COLLISION-q` 的正例恢复缺失 source，token 数仍等长 | target 不变 | 证明错误来自 source absence 而非例子难度 |

为了固定 label marginals，`COLLISION-q` 不是凭空增加 positives，而是在同一 claim type、同一 split、matched difficulty 内把 positive slot assignment 从 closed cases 与 open cases做成预注册 permutation；`RANDOM-q` 使用同一 permutation size，但不与 source state 对齐。这样唯一系统变化是 positive supervision 与 source closure 的相关性。

预注册主检验：

\[
\operatorname{logit}P_\theta(\text{definite-positive}\mid \text{source-open})
=\alpha+\beta q+u_{claim}+u_{patient}.
\]

必须同时满足：

- `COLLISION` 的 \(\beta>0\)，patient/claim-cluster bootstrap 95% CI 排除 0；
- `COLLISION` slope 显著大于 `RANDOM` slope；
- `RESTORED` 消除 source-open selective error，而不是仅提高所有输出的 positive rate；
- fixed claim slots 下 unsupported definite emission 相对增加至少 50%，clear supported recall 变化不超过 1pp；
- 至少两个 source families 的标准化 slope 同号，且 leave-one-claim-type-out 后保持；
- 不用 report length、claim count、refusal、uniform hedge 或温度解释结果。

一项失败即 NO-GO；不调 \(q\)、不换 truth source、不把两种 source 的不同 failure 平均成“总体显著”。

## 8. Fatal-flaw audit

### 8.1 最强 novelty claim 已被 LLM-RG4 直接写出

若论文摘要写“现有 RRG 目标包含输入无关信息，我们构造与 available input 对应的输出”，它与 LLM-RG4 的摘要级贡献几乎同义。加入更抽象的 `evidence set` 名称不能创造 novelty。

### 8.2 跨 source 统一会损害单一机制

View visibility 是投影几何问题，prior comparison 是时间关系问题，measurement 是 calibration/landmark 算子问题，history 是 pragmatic intent，knowledge 是外部事实检索。它们共享“缺输入”这个逻辑形式，却不共享一个已证实的表示或优化机制。只有 collision-dose law 可能统一它们；当前没有自然数据支持该 law。

### 8.3 `unobservable` 不是所有缺 source 的正确答案

有些 current-image claims 在 prior 缺失时仍完全可观察；有些 knowledge claims 可从模型参数正确回答；有些临床 finding 可由替代 source 支持。若没有 \(\mathfrak M(c)\) 的专业标注，硬性 source gate 会制造遗漏或错误拒答。这个问题不能由 entropy 或自动 labeler 可靠解决。

### 8.4 Exact-parent synthetic result 不足以支撑医学 oral ceiling

人工 label permutation 能漂亮地证明 learner 会学习 structured noise，但 weak-supervision literature 已预期这一点。没有自然 collision prevalence、独立 clinical source truth 和真实 frozen VLM behavior bridge，结果最多是一个诊断性 model organism，不足以成为 ICLR oral 主线。

### 8.5 最可信的研究动作是保持窄问题，而不是上升叙事

SISC 若未来获得双医生 per-view `visible/refuted/unassessable` 标注，仍可以作为**单一** study-to-image supervision collision 问题重开；measurement gauge 若获得严格 DICOM/landmark truth，也应作为**单一** identifiability 问题研究。把两者与 prior/history 拼成 Evidence-Set Closure 会降低而不是提高论文 taste。

## 9. 结论：逐项回答 RQ

**RQ1：现有工作是否已覆盖核心问题？** 是。CXR-ReDonE 和 Pragmatic RRG 已清理 image-uninferable targets；LLM-RG4 已以 input-output mismatch / perfectly corresponded contexts 为中心贡献；MAIRA-2 已把 lateral/prior/sections 作为现实输入并做消融；ReMIND 已直接研究 missing MRI sequence hallucination；FactCheXcker 已专门解决 measurement。MIL/MIML 和 partial-label theory又覆盖了 bag-to-instance ambiguity 与 false-positive supervision。

**RQ2：是否有真正统一且高 ceiling 的机制定律？** 目前只有一个条件性候选：structured collision dose \(q\) 对 source-open definite emission 的 source-selective monotonic law，并以 random-noise 和 restored-source controls 区分。但它是可识别性/监督规律，不是已发现的 VLM 内部机制；跨 source 时容易变成拼盘。没有 natural bridge 前，不具备 ICLR oral ceiling。

**RQ3：本地是否有两种合格 substrate？** 本地有两种独立 source-typed annotations：VinDr 的 image-local readers/boxes + DICOM metadata，以及 SLAKE 的 physician-authored visual/knowledge tags + KG triples；但它们都不能形成同一 claim 的 source support/refute/unavailable 四格。MIMIC 和 MedVIGIL 也不能补齐。因此一天 gate 当前明确失败。

最终决定为 **NO-GO**：不运行 GPU，不把 SISC 包装为 Evidence-Set Closure，不把 measurement/prior/history/knowledge 拼成一个主线。保留本报告中的 formalism 和 exact-parent protocol 作为未来数据到位时的审计工具；当前研究树应 pivot 到具有单一自然变量、独立真值和未被摘要级占据的问题。

## References

[1] Vignav Ramesh, Nathan Andrew Chi, Pranav Rajpurkar, “Improving Radiology Report Generation Systems by Removing Hallucinated References to Non-existent Priors,” ML4H / PMLR 193, 2022.

[2] Dang Nguyen, Chacha Chen, He He, Chenhao Tan, “Pragmatic Radiology Report Generation,” Machine Learning for Health / PMLR 225, 2023.

[3] Zhuhao Wang, Yihua Sun, Zihan Li, et al., “LLM-RG4: Flexible and Factual Radiology Report Generation across Diverse Input Contexts,” AAAI, 2025.

[4] Jinlong Zhu, Ping Lu, “KCLVA: Knowledge-enhanced Contrastive Learning and View-specific Attention for Chest X-ray Report Generation,” MIUA / LNCS 15916, 2025.

[5] Shruthi Bannur, Kenza Bouzid, Daniel C. Castro, et al., “MAIRA-2: Grounded Radiology Report Generation,” arXiv:2406.04449, 2024.

[6] Alice Heiman, Xiaoman Zhang, Emma Chen, et al., “FactCheXcker: Mitigating Measurement Hallucinations in Chest X-ray Report Generation Models,” CVPR, 2025.

[7] Diala Lteif, Shuyue Jia, Subhrangshu Bit, et al., “Vision-language Framework for Multi-sequence Brain Magnetic Resonance Imaging,” medRxiv:2026.03.30.26349106, 2026.

[8] Zhi-Hua Zhou, Min-Ling Zhang, Sheng-Jun Huang, Yu-Feng Li, “Multi-Instance Multi-Label Learning,” Artificial Intelligence 176(1), 2012; arXiv version 2008.

[9] Forrest Briggs, Xiaoli Z. Fern, Raviv Raich, Qi Lou, “Instance Annotation for Multi-Instance Multi-Label Learning,” ACM TKDD, 2013.

[10] Anh T. Pham, Raviv Raich, Xiaoli Z. Fern, “Dynamic Programming for Instance Annotation in Multi-instance Multi-label Learning,” arXiv:1411.4068, 2014.

[11] Gary Doran, Soumya Ray, “Learning Instance Concepts from Multiple-Instance Data with Bags as Distributions,” AAAI, 2014.

[12] Renjie Wu, Hu Wang, Hsiang-Ting Chen, Gustavo Carneiro, “Deep Multimodal Learning with Missing Modality: A Survey,” TMLR, 2026; arXiv:2409.07825.

[13] Jiaqi Lv, Miao Xu, Lei Feng, et al., “Progressive Identification of True Labels for Partial-Label Learning,” ICML, 2020.

[14] Shuo He, Chaojie Wang, Guowu Yang, Lei Feng, “Candidate Label Set Pruning: A Data-centric Perspective for Deep Partial-label Learning,” ICLR, 2024.

[15] Jiaqi Lv, Yangfan Liu, Shiyu Xia, et al., “What Makes Partial-Label Learning Algorithms Effective?” NeurIPS, 2024.

[16] Bo Liu, Li-Ming Zhan, Li Xu, et al., “SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering,” ISBI, 2021.

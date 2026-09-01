# CECD 与 Specificity Ratchet：2025--2026 机制级碰撞审计

日期：2026-08-02  
范围：只读扫描；不涉及 common eval 或 GPU 实验。

## 摘要

本次审计冻结两个问题：

1. 是否已有工作测量两个**各自语义/临床等价**的视觉与语言变化对 VLM 错误的不可加联合效应，并给出机制与因果干预？
2. 是否已有工作把开放式生成中的错误刻画为：模型先停留在视觉支持的父 claim，随后沿临床语义层级自发升级到不受支持的子 claim？

首轮检索到的五个最近邻已经覆盖了医学 VLM 的 paraphrase flip、医学图文扰动审计、图像×文本的 (2\times2) 设计、prompt-copying 的因果 attention heads，以及随输入 query granularity 增长的 fine-grained hallucination。第二轮又核验了 grounding--sycophancy 联合评测、ICLR 2026 的主动视觉重定位训练，以及 anatomy-aware 多尺度模型。因而，CECD 不能再声称首次发现医学 VLM 的措辞敏感性、图文联合效应或 prompt-over-vision；Specificity Ratchet 也不能声称首次研究 fine-grained/attribute hallucination或首次利用局部视觉证据。仍未被这些工作覆盖的窄而清晰的空间是：

- **CECD**：保持 clinical proposition 与视觉证据均不变，只改变表达/显示 nuisance，估计经过两个主效应中心化后的 mixed discrete derivative，并定位这种非可分性如何形成 reader-grounded error。
- **Specificity Ratchet**：不把错误子 claim 写入问题，而是在开放生成轨迹中观察“受支持父节点 \(\rightarrow\) 不受支持子节点”的自发越界，并以回退到最近受支持祖先、而非删除整个 claim，做保内容因果干预。

审计结论是两个方向均为 **conditional go**：前者新颖性尚在但门槛很高，必须通过人工 clinical-equivalence admission 并在至少两个模型上得到二阶效应；后者的问题空间更干净，但在人类 adjudication 证明 parent/child/support 三者之前，仍只是有吸引力的假说。

## 1. 检索方法与研究问题

检索窗口以 2025--2026 为主，补充直接定义问题边界的相邻工作。入口包括 arXiv、ACL Anthology、OpenReview、官方项目页、官方 GitHub/Hugging Face 数据页。关键词从四组互相对抗的视角展开：

- equivalence / paraphrase / image style or render / prompt-image interaction / factorial design；
- prompt-induced hallucination / modality dominance / causal heads / hidden-state intervention；
- specificity / granularity / over-specification / parent-child ontology / attribute-location-severity；
- open-ended generation / minimal editing / clinical claim / radiology hallucination。

只纳入标题、作者和核心方法能够从官方论文页或官方 release 核验的工作。下面五项是按**机制碰撞强度**而非标题相似度筛出的最近邻。

## 2. 五个最近邻：四层比较

| 工作 | Phenomenon | Mechanism | Intervention | 原论文 claim | 与我们的精确碰撞 | 仍然存在的 delta |
|---|---|---|---|---|---|---|
| [PSF-Med](https://arxiv.org/abs/2602.21428)（2026）及其[官方数据页](https://huggingface.co/datasets/saillab/psf-med) | 同一胸片上的 meaning-preserving paraphrase 可导致 Yes/No flip；六个医学 VLM 的 flip rate 差异很大。后续数据页还报告了 clinician audit，并加入 VinDr。 | 在 MedGemma-4B 的 SAE 中找到 layer-17 prompt-framing feature；causal patching 将其与 margin shift 联系起来。 | clamp 该 feature；另有 prompt normalization。 | paraphrase stability 必须与 image reliance 联合评价；一致不等于视觉 grounding。 | **CECD 的语言等价轴、医学场景、hidden mechanism 与 intervention 均被直接压住。** | 它保持图像固定，没有把临床等价 render 与 paraphrase 完整交叉，也没有估计去除 render/prompt 主效应后的二阶项。CECD 只能主张 **clinical-equivalence cross-modal non-separability**，不能主张“首次医学 paraphrase mechanism”。 |
| [VB: Visibility Benchmark](https://arxiv.org/abs/2603.06680)（2026）及其[官方代码](https://github.com/neilt93/Visual-Benchmark) | 用 minimal image edit × minimal text edit 构造 (2\times2) XOR family；正文明确把 double-flip 当作 unexpected interaction 的诊断。 | 无 hidden-state mechanism；比较 text-flip 与 image-flip robustness。 | 无正式 mitigation。 | visibility、abstention 与 minimal-edit robustness benchmark。 | **CECD 的 factorial 外形、四单元设计和“联合变化可能出现意外交互”的动机已经存在。** | VB 的两个 edit 被设计为改变 gold semantics：单 flip 使标签翻转，double flip 再翻回。CECD 的两个操作都必须保持同一 clinical claim 和同一证据状态；目标是测量本应为零的 centered interaction，而不是 XOR 能力。VB 也未做 reader-grounded clinical truth 或因果定位。 |
| [Mechanisms of Prompt-Induced Hallucination in Vision--Language Models](https://aclanthology.org/2026.acl-long.1941/)（ACL 2026） | prompt 故意夸大图中对象数量；随着计数难度增加，模型更服从文本而非图像。 | 三个 VLM 中识别少量 model-specific prompt-induced-hallucination attention heads。 | head ablation 使模型更多回到视觉证据，摘要报告 PIH 至少下降 40%。 | prompt copying 是可被少数 heads 因果调控、但实现方式依架构而异的机制。 | **“文本压过视觉”“模型特异的 cross-modal circuit”“causal ablation”都不能作为 CECD 的独立新颖点。** | 它使用 false-premise/leading prompt，改变 proposition；CECD 使用 proposition-与 speech-act-equivalent prompts，不向模型注入错误答案。CECD 的必要新增量是：错误只在两个等价 nuisance 联合时超过其可加主效应，并能通过 interaction-specific、而非 prompt-head 通用抑制来修复。 |
| [MedVIGIL](https://arxiv.org/abs/2605.07919)（2026）及其[官方数据页](https://huggingface.co/datasets/jhq0709/MedVIGIL) | 医学 VLM 在 paraphrase、negation、specificity-drop、false premise、knowledge-only 与 ROI-masked/ROI-only 图像下发生 silent failure；含 paired OE artifact。 | 行为层面的 evidence-conditional safe-failure 分解；没有定位内部生成 circuit。 | 主要是 benchmark/audit，不是隐藏态 mitigation。 | 由放射科医生构造和 adjudicate 的 broken-evidence 医学 VLM 安全评测。 | **医学图文双侧扰动、specificity 操作、ROI 图像变换与医生 adjudication 均已被覆盖。** | 它的主要 probes 要么改变 proposition/gold，要么破坏证据；不是“两个临床等价操作联合后才出错”。其 specificity-drop 是输入侧、gold-preserving rewrite，不是 OE 输出侧的 spontaneous parent-to-child escalation。它反而把我们的 admission 标准抬高：人工等价审查不能省。 |
| [FINER](https://arxiv.org/abs/2603.17662)（2026）及其[官方项目页](https://explainableml.github.io/finer-project/) | query 从单对象逐步增加到多对象、属性和关系；一个错误细节被多个真实元素包围时，false-positive hallucination 随 granularity 明显增长；另有 ill-posed “what” queries。 | 归因于 fine-grained mismatch detection / binding failure；没有追踪开放生成中父 claim 到子 claim 的时间或层级轨迹。 | 用正负 fine-grained queries 做 DPO（FINER-Tuning）。 | 现有 coarse hallucination benchmark 漏掉 fine-grained negative queries；输入 granularity 是重要压力源。 | **“越具体越易 hallucinate”“真实 coarse content 掩护错误 attribute/relation”“fine-grained mitigation”均已被占据。** | FINER 把错误 detail 预先写入 CE query，正确答案始终是拒绝/否定。Specificity Ratchet 的剩余空间是 OE 模型自己先生成正确父 claim，再在后续 token/句法扩展中选择不受支持的后代；评价单位是 ontology edge 与生成轨迹，干预是 child→nearest-supported-ancestor 的等长/保 claim 修复。 |

### 2.1 第二轮补充：三个必须进入边界或 baseline 治理的邻居

| 工作 | 它已经占据什么 | 与当前机制是否直接碰撞 | 对实验设计的约束 |
|---|---|---|---|
| [To Agree or To Be Right?](https://arxiv.org/abs/2603.22623)（2026）及其[官方代码](https://github.com/UTSA-VIRLab/AgreeOrRight) | 在 6 个 VLM、3 个医学 VQA 数据集上联合测 hallucination 与 sycophancy，并提出 L-VASE、CCS、CSI；论文报告二者存在 trade-off。 | **不直接碰撞。** 它是输出决策的联合安全评测，没有定义 physician-supported parent→unsupported child，也没有逐层 own/swap replay。 | 任何降低 hallucination 的方法都应检查是否只是更服从提示、统一保守或改变 calibration；不能只报单一 grounding 指标。 |
| [MedVR](https://arxiv.org/abs/2604.08203)（ICLR 2026）及其[官方仓库](https://github.com/alibaba-damo-academy/MedVR) | 以 Entropy-guided Visual Regrounding 主动选择 zoom-in 区域，并用 Consensus-based Credit Assignment 做无中间标注的 agentic RL。 | **相邻但不直接碰撞。** 它改变训练与推理时可见图像，通过主动局部重观察提高视觉推理；当前 Ratchet 测的是冻结完整回答中 unsupported descendant 的层间形成及 same-anatomy image-swap survival。 | 必须承认“entropy→精细视觉重定位”已被占据。MedVR 只应进入 `paper_native` 候选，不能伪装成跨 Huatuo/Hulu/LLaVA 的通用 decoder。2026-08-02 官方 HEAD 只有 Apache-2.0 训练入口；README 明示工具评测代码后续发布，模型徽章回链 GitHub，未给已训练 checkpoint，故当前 `not_admissible`。 |
| [Anatomy-VLM](https://arxiv.org/abs/2511.08402)（WACV 2026） | 用解剖 ROI、多尺度信息和结构知识做 fine-grained 医学解释与 OOD/分割验证。 | **不直接碰撞。** 它是输入表示与模型架构，未研究自然 OE 回答中 parent→child 的生成轨迹或 image-swap 后的 late transition。 | 若未来声称 Ratchet 来自“缺少局部解剖证据”，必须与 anatomy/ROI-aware 模型区分；当前冻结实验只允许回答 unsupported specificity 何时形成，不允许把原因写成 ROI 缺失。 |

第二轮的关键结论不是“再加三个方法跑满矩阵”，而是分层治理：grounding--sycophancy 工作进入评价边界，Anatomy-VLM 进入相关模型边界，MedVR 进入 `paper_native` 的 T0 清单。三者都不进入当前 Huatuo/Hulu `common_protocol` 解码主表。

## 3. 跨论文综合：哪些表述已经不能再用

PSF-Med 与 ACL 2026 PIH 从互补方向共同表明：医学 paraphrase flip 和一般 prompt copying 都已可被定位并干预；前者固定图像并定位 SAE feature，后者使用错误文本前提并定位 attention heads。因此，CECD 若只报告“换 prompt 后回答不同”或“文本先验压过视觉”，即使在更多医学模型上复现，也只是扩展而不是新机制。

VB 与 MedVIGIL 又从设计层面压缩空间：VB 已有明确的 image×text (2\times2) family，MedVIGIL 已有医生构造的医学 text/image perturbation family。两者均未回答的不是“联合评测是否有价值”，而是一个更严格的问题：当两个变化分别都被医生判为不应改变 claim support 时，模型的联合响应是否仍无法由两个边际响应相加解释。换言之，CECD 的科学对象必须是 **non-separability under admitted equivalence**，不是 robustness 平均下降。

FINER 与 MedVIGIL 对 Specificity Ratchet 形成边界：前者证明输入更细粒度会增加错误，后者已把 specificity-drop 纳入医学安全 probe；MedVR 与 Anatomy-VLM 又占据了 entropy-guided zoom 和 anatomy-aware fine-grained representation。它们都没有把自然开放回答看作沿 claim ontology 行走、并用冻结完整回答的层间 own/swap replay 区分图像存活与语言延展的过程。因而，真正未被覆盖的 cell 是 **output-side, spontaneous, trajectory-level, clinically adjudicated hierarchy crossing**。如果实验只统计 unsupported attributes 的比例、把更细 query 作为诱因，或只显示 zoom/ROI 提升，它会退化为已有工作的医学版本。

## 4. 方向 A：CECD 的 go/no-go

### 判定：Conditional GO；当前不得进入正式 claim 阶段

CECD 可以继续，但论文叙事必须冻结为：

> Under independently clinician-admitted equivalence of image rendering and clinical wording, medical VLM evidence scores exhibit a reader-grounded cross-modal non-separability that is not explained by either marginal sensitivity.

继续的最低门槛：

1. 人工 admission 必须先证明每个 render family 不系统改变 finding visibility/support，每个 prompt pair 保持 proposition 与 speech act；否则所谓 interaction 只是两个语义变化的普通组合。
2. 主统计量必须是预注册的 centered mixed derivative，并同时控制 clean margin、render main effect、prompt main effect、view、entropy、length 与 full-orbit exclusion。
3. 至少两个模型、多个 finding、image-cluster bootstrap CI 排除零；单一模型或均值显著不够。
4. 机制定位必须区分三种解释：prompt-only feature（PSF-Med）、prompt-copying heads（PIH）和真正依赖 render×prompt joint cell 的 path。若 patch 仅抑制所有 prompt sensitivity，则机制 claim 失败。
5. 干预需选择性降低 interaction-associated clinical error，同时不损伤等价 orbit 的 marginal accuracy；否则只是普通 calibration/robustness tuning。

明确 no-go 条件：人工 admission 不通过；二阶效应被任一主效应或输出长度解释；仅一个模型成立；或 hidden intervention 对 prompt-only case 同样有效而没有 interaction specificity。发生任一项时，CECD 最多作为 benchmark negative result，不能包装为新机制。

### 推荐改名与禁用措辞

- 推荐：**Clinical-Equivalence Cross-Modal Non-Separability** 或保留 CECD 但在标题中写明 *under equivalence*。
- 禁用：“first image-text factorial benchmark”“first medical paraphrase mechanism”“prompt-induced hallucination is caused by ...”“composition hallucination”这类宽 claim。
- mixed derivative 不应称 commutator；输入操作没有顺序结构。

## 5. 方向 B：Specificity Ratchet 的 go/no-go

### 判定：Conditional GO，问题空间比 CECD 更干净，但机制证据尚为零

可以保留的核心命题是：

> In open-ended medical generation, a model can preserve a reader-supported parent finding while irreversibly adding an unsupported descendant qualifier; this output-side escalation is distinct from failure on a fine-grained query.

必须把“层级”落实为每条 edge 的可审查结构，而不是修辞：

\[
c_{child}\Rightarrow c_{parent},\qquad
S(c_{parent}\mid x)\ge t_p,\qquad
S(c_{child}\mid x)<t_c.
\]

随后证明生成轨迹出现 parent→child 的风险增长，例如 finding→laterality、finding→size/morphology、finding→subtype；etiology 由于常常不是单图可观察，应单独标为 knowledge/inference edge，不混入视觉 ratchet。

继续的最低门槛：

1. 两名医生独立确认 parent support、child unsupported 与 edge validity，并 adjudicate；自动 RadGraph/LLM judge 只能抽候选，不能定义真值。
2. 使用自然 OE 草稿，不在问题中植入错误子节点；否则与 FINER 的 negative query 设置重合。
3. 必须证明 transition，而不只是最终回答含细节：至少利用 token logit、候选 continuation、layerwise claim score 或受控 completion，显示父 claim 在升级前可被支持、子 claim 在语言展开中获得额外概率。
4. 机制要排除“答案更长所以错误更多”：matched token budget、matched claim count、matched parent presence，报告每 parent-conditioned edge hazard。
5. 干预必须把 unsupported child 回退到最近受支持祖先，保持 parent identity、极性和 claim count；与删除 child、整句 hedge、缩短回答、统一阴性做 matched-content 对照。

明确 no-go 条件：医生无法稳定同意 edge/support；错误主要是 fabricated parent 而非 supported-parent→unsupported-child；没有时间/层级上的升级证据；或 mitigation 的收益完全来自删词/缩短/拒答。此时保留“细粒度错误 taxonomy”没有 ICLR oral 级机制贡献。

## 6. 审计后的实验优先级

1. **CECD 不绕过 human admission。** 已产生的 pre-admission model cells 只能算 engineering evidence；先完成盲审再恢复正式统计。
2. **Specificity Ratchet 优先完成医生 pack adjudication。** 它决定 ontology edge 是否真实，是该方向最便宜也最关键的 substrate gate。
3. 若两个 gate 都通过，先做最小因果区分：CECD 测 joint-cell-specific patching；Ratchet 测 child token 出现前后的 parent/child margin transport。不要先开发大而全 mitigation。
4. 论文组合只有在两个现象共享一个可验证机制时才成立，例如同一 cross-modal integration stage 同时放大 joint nuisance 与 unsupported descendant；仅仅都能用 steering 修复不构成统一机制。

## 7. 自我对抗检查

- **反向证据是否搜索过？** 已特别搜索 image-text factorial、paraphrase mechanisms、fine-grained negative queries、ontology over-specification、minimal editing 和医学 ROI perturbations；最危险的结果是 PSF-Med、VB、MedVIGIL 与 FINER，而非被排除在叙事外。
- **是否把“没找到”写成绝对 first claim？** 没有。结论仅限“本次核验的五个最近邻未覆盖”，论文仍应避免 first/only。
- **是否可能把 dataset design 当 mechanism？** 是主要风险。CECD 只有二阶 effect + causal localization 才是 mechanism；Specificity Ratchet 只有生成轨迹 + selective intervention 才是 mechanism。
- **是否可能由评判不一致制造现象？** 两个方向都可能，因此 human admission/adjudication 是前置 gate，而不是事后补充。

## 8. ICLR Oral reviewer fatal-flaw 结论

### CECD

两个致命风险：

1. **Construct validity（F6）**：只要医生认为任一 render 改变了 finding visibility，或任一 prompt 改变了 proposition/speech act，mixed derivative 就不再表示“等价条件下的非可分性”，而只是两个真实 signal change 的交互；更多样本无法挽救这个解释。
2. **Mechanism identifiability / novelty（F1+F6）**：PSF-Med 已有医学 paraphrase feature，PIH 已有 prompt-copying heads，VB 已有 image×text 四格设计。若 CECD 的 patch 同样修复 prompt-only cell，或二阶项不能定位到 joint-cell-specific causal path，reviewer 会把它判为已有 prompt sensitivity 的 factorial 重测。

只够普通 accept 的结果：医生 admission 通过，二阶效应在两个模型和多个 finding 上稳健，且控制主效应后仍显著，但只有行为统计、没有 joint-specific causal localization；这是一篇扎实的 robustness/benchmark paper，不是 oral 级机制论文。

最能改变 verdict 的 decisive experiment：对四格轨道在同一层构造

\[
\Delta h_l=h_l^{11}-h_l^{10}-h_l^{01}+h_l^{00},
\]

只 patch/消去这个 norm-matched interaction residual。若它能在错误 joint cell 中恢复 reader-grounded claim，同时对两个 marginal cells、clean cell、prompt-only PSF cases 和 clear findings 几乎无影响，并在第二个架构复现，这一结果会把 CECD 从“统计交互”提升为新的因果对象。

### Specificity Ratchet

两个致命风险：

1. **Clinical hierarchy 不可验证（F6）**：parent→child edge、parent support 和 child unsupported 任一项若医生不能稳定一致，所谓 ratchet 只是 claim parser 的粒度选择或报告风格差异，现象本身不成立。
2. **Trajectory 伪影（F6）**：最终回答里出现 unsupported detail 不证明发生了“升级”；它可能只是答案长度、模板共现或从一开始就存在的错误先验。若回退法靠删词降低错误，更会被 reviewer 视为 coverage trade-off。

只够普通 accept 的结果：跨模型获得医生确认的 parent-supported/child-unsupported prevalence，并且 ancestor backoff 在 matched length/claim-count 下改善 precision 而不伤 parent recall，但没有证明生成时的方向性转移；这是有价值的 error taxonomy + mitigation，不是 oral 级机制。

最能改变 verdict 的 decisive experiment：构造医生确认、parent 与 lexical prefix 匹配的两组 edge——child 真正受支持与仅 parent 受支持——在 teacher-forced parent 结束处逐层追踪 child-vs-parent margin。若两组在视觉整合期可分，而 parent-only 组只在后续语言生成阶段跨过 child decision boundary；随后只 patch 该 crossing stage 就把 unsupported child 替换为最近受支持祖先，同时保持长度、positive-claim count、parent polarity，并不抑制 matched supported children，这将直接证明“ratchet”而非静态 fine-grained error。

## References

[1] B. Sadanandan and V. Behzadan, “PSF-Med: Measuring and Explaining Paraphrase Sensitivity in Medical Vision Language Models,” arXiv:2602.21428, 2026.

[2] N. Tripathi, “VB: Visibility Benchmark for Visibility and Perspective Reasoning in Images,” arXiv:2603.06680, 2026.

[3] W. Rudman et al., “Mechanisms of Prompt-Induced Hallucination in Vision–Language Models,” ACL, 2026.

[4] H. Jiang et al., “MedVIGIL: Evaluating Trustworthy Medical VLMs Under Broken Visual Evidence,” arXiv:2605.07919, 2026.

[5] R. Xiao et al., “FINER: MLLMs Hallucinate under Fine-grained Negative Queries,” arXiv:2603.17662, 2026.

[6] “To Agree or To Be Right? The Grounding-Sycophancy Tradeoff in Medical Vision-Language Models,” arXiv:2603.22623, 2026.

[7] Z. Jiang et al., “MedVR: Annotation-Free Medical Visual Reasoning via Agentic Reinforcement Learning,” ICLR, 2026.

[8] “Anatomy-VLM: A Fine-grained Vision-Language Model for Medical Interpretation,” WACV, 2026.

# 下一轮 problem-first 机制发现：在连续负结果之后还剩什么

> **2026-08-02 构造审计更正：CFED 已 REJECT，不得进入 GPU 队列。**
> 全量 VinDr DICOM 与 bbox 审计推翻了本文第 3、6、7 节所依赖的“标签现成”
> 和“水平翻转有唯一 patient-side 真值”前提。旧文字仅保留为决策轨迹，不能
> 继续作为实验授权。正式审计与可行 pivot 见
> [`CFED_FATAL_CONSTRUCT_AUDIT_20260802.md`](CFED_FATAL_CONSTRUCT_AUDIT_20260802.md)。
> 当前顺序删除 CFED；AR-SoS 仍只保留单次低成本 screen。

**冻结日期：** 2026-08-02  
**范围：** 不复活 Two-Plane、reader residual、source-domain style、spatial
specificity、RAG、localized evidence survival、Clinical Presupposition 或
template-collapse 主线；只保留有本地/文献现象、现成真值、便宜反证和 OE
无遗漏路径的机制。  
**历史结论（已由顶部构造审计撤销）：** 广泛碰撞搜索后只保留两个新候选。第一名 **Clinical Frame
Equivariance Defect (CFED)** 值得立即插到 GPU 队列最前，先做一个不需人工
标注的 admission screen；第二名 **Autoregressive Satisfaction of Search
(AR-SoS)** 只值得做低成本 teacher-forcing screen。二者都不应在首个因果
gate 之前取代 Specificity Ratchet；CFED 可以暂时先于被人工门控的 CECD，
AR-SoS 不能。

## 1. 冻结的研究问题

本轮回答三个问题：

1. 在“视觉证据总量、早晚层清晰度、reader disagreement”均没有形成统一
   机制后，是否存在医学报告独有、比 generic grounding 更小的错误计算？
2. 该计算能否用现有 VinDr/VQA-RAD 真值区分于语言先验、边际置信度、输出
   长度、通用视觉遗忘和数据共现？
3. 若机制成立，能否只修改错误 claim 的一个坐标或 claim 间的因果耦合，
   而不通过少说、拒答、统一 hedge 或删除 claim 获益？

搜索角度覆盖：2025--2026 医学 VLM hallucination、RRG causal bias、
laterality/reference-frame、attribute binding、inattentional blindness、
autoregressive prefix dynamics、negation/absence、normal-image failure、
grounded report generation，以及最新 CVPR/ICLR/ACL/MIDL/ML4H/arXiv 工作。

## 2. 本地事实重新解释

本地最重要的新线索不是“模型又对图像不敏感”，而是**模型把一个空间坐标
和一个时间状态写进了临床 claim**：

- Huatuo 的 200 个 outcome-blind VinDr existential outputs 中，前 12 个
  token 的最大模板是 `right-sided pleural effusion`，覆盖 `124/200`
  图像；完整相同的 right-effusion 报告仍覆盖 `58/200` 图像。该事实见
  `corrected_runs/vindr_v2/clinical_template_attractor_diagnostic_v1/summary.json`。
  Template Collapse 已经碰撞“重复模板”本身，但**为什么固定为 right**仍是
  一个 reference-frame 问题。
- VQA-RAD native OE 中存在不依赖 judge 的显式 laterality 反例：
  `vqa-rad-test-0006` 和 `0007` 的 gold 均为 `left`，Huatuo 却明确回答
  patient `right`；`vqa-rad-test-0049` 的 gold 是 `mid left subclavian
  vein`，回答转为 `right upper lobe`。这些只证明现象存在，不估计错误率。
- 同一 200-case 诊断中，12-token prefix 的 top-1 concentration 为
  `0.62`，而完整 exact report 的 top-1 concentration 为 `0.29`。这说明
  首个临床 claim 很早就锁定后续回答分布，但**尚未证明**它因果抑制其他
  真 finding。
- localized evidence-survival 中 lesion ablation 比等量 background
  ablation 强，说明 manipulation 确实触及视觉证据；失败的是
  `3/3 > 2/3 survival` 预测，而不是“模型从不使用 lesion”。因此后续问题
  应研究 evidence 与 claim 的**绑定/竞争规则**，不再研究 reader clarity
  的总量。

## 3. Ranked candidate 1 — Clinical Frame Equivariance Defect (CFED)

### 3.1 起始异常现象

局部数据同时出现强 right-side 默认模板和明确 left→right 回答错误。外部
行为证据也表明 laterality 不是小问题：MIDL 2026 的 30-model stress test
报告医学与通用 VLM 在临床照片上出现 side-label bias、近随机和不稳定空间
grounding；其 9-model pilot 的 aggregate scorable accuracy 为 42.6%。
([Liu et al., MIDL 2026](https://openreview.net/forum?id=qejEHZL4pH&noteId=BJbRK0c234))

### 3.2 最小机制命题

> 模型可以保留 finding identity 和病灶所在 pixel hemifield，却没有把
> hemifield 与 DICOM patient reference frame 组合成稳定的 laterality；
> decoder 因而用一个固定 side prior 补全未绑定的坐标。

该命题不是“空间能力差”。它要求一个精确的群等变性：对水平反射操作
`g`，lung finding/polarity 应保持不变，而 laterality margin 应反号：

\[
f_c(gx)=f_c(x),\qquad s_c(gx)=-s_c(x).
\]

定义 frame defect：

\[
D_{frame}(x,c)=\lvert s_c(x)+s_c(gx)\rvert,
\]

并把 finding identity 的不变性作为 admission，而不是把所有 flip 当作
“临床等价图像”。机制成立的核心是 **identity survives, side fails to
transform**。

### 3.3 与 closest works 的不可替代差异

| Closest work | 已覆盖 | CFED 剩余的不可替代差异 |
|---|---|---|
| [Laterality Failure, MIDL 2026](https://openreview.net/forum?id=qejEHZL4pH&noteId=BJbRK0c234) | 临床照片上的行为 stress test、prompt/abstention controls、reference-frame failure 定性结论 | radiograph 上的 finding-preserving/side-swapping 等变分解；layerwise conjunction；只交换 side 而不改变 finding/polarity 的 causal patch |
| [ARO](https://arxiv.org/abs/2210.01936) | 一般 VLM 的 attribute/relation/order compositionality | DICOM reference frame、患者侧别、独立 radiologist truth 和临床 OE side-only repair 均未覆盖 |
| [MedVIGIL](https://arxiv.org/abs/2605.07919) | laterality-flip text probes | 没有 counterfactual image，也没有内部 frame-binding test；本地 substrate 审计还发现 option-position 与 split 问题 |
| [CounterVHD](https://arxiv.org/abs/2606.28520) | medical entity grounding 与 counterfactual uncertainty | 检测实体是否有 grounding，不检验 finding 已 grounded 但 reference frame 未绑定 |
| [FINER, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_FINER_MLLMs_Hallucinate_under_Fine-grained_Negative_Queries_CVPR_2026_paper.html) | fine-grained negative-query hallucination | query-side subtle mismatch，不是 output-side coordinate equivariance |

检索到的是相邻行为问题，没有检索到同时匹配 `CXR + exact flip
equivariance + finding/side decomposition + selective causal patch + fixed-K OE`
的工作。不能写“first”；首轮结果仍可能证明 MIDL 结论已足够解释一切。

### 3.4 可用现成数据与模型

- `/workspace/vinbigdata/train.csv` 有可审计 bbox：Nodule/Mass `2,580`、
  Lung Opacity `2,483`、Pleural effusion `2,476`、Consolidation `556`、
  Pneumothorax `226` boxes；足够先筛 unilateral lung findings。
- VinDr image labels 有三位独立 `rad_ID`，reader manifest 已按 image 切分；
  bbox 提供 pixel hemifield，DICOM orientation/marker 只用于 patient-side
  mapping 和 exclusion。
- Huatuo、Hulu、LLaVA-Med 的 hook 与 DICOM renderer 已存在。主 gate 先用
  Huatuo + Hulu 两种架构；LLaVA-Med 只做确认。

不需要新医生标签。若 DICOM orientation 无法可靠映射到 patient side，研究
单位自动降为 `image-hemifield binding`；不得靠常规展示习惯猜标签。

### 3.5 最便宜 decisive experiment

1. outcome-blind 选择 4 个 bbox finding，每个 50 个单侧、单主病灶、
   orientation 可审计的 test images；开发集独立。
2. 对 native 与水平反射图像，用完全相同的 claim template 计算：
   finding-present margin、left-vs-right margin，以及 projector/decoder
   architecture-relative layers 的两个 readout。
3. 先做 behavior gate：finding margin 在 flip 后保持，side margin 应反号；
   只有 `identity invariance` 通过的 case 可进入 frame test。
4. 在错误 case 上 patch `native↔flip` 的 side residual，并投影掉 finding
   direction；恢复 activation norm。若 patch 只交换 side token 而不改变
   finding/polarity，才是因果证据。

**最便宜成本：** 200 images × 2 views × 2 models 的 teacher-forced forward，
加小规模 patch；约 `2–4 GPU-hours total`，无需自由生成或医生 review。

### 3.6 因果控制

- identity transform 与 double-flip 必须数值闭环；
- 图像边缘/L-R marker mask 前后复现，排除文字 marker shortcut；
- 排除 dextrocardia、situs、明显非对称 device 和 orientation 不确定病例；
- same-finding/same-side image swap、side-balanced sampling、left/right token
  frequency、bbox size、view position 和 clean margin controls；
- vertical flip/brightness 是 nuisance controls，不作为临床 counterfactual；
- text-only left/right swap、随机同维方向、norm-matched patch、temperature
  scaling；
- image-cluster bootstrap，所有阈值在 dev 冻结。

### 3.7 自然方法与 OE 无遗漏路径

方法应是一个很小的 **Frame-Equivariant Projection**：把 native/flip paired
laterality score 投影到反对称分量，只替换 claim 的 anatomy-side attribute：

\[
\hat s_c(x)=\tfrac12\{s_c(x)-s_c(gx)\}.
\]

OE 中先生成原草稿，再只对含 laterality 的 atomic claim 做 side projection。
finding identity、polarity、certainty、positive claim 数 `K`、报告长度和其他
claims 全部冻结；因此 omission 在构造上不能增加，且该方法不能靠删 claim
降低 hallucination。论文 claim 只限 wrong-location hallucination，不冒充
通用医学幻觉解法。

### 3.8 Kill gate

立即终止 CFED，只要任一项成立：

- 两个模型中任一个没有先通过 finding-identity flip invariance；
- side error 可被 marker、view、clean margin 或固定 `right` token prior 完全
  解释，等变 defect 不增加 held-out side-error AUROC 至少 `0.05`；
- 少于 3/4 findings 在两个模型上同向，cluster-bootstrap 95% CI 包含 0；
- causal patch 交换 side 时，clear finding identity/polarity 改变超过 `1%`；
- OE fixed-K wrong-side error 相对下降不足 `20%`，或任何 omission/claim-count
  增加。

### 3.9 Reviewer verdict

**GO for admission; not yet paper-mainline.** 重要性高、机制极简、标签现成、
反证便宜。最大风险是 MIDL 2026 已占据“laterality reference-frame failure”
叙事；只有内部等变分解和 side-only causal correction 都成立，才足以从行为
短文跃迁为 ICLR 机制论文。单独聚焦胸片侧别仍不够 Oral，后续至少需在
另一成对器官/模态复现同一群等变规律。

## 4. Ranked candidate 2 — Autoregressive Satisfaction of Search (AR-SoS)

### 4.1 起始异常现象

本地 12-token clinical prefix 比完整报告更早、更集中地坍缩，说明首 claim
可能成为后续生成的内生条件。外部两条独立证据给出问题两侧：

- [HalluCXR](https://arxiv.org/abs/2605.20469) 在 856 个分层 MIMIC-CXR、
  15,408 次评估中报告 normal images 的严重 hallucination、common finding
  over-fabrication 与 rare finding under-detection；
- [The Inattentional Gap](https://arxiv.org/abs/2606.26529) 发现窄任务指令会
  抑制模型本可报告的共存安全信号，并用独立 open-ended critic 恢复遗漏。

这些并不证明 AR-SoS。它们只支持问一个更窄的问题：**不是 user task，而是
模型自己刚生成的第一条 finding，是否成为下一条 finding 的 task capture？**

### 4.2 最小机制命题

> 第一条 clinical claim 写入 decoder KV cache 后，作为内生 scope state
> 降低独立共存真 finding 的 image-causal margin，同时提高与第一条 claim
> 有语言共现但图像不支持的 completion；因此 omission 与 fabrication 是
> 同一个 prefix-induced search-closure 的两面。

关键量不是 token position 上的一般视觉衰减，而是同一位置、同一长度下，
supported clinical prefix 相对于 neutral/nonclinical prefix 对第二条 claim
造成的**有方向的双效应**：真 B 被抑制，假 C 被放大。

### 4.3 与 closest works 的不可替代差异

| Closest work | 已覆盖 | AR-SoS 剩余差异 |
|---|---|---|
| [The Inattentional Gap](https://arxiv.org/abs/2606.26529) | 外生 focused/exclusive task instruction 抑制未被询问信号；external critic 可恢复 | 同一开放任务内由模型自己生成的 claim prefix 造成 search closure；prefix-state causal patch |
| [Rethinking RRG via Causal Counterfactual Augmentation](https://arxiv.org/abs/2311.13307) | 数据共现的 Joint Vision Coupling 与 Conditional Sequential Coupling，训练时 counterfactual augmentation | image-fixed、truth-controlled prefix intervention；区分训练共现与运行时 KV causal suppression |
| [The Hidden Life of Tokens, ICML 2025](https://proceedings.mlr.press/v267/li25ca.html) | 一般视觉信息随层/生成下降、early excitation 与 VISTA | 不主张普遍早层；要求某一 supported clinical prefix 对独立 B/C 的选择性双效应 |
| [Template Collapse](https://arxiv.org/abs/2605.30984) | generic/normal template、rare-finding survival、what-to-say/how-to-say decomposition | 重复率不是 endpoint；同一 image/task/position 下的 prefix causal intervention 才是 endpoint |
| [Contextual Entropy Calibration, ICMR 2026](https://doi.org/10.1145/3805622.3810847) | generated-text reliance 与 visual attention balance | 不以 attention/entropy 定义机制；需 truth-controlled co-finding suppression 与 false co-completion 同时出现 |

该方向碰撞风险明显高于 CFED。若仅得到“越写越不看图”“首句决定模板”或
“共现病种更易一起出现”，都属于已有工作，必须判失败。

### 4.4 可用现成数据与模型

- VinDr 三 reader votes 可无医生新增标注地构造 `A=3/3, B=3/3,
  C=0/3` 三元组；八 finding ontology 足够先筛至少 4 个 A→B 组合。
- reader manifest 已有 image-disjoint split，bbox 可作为 A/B salience 的
  secondary covariate，不作为真值。
- Huatuo/Hulu hook 支持 teacher forcing、per-layer state 和 image-null/swap
  control。现有 OE 输出只用于确定自然 claim wording，不能定义 B/C 真值。

### 4.5 最便宜 decisive experiment

对 6 个 A-B pairs、每对 40 个 A/B 均 3/3 的 images，在同一 B token position
teacher-force 四种等长 prefix：

1. supported A；
2. 同图 0/3 unsupported A'；
3. token/frequency-matched neutral radiology sentence；
4. same-support other-image A。

同时选择一个与 A 训练共现高、但该图为 0/3 的 C。测 B 与 C 的 signed
margin、original-vs-same-support-image-swap causal effect，并在 prefix-end
做 state patch。

机制的独特预测是：相对 neutral，supported A 让 **B 的 image-causal margin
下降且 C 的 language-only margin上升**；把 A-prefix state patch 到 neutral
run 应转移两种变化。只出现一个方向不够。

**最便宜成本：** `240 images × 4 prefixes × 2 models` teacher-forced
forwards，加局部 patch，约 `4–8 GPU-hours total`；不先做自由生成。

### 4.6 因果控制

- exact generated-token position、prefix token count 与 visible length matching；
- A/B/C prevalence、pairwise co-occurrence、token frequency、claim order、
  clean margin 与 bbox area；
- A↔B order reversal，非临床同长度 prefix，语义等价 A paraphrase；
- same-support image swap、text-only run、随机 prefix/state、norm-matched patch；
- 检验效应是否只是所有 token 随 position 同比下降；
- image-cluster bootstrap，pair leave-one-out generalization；
- attention 只能作描述，不能定义机制。

### 4.7 自然方法与 OE 无遗漏路径

若机制成立，方法不是再做一个 generic critic，而是 **Claim-Boundary State
Reset**：在每个 atomic clinical claim 边界，只投影掉上一 claim 相对等长
neutral prefix 造成的 search-closure residual，保留图像、已生成表面文本和
非临床语法状态。它应同时恢复真 B 并压低假 C。

正式 OE 采用固定 `K` paired decoding：原草稿与 reset 版本都生成相同数量
positive claims；不得截断、拒答或增加统一 uncertainty。主指标同时要求
fabrication 降低与 supported-finding recall 不降。若 reset 只是多生成内容，
则在 matched-K 下不会获益；若只少说，直接失败。

### 4.8 Kill gate

- 两模型、至少 4/6 pairs 未同时出现 `B suppression + C amplification`；
- grouped-CV 中 prefix effect 相对 position、length、co-occurrence、frequency
  和 clean margin的增量 AUROC < `0.03`，或 bootstrap CI 包含 0；
- same-support other-image A 与 supported same-image A 等效，说明只是语言
  prefix prior；
- patch 不能选择性恢复 B/压低 C，或改变第一条 claim identity/polarity超过
  `1%`；
- OE fixed-K hallucination 相对下降 < `20%`，或 omission/rare-finding recall
  变差；
- 结果可完全由 Template Collapse、Inattentional Gap 或 2023 Conditional
  Sequential Coupling 的已有量解释。

### 4.9 Reviewer verdict

**CONDITIONAL GO for one admission screen only.** 这个问题的医学解释很强：
它是 machine analogue of radiological satisfaction of search，并可能统一
fabrication 与 omission。但 collision 风险和方法复杂度均高于 CFED；首轮
任何单边效应都不足以继续。

## 5. 被直接淘汰的分支

| 分支 | 淘汰原因 |
|---|---|
| Normal-image/template attractor | 本地模式真实，但 [Template Collapse](https://arxiv.org/abs/2605.30984) 已直接覆盖 normal-template bias、rare-finding survival 和 decoupled generation；Pensieve 又覆盖 same-context subtraction。 |
| Prevalence/co-occurrence calibration | [HalluCXR](https://arxiv.org/abs/2605.20469) 已报告 common over-fabrication/rare under-detection；[causal RRG](https://arxiv.org/abs/2311.13307) 已把 disease co-occurrence 写成 Joint/Conditional Sequential Coupling。仅换 VinDr 或做 calibration 是 cosmetic delta。 |
| Generic visual-information decay / prefix attention | [Hidden Life](https://proceedings.mlr.press/v267/li25ca.html)、Contextual Entropy Calibration、Hallucination Backtracking 与本地 failed localized-survival 已把这一叙事挤满；AR-SoS 仅因有 truth-controlled 双向 prediction 才暂留。 |
| Negation/absence direction | [Bi-MCQ](https://arxiv.org/abs/2601.22696)、[NegHalu](https://aclanthology.org/2025.emnlp-main.684/) 与 FINER 已覆盖 negation alignment/negative-query failure；“coverage-bounded negative”在 full-image OE 上很可能只增加 hedge，不能自然降低 positive fabrication。 |
| ROI/grounding verifier | [CounterVHD](https://arxiv.org/abs/2606.28520)、[phrase-grounded fact checking, MICCAI 2025](https://papers.miccai.org/miccai-2025/0693-Paper3526.html) 与本地 spatial/localized negative results 共同使普通 ROI score 无新机制空间。 |
| Small-lesion resolution/token compression | 2025--2026 small-patch、medical feature magnification、Med-VCD/SeeMe 已很密集；且本地没有“sub-token size phase transition”的预注册现象，暂不立项。 |
| Evidence-source erasure | 本地 MedVIGIL audit 已证明缺少同 claim source labels/source-preserving pairs；需要不可得标签，维持 halt。 |

## 6. 优先级决定

### 是否替换 CECD？

**执行优先级上：是，CFED admission 应先跑；科学主线地位上：暂时否。**
CFED 不需两位医生先确认 render equivalence，也不依赖空白 physician sheet，
2--4 GPU-hours 即可被杀死。CECD 仍有更宽的 cross-modal composition claim，
但 construct validity 和 Treble collision 风险更高。建议顺序：

1. CFED 两模型 200-case admission；
2. 若 fail，立即回 CECD，不做 CFED 修阈值；
3. 若 pass，再用 selective patch 决定它是否真正升为主线。

AR-SoS 不值得先于 CECD，因为它必须越过三项 close collision 才能生存。

### 是否替换 Specificity Ratchet？

**否。** Specificity Ratchet 仍是最直接连接 OE hallucination、临床 claim
hierarchy 和不删 claim mitigation 的候选，且 runtime 已准备好。CFED 的优势
是无需人工标签、机制更干净，但 endpoint 仅是 wrong-side attribute；它应在
等待医生 adjudication 时并行完成，而非替代 Specificity。AR-SoS 只有通过
双向因果 gate 后才有资格与 Specificity 比较。

### 当前推荐总序

1. **CFED admission（立即、便宜、无人工阻塞）**；
2. **Specificity Ratchet（人工标签到位即恢复）**；
3. **CECD（clinical-equivalence admission 到位后）**；
4. **AR-SoS 单次 screen（资源空档；不允许救阈值）**。

## 7. Oral 标准审美判断

CFED 的高点不是“做一个 laterality benchmark”，而是提出一个可交换图：
**finding identity 对 reflection 不变，patient-side attribute 对 reflection
反变，而医学 VLM 在两者的 conjunction 处破坏群等变性。** 这允许 side-only
causal repair，天然固定 claim count 和 omission。若跨至少两种 anatomy/
modality 复现，它具有 ICLR 主会乃至 Oral 所需的简洁机制形态；若只在 Huatuo
胸片 right-effusion template 上成立，则只是一个可靠诊断短文。

AR-SoS 的高点是把 radiology 的 satisfaction-of-search 变为一个可操纵的
autoregressive state，而不是心理学类比。它必须同时解释“遗漏真 B”和“补全
假 C”，否则没有统一性，也不值得继续。

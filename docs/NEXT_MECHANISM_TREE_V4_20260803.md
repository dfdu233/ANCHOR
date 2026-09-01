# 下一机制树 V4：在 Evidence-Set Closure 之后停止“候选通胀”

> 冻结日期：2026-08-03 UTC  
> 范围：医学 VLM 的 OE / report hallucination；outcome-blind、CPU-only；不修改共享评测。  
> 最终决定：**ALL-NO-GO。当前没有一个新候选同时满足全部硬门槛；不授权 GPU。**

## 0. 结论先行

本轮不是没有找到有趣现象，而是没有找到一个同时具备下列六项的、可立即执行的论文主线：

1. 单一自然变量；
2. 独立于被测模型的临床真值；
3. 本地或公开可取得的充足 substrate；
4. 一天内可以得到 GO / NO-GO 的 kill gate；
5. 可做只改变该变量的 exact-parent / causal intervention；
6. 在固定 claim count、coverage 和长度后仍有机制级 novelty。

三个最接近的分支分别死在不同位置：

| 近候选 | 真实现象 | 当前结论 | 死因性质 |
|---|---|---|---|
| Correction-Shadow Supervision（CSS） | 临床 addendum 会保留原始错误并追加最终修订 | **NO-GO，但机制未被证伪** | 公开/本地 substrate 不可识别：没有 `DICOM + 原报告 + addendum + final claim truth` |
| Synthetic-Lineage Error Multiplicity（SLEM） | PubMedVision 从同一 human caption 生成多个 synthetic descendants | **NO-GO** | absence-from-caption 不是视觉反证；且 synthetic rewrite degradation / low-hallucination recaptioning 已很拥挤 |
| Acquisition-Adequacy Conditionality（AAC） | 低质量、错误 view、低吸气或旋转会改变可判定性 | **NO-GO，直接碰撞** | 2026 LRRG 已直接把低质量 CXR report generation 定义为问题并给出方法；CXReasonBench 已含质量任务 |

这三个判定不能被平均成一个正向故事。尤其：

- **机制被实验否定**：reader two-plane、ASCC 等已有结果属于这一类；更多数据不能自动挽救其原主张。
- **机制尚未可识别**：CSS 与 SISC 属于这一类；当前不能说机制无效，只能说现有公开 substrate 不允许可信检验。
- **现象成立但 novelty 被占据**：synthetic hedging collapse、低质量输入 RRG、metric gauge 原版属于这一类；扩大数据不能恢复主线 novelty。

因此当前最优研究动作不是再给旧变量换名，而是停止 GPU，并把下一步变成有明确价值排序的数据采集决策。

## 1. 硬约束与不可复活分支

本轮把以下结论当作搜索边界，而不是待调阈值的 baseline：

- SISC：本地 694 image rows 只有 44 个 multi-image studies，且没有独立 claim–view `visible/refuted/unassessable` 真值；共享 study report 不能定义 per-image truth。
- SSEP：Hulu 有 254 个 parser candidates，但同一 native report contract 下 LLaVA-Med 为 0；不能用 reference 或另一任务补第二模型。
- reader mixture / chimera：VinDr reader sets 不是独立完整报告，union 不等于 clinical error；没有被测模型的自然 multi-reader-report training exposure。
- Evidence-Set Closure：缺少两个 source families 的同一 claim 四格真值，且 CXR-ReDonE、Pragmatic RRG、LLM-RG4、ReMIND、FactCheXcker 已分别占据 typed input-output mismatch。
- Metric Gauge 原版：MedVision `scaledPS` 已几乎同构地做 same pixels + changed spacing，FactCheXcker 已做 deterministic coordinate-to-unit correction。
- reader two-plane / ASCC：跨模型机制 gate 已失败；不能把 uncertainty、source 或 lexical operator 换名后重做。
- crop、DICOM render、style/DG、generic source shortcut、prior comparison、generic confidence/entropy/VCD、activation steering、claim deletion/shortening、generic spatial grounding：均已失败、碰撞或被限定为 baseline。

本轮曾量化 square center-crop 的真实效应：VinDr 36,096 个 positive boxes 中 908（2.52%）会被触及，637 个保留面积低于 90%，85 个完全移出；但三套主模型当前均为 pad/dynamic resolution，且旧树已明确禁止 crop 复活。因此这些数只作为审计，不进入候选排序。

DICOM intensity/HU/SUV 也不进入候选：旧 Metric Gauge 报告已把 HU、SUV、ADC 明确列作 gauge family extension；此时改用“强度单位”只是扩大原碰撞分支，不是新机制。

## 2. 研究构造路径

本轮沿 mechanism-research rubric 使用四个构造范式，但只迁移问题构造方式：

| exemplar | 分析性构造路径 | 本轮约束 |
|---|---|---|
| ViT | 重新定义基本单位 | 检查 report 的合法训练单位是否是 final clinical state，而不是所有历史文本的拼接 |
| SigLIP | 删除目标中的偶然耦合 | 检查原始错误 claim 是否仅因 PACS addendum 保存策略而继续承受 loss |
| Chinchilla | 找正确的资源比 | synthetic descendants 应按独立 source groups 而非生成条数计量，但必须先有 truth |
| Model Collapse | 把数据生成链变成可控动力学 | 比较一个 human source 派生一个或多个 correlated synthetic targets 时错误如何被放大 |

这些路径没有把任何 near-candidate 自动变成论文。硬门槛仍优先于审美分数。

## 3. Near-candidate 1：Correction-Shadow Supervision（CSS）

### 3.1 自然变量与机制

PACS 中的 addendum 通常不会物理删除原报告，而是让原始文本和修订文本共同可见。临床最终状态可能是：

\[
y^{(0)}_c=\text{wrong},\qquad y^{(1)}_c=\text{corrected}.
\]

若训练导出把两段串成单一 target，teacher forcing 会同时奖励 superseded claim 和 corrected claim。唯一自然变量是：**一个已被最终临床签名推翻的 claim 是否仍进入训练 loss**。

这不同于普通 random label noise：旧 claim 后面带有明确的 correction relation，且错误类型具有 laterality、location、observation、interpretation、template 等临床结构。若 learner 没有编译 final state，它可能形成“correction shadow”——在当前图像上继续生成已被撤销的旧 claim。

独特预测：在固定图像、最终 claims、token budget 和 claim slots 后，`raw-original+addendum` child 对 superseded old claim 的 emission 应显著高于：

- 只保留 final state 的 `compiled-final` child；
- 同剂量、同 finding marginal 的 random label-noise child；
- 将原错误 claim 换成等长非临床文字的 discourse control。

若错误只随普通 noise dose 增长，或 correction marker 消除后仍相同，CSS 被否定。

### 3.2 真实临床 grounding

[Patra et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8448237/) 审计 97,003 份 cross-sectional radiology reports，发现 1,076 份 addenda（1.1%）；其中 767（71.2%）属于 errors，190（17.6% of addenda）为 clinically significant，224 为 observational errors，191 为 interpretation errors。该工作还明确说明原报告和 addendum 会共同保留。

因此 phenomenon 真实、重要，而且不是我们人为制造的 style perturbation。

### 3.3 本地与公开 substrate 审计

本地 `data/mmedrag/test/report/mimic_test.json` 共 700 rows。对每份 `report` 搜索冻结词表：

```text
addendum | correction | corrected report | amended |
please disregard | should read | typographical | dictation error
```

命中为 **0/700**。这不是 addenda 在临床中不存在，而是本地评测子集不携带 revision lineage。

最接近的公开资源仍不合格：

- [RadRevise, 2025](https://proceedings.mlr.press/v281/huang25a.html) 有 6,402 editing instructions 和 2,922 modified reports，但它是 report-editing benchmark，不发布与自然 preliminary/final revisions 绑定的影像和临床 revision provenance。
- [Radiology Reporting Errors, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8448237/) 给出 1,076 个真实 addenda 的统计和类型，但不公开可训练的逐例 DICOM、原报告和 final truth。
- [Image-Conditioned Autocorrection in Medical Reporting](https://medautocorrect.cs.columbia.edu/health-autocorrect.pdf) 从 intentionally injected errors 出发；它不能证明真实 PACS revision shadow 污染了 VLM training。
- ReXVal / ReFiSco 提供 generated-report 的专家 error ratings，不是自然报告修订 lineage。

检索未找到同时公开 `same DICOM + original signed report + addendum/final report + proposition-level correction`，并研究其对 image-to-report learner 的 causal imprint 的工作。这个措辞只表示“在记录的检索中未找到”，不证明绝对 novelty。

### 3.4 Exact-parent 设计（仅在数据到位后）

同一公开 parent、optimizer、steps、seed family、训练 token 数和 ontology slots：

| child | 图像 | claim slots | target construction |
|---|---|---|---|
| `COMPILED-FINAL` | 相同 | 相同 | 每个 corrected proposition 只保留 final state |
| `SHADOWED` | 相同 | 相同 | superseded old state 与 final state 都承受 loss |
| `RANDOM-NOISE` | 相同 | 相同 | 同剂量、同 finding/edit marginal，但错误不与 correction lineage 对齐 |
| `DISCOURSE-CONTROL` | 相同 | 相同 | old claim 替换成等长非临床 correction prose |

正式读数只能是 superseded-claim emission、final-claim accuracy、contradiction rate 和固定 coverage 下的 OE precision/recall。不得删除 slot、缩短报告或统一 hedge。

### 3.5 Fatal decision

CSS 通过 grounded phenomenon、falsifiability 和初步 collision gate，但**没有可取得的充足识别 substrate**。当前不能运行 exact-parent child；不能用 synthetic RadRevise edits 冒充自然 PACS revision；不能用 LLM judge 生成 final truth。

**决定：NO-GO because unidentifiable, not because mechanism was falsified.**

## 4. Near-candidate 2：Synthetic-Lineage Error Multiplicity（SLEM）

### 4.1 机制候选

医学 VLM instruction data 常从一个 human image-caption source 生成多个 QA descendants。若 descendants 共享同一 unsupported clinical claim，把它们当作独立样本会增加该错误的 effective weight；增加的不是独立信息，而是同一 teacher error 的 multiplicity。

单一变量是每个 human source group 的 synthetic descendant multiplicity (m_g)，而不是笼统 synthetic-data proportion。

一个可区分 random noise 的预测是：在总 tokens、source groups、claim marginals 和图像完全匹配时，shared-correlated error 的 downstream imprint 随 (m_g) 增长，应大于把相同错误数量均匀分散到独立 source groups 的结果。

### 4.2 本地真实计数

本地 PubMedVision 包含：

| artifact | rows |
|---|---:|
| human `Original_Caption` | 636,092 |
| `Alignment_VQA` | 646,759 |
| `InstructionTuning_VQA` | 646,759 |

按冻结 uncertainty/differential regex，53,354 个 human captions 至少含一个候选 operator。配对审计得到：

| transformation | source-operator rows | output 丢失全部冻结 operators | 丢失后同时出现直接 assertive frame |
|---|---:|---:|---:|
| Alignment | 53,354 | 19,304 | 12,477 |
| Instruction | 53,354 | 20,910 | 6,211 |

Alignment 的 operator-loss rate 为 36.18%；其中 64.64% 同时命中 `image shows/reveals/demonstrates/there is` 等直接断言 frame。

这些是真实字符串计数，不是 proposition-level hallucination 率。特别是 `consistent with` 在部分上下文中可表达强诊断而不是弱 hedge；caption 没提及某 claim 也绝不等于图像反驳它。

### 4.3 Fatal collision

这一分支有两个独立致命问题。

第一，2026-06 的 [The Slop Paradox](https://arxiv.org/abs/2606.17791) 已直接研究 AI rewrite 对 radiology text 的 entity erosion、hedging collapse 和 image-text alignment degradation；其 450 份 IU reports 中，标准化/teaching-case rewrites 造成 14.9–16.5% cross-modal alignment drop。把上面的 PubMedVision operator counts 包装成“synthetic certainty laundering”与其机制和主张直接相邻。

第二，SLEM 要声称 correlated descendants 放大**视觉 hallucination**，必须证明两个 descendants 共享的是相同 image-unsupported proposition。human caption 的 omission 不能充当 negative truth；模型 classifier、LLM judge 或 source caption itself 都不能完成这个 gate。

[Low-hallucination Synthetic Captions](https://arxiv.org/abs/2504.13123)、MedHallTune 和 model-collapse / repeated-data literature 又分别占据了 synthetic caption fidelity、hallucination-aware tuning 和 repeated synthetic error 的相邻机制。即使 multiplicity law 尚未被精确复现，当前 delta 也不足以抵消 truth 缺口与 Slop collision。

**决定：NO-GO；现有计数只进入 PubMedVision data card，不进入论文主线。**

## 5. Near-candidate 3：Acquisition-Adequacy Conditionality（AAC）

### 5.1 原始机制问题

临床图像有一个不同于 finding polarity 的状态：是否足以评价该 finding。低吸气、旋转、错误 projection、截断 anatomy、曝光和 artifact 都可能让一个 claim 从 supported/refuted 变成 technically limited。若 VLM 直接输出 definite report，它可能把 acquisition artifact 解释成疾病。

这有自然 metadata / quality labels，且可保持 claim slots 不变：每个 finding slot输出 `supported / refuted / technically-unassessable`。

### 5.2 公开计数与 collision

[Zhu et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12091825/) 发布的 MIMIC-CXR metadata audit 覆盖 377,100 images，报告：

- poor quality：83（0.02%）；
- wrongly labelled views：1,054（0.28%）；
- previously unlabelled views：15,769（4.18%）。

但这不足以形成新主线：

- [Radiology Report Generation for Low-Quality X-Ray Images](https://arxiv.org/abs/2604.10188) 已直接提出 LRRG benchmark、自动质量 agent 与 dual-loop training，核心问题就是低质量输入下 report generation degradation。
- [CXReasonBench](https://physionet.org/content/chexstruct-cxreasonbench/1.0.1/) 已把 inclusion、inspiration、rotation 和 projection 纳入专家设计的 CXR reasoning / quality tasks。
- 83 个 public poor-quality positives 也不足以支持多 finding、patient-disjoint、两个模型的 formal mechanism gate；其 quality 标签来自 automated pipeline 加抽样人工检查，不是每个 finding 的 independent assessability truth。

若退回 synthetic degradation，就会回到已关闭的 style/render/crop 分支；若只做 view correction，则 Zhu et al. 已直接发布 corrected metadata 并用于 cardiomegaly/report pipeline。

**决定：direct-collision NO-GO；不是可通过增加 GPU 挽救的执行问题。**

## 6. 被搜索但在展开前剪枝的分支

| 分支 | 剪枝原因 |
|---|---|
| HU/SUV/ADC intensity calibration | 旧 Metric Gauge 已明确列作同一 gauge family extension；换单位不是新机制 |
| square crop / peripheral pathology | 旧树明确禁止 crop 复活；本地主模型用 pad/dynamic；且属于 input information destruction |
| horizontal flip / laterality target corruption | REFERS 已明确在 flip 时同步交换 report 中 left/right；laterality error 与 spatial grounding 亦高度拥挤；本地 VinDr 又缺 patient-side orientation truth |
| Findings–Impression contradiction | impression generation、section consistency、RadCouncil、SRRG 已直接覆盖；final section 也不自动是 image truth |
| AP/PA projection-conditioned cardiomegaly | view metadata correction与 segmentation-CAD 已公开；缺 independent per-image clinical truth时会退化成 metric/quality branch |
| multi-frame CT/MRI/WSI bag-to-instance | 与 SISC / MIL 的数学结构相同，不能用 modality 换名复活 |
| differential `A versus B` 被生成成 `A and B` | 是已失败 SSEP 的 shared alternative scope 子例，且自然第二模型 gate 已失败 |
| preliminary/resident → attending revision | 科学上接近 CSS，但没有公开 same-image natural revision lineage；RadRevise 是 synthetic editing，不可替代 |
| body-part/modality metadata errors | 普通 structured label noise；metadata 不一定进入训练 conversation，不能解释 OE finding hallucination |

## 7. 硬门槛评分

评分只帮助解释为何停止，不能覆盖 hard-gate failure。(I/M/N/E\in[0,3])，基础分：

\[
S=0.30I+0.30M+0.20N+0.20E.
\]

| rank among near-candidates | branch | I | M | N | E | base | hard gate | final |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | CSS | 3 | 3 | 2 | 0 | 2.40 | G1/G2/G3 pass；substrate/E fail | **NO-GO** |
| 2 | SLEM | 2 | 2 | 1 | 2 | 1.90 | independent visual truth + Slop collision fail | **NO-GO** |
| 3 | AAC | 3 | 2 | 0 | 2 | 1.90 | direct collision fail | **NO-GO** |

CSS 在 impact-first、mechanism-first 中仍是 near-candidate 第一，但 E=0 是硬失败，不得用加权平均掩盖。没有候选进入 research-ops handoff。

## 8. 最小新数据采集：按预期信息增益排序

这里的排序不是当前论文方向排序，而是“哪一种新增、非模型真值最能改变当前科学判定”。

### Rank 1 — Natural revision-lineage cohort（最高信息增益）

目标：让 CSS 首次可识别。

最低字段：

```text
study_id / series_id / DICOM ids
original signed report + timestamp
every addendum/revision + timestamp + author role
final compiled report
claim_id + old state + final state + edit type
reason: observation / interpretation / laterality / location /
        transcription / new context / additional image
whether final state is supported by the original image set
```

最低规模：

- 从至少约 50,000 连续 reports 中保留全部 addenda lineage；按 Patra et al. 的 1.1% rate，期望约 550 addenda，而不是事后只挑显著病例；
- 至少 300 个 proposition-changing error pairs；
- 至少 3 个 finding/edit families，每类 50 个以上；
- patient/study-disjoint；两位 radiologists 独立判 final-state image support，分歧交第三位 adjudicate；
- 新增临床 context 导致的 revision 与原图误读分开，不能混成同一机制。

一天 kill gate（拿到 metadata 后）：若 proposition-changing image-adjudicable pairs <300，或任一主要 family <50，或大多数修订依赖新增 context/sequence，则 CSS 立即继续 NO-GO。

预期信息增益最高，因为正结果会打开一个尚未检索到机制等价 image-to-report causal study；负结果则永久剪掉 correction-shadow，而不是只得到又一个模型失败率。

### Rank 2 — Independent per-view claim truth（重开 SISC，不是新名字）

目标：解决 SISC 的识别失败，而不把共享 report 当 truth。

最低规模沿既有冻结 gate：

- 至少 100 paired-view studies；
- 至少 3 findings；
- 每 finding 至少 30 个由医生独立确认的 `visible in view A / refuted or unassessable in view B` claims；
- 同时保存完整 study report、DICOM view metadata 和 claim boxes/visibility；
- absence 与 unassessable 必须分开。

这项数据比再下载更多 MIMIC reports 有价值；现有 local 44 paired studies / 91 rows 不够，且没有 truth。它可重开 SISC，但仍需面对 multi-view RRG、MIML/MIL 的 novelty collision，所以信息增益低于 Rank 1。

### Rank 3 — Same-image independent full-reader reports + joint adjudication

目标：判定 reader-crossing composition 到底是 clinical error 还是合法 communication variation。

最低规模：

- 至少 500 images，每图 3 份互不知情的完整 reports；
- 同一 prompt / reporting obligation；
- 独立 joint adjudication 给出可共存 claim sets、互斥关系和可接受 omissions；
- 保存 reader identity 但 blind test split；
- 至少 3 findings 各有 100 个 reader-disagreement cases。

VinDr 的三套 labels/boxes 不能替代 full reports。该数据若显示 cross-reader compositions 大多临床可接受，会直接否定 chimera 叙事；即使正向，D-Persona、多标注者学习和 pragmatic reporting 的 collision 仍高，因此排第三。

### 明确不建议采集

- 不为 synthetic uncertainty laundering 再做大规模 clinician review：Slop Paradox 已占据核心 phenomenon。
- 不为 crop/HU/SUV 下载新数据：它们属于已关闭/碰撞的原分支。
- 不用更多生成结果替代 revision/per-view/reader truth；增加模型 output 不能补独立真值。

## 9. 一日 kill gate 与后续动作

当前 gate 已完成：

```json
{
  "candidate_count_passing_all_hard_gates": 0,
  "gpu_authorized": false,
  "shared_evaluation_modified": false,
  "highest_information_action": "acquire_natural_revision_lineage",
  "fallback": "retain_credible_negative_map"
}
```

允许的后续只有：

1. 将 PubMedVision operator-loss counts 写入数据质量 appendix / data card；不训练 mitigation。
2. 若能取得 institutional revision lineage，先只跑 Rank-1 metadata/claim-count gate；不先开模型。
3. 若无法取得新真值，保留本轮 zero-admissible 结果并继续搜索其它临床数据生成链；不得从 crop、HU、source、scope、reader mixture 中挑一个改名。

## 10. 检索与计数可追溯性

本轮第二遍 fatal-collision queries 覆盖：

- `radiology report generation addendum correction training data errors`
- `MIMIC-CXR report addendum corrected report dataset`
- `medical VLM synthetic instruction data uncertainty hallucination`
- `synthetic standardization hedging collapse radiology`
- `radiology report generation low-quality image hallucination`
- `chest x-ray image quality inclusion inspiration rotation projection dataset`
- `medical VLM Hounsfield unit DICOM metadata quantitative VQA`
- `radiology report findings impression contradiction generation`
- `medical image-text horizontal flip laterality report`

本地计数来源：

- `/home/dbw/data/PubMedVision/PubMedVision_Original_Caption.json`
- `/home/dbw/data/PubMedVision/PubMedVision_Alignment_VQA.json`
- `/home/dbw/data/PubMedVision/PubMedVision_InstructionTuning_VQA.json`
- `/home/dbw/ANCHOR/data/mmedrag/test/report/mimic_test.json`
- `/workspace/vinbigdata/train.csv`
- `/workspace/vinbigdata/train/*.dicom`

相关先验决策：

- `docs/SISC_OUTCOME_BLIND_TRUTH_GATE_20260803.md`
- `docs/SSEP_SCOPE_ADMISSION_GATE_20260803.md`
- `docs/READER_MIXTURE_CHIMERA_COLLISION_20260803.md`
- `docs/EVIDENCE_SET_CLOSURE_COLLISION_20260803.md`
- `docs/METRIC_GAUGE_HALLUCINATION_COLLISION_20260803.md`

## 11. 最终研究判断

当前最高 taste 的判断不是“再找一个 decoding method”，而是承认现有公开材料的识别边界：

> 我们已经有很多 image/report pairs，却缺少能区分“模型学错了”与“训练 target 本来就不是当前图像真值”的 revision、view 和 reader lineage。

reader two-plane / ASCC 是机制被结果否定；synthetic hedging / AAC 是 novelty 被文献否定；CSS / SISC 是现有 substrate 无法识别。三者必须在论文和项目管理中分开。

在取得 Rank-1 或 Rank-2 的独立真值前，继续跑 GPU 只会得到一个无法归因的 performance delta。因此 V4 的可信结论是：**zero admissible candidate、zero GPU，而不是 zero insight。**

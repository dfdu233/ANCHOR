# PPI 之后的下一机制树：从“像素—报告”错配中寻找可证伪问题

> 冻结日期：2026-08-03；本轮是 outcome-blind、CPU-only 的方向筛选，不把本地频数当作模型有效性结果，也不授权 GPU。评测框架由另一 session 维护，本文件不修改共享评测代码。

## 0. 结论先行

下一步不应再从一个 decoding trick 出发，而应检查医学报告任务的**基本监督单位和语言真值单位是否定义错了**。在已经失败或拥挤的方向之外，剩下三个可证伪候选：

1. **Study–Image Supervision Collision（SISC，首选）**：一份报告描述整个检查 study 的多张视图，但训练/评测常把它复制给每张单图；于是单图被迫学习其它视图才可见的 claim。这里的幻觉可能不是 decoder “不看图”，而是数据在训练时把不可见 claim 教成了正例。
2. **Shared-Scope Evidence Pooling（SSEP，并列高优先）**：报告把多个 finding 压缩进同一否定、选择或不确定性作用域；模型可能给词项分别找到了视觉证据，却把一个共享 operator 的状态错误复制给所有 sibling claims。
3. **Evidence-Sufficiency Type Error（ESTE，仅条件保留）**：数值、比较和病史型 claim 需要像素之外的特定证据源，但生成接口把它们与纯视觉 finding 当作同一种输出。该方向与 FactCheXcker 等碰撞严重，只有两个以上来源类型服从同一因果规律时才值得继续。

建议顺序不是先训练方法，而是先用一天 CPU/小规模人工 gate 同时淘汰 SISC 和 SSEP。SISC 若取得独立的 claim–view truth，是当前最像 ICLR oral 主线的候选；SSEP 若能证明“scope × sibling truth”的交互而非普通 negation failure，则是更简洁优雅的备选。ESTE 默认不立项。

## 1. 不可回收的失败约束

本轮明确不允许通过换名字复活以下分支：

- ASCC：观察到的是通用词汇/提示偏移，不是 reader-disagreement-selective commitment erasure。
- reader-grounded two-plane：没有形成跨模型的统一 clarity erasure。
- coverage-certified decoding：Hulu 的局部现象没有跨模型成立。
- prior titration、CECD、style/DG、source/prevalence、PPI natural bridge：或机制 gate 失败，或自然样本不足，或需要尚未通过的人工 admission。
- generic confidence、entropy、VCD、crop、style transform、source shortcut、简单 activation steering、claim 删除/缩短回答。
- specificity ratchet、宽泛的 AR lock-in/视觉衰减、宽泛的 token 顺序偏差：已有失败或最近工作直接拥挤。

因此新候选必须同时满足：有**自然可观测量**、能做**只改变一个机制变量**的因果实验、结果不能由答得更短/更保守解释、并且在一天内可以被杀死。

## 2. 从顶会论文反推的研究构造审美

这里不照搬方法模块，而只抽取四种高质量论文的构造方式：

| 构造范式 | 高水平论文做对了什么 | 对本项目的约束 |
|---|---|---|
| ViT | 重新定义基本计算单位（patch/token），再让架构服从该单位 | 先问 RRG 的样本单位究竟是 image、study 还是 evidence set，而不是先设计 decoder trick |
| SigLIP | 找出一个目标函数中的偶然耦合，删掉不必要的全局 normalization | 寻找“单图—全报告复制”或“共享 scope—多 claim”这种监督耦合，而非新增复杂模块 |
| Chinchilla | 用受控 scaling law 把经验争论变成可预测关系 | 新机制必须给出单调或交互型预测，而不只报告平均分提升 |
| Model Collapse | 用最小受控生成链揭示系统性退化，再回到真实数据验证 | 先做最小 causal child / matched construction，再上完整 OE 和报告生成 |

共同审美是：**发现任务定义中的一个隐含等号不成立**，并让修复几乎是该机制的必然结果。

## 3. 领域图与碰撞边界

截至冻结日，相关领域已明显拥挤：

- 多视图 RRG 已有 [MCL](https://arxiv.org/abs/2411.10224)、[CVPR 2025 MLRG](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Enhanced_Contrastive_Learning_with_Multi-view_Longitudinal_Data_for_Chest_X-ray_CVPR_2025_paper.html) 和更早的 [multi-view fusion](https://arxiv.org/abs/1907.09085)。它们主要论证“加入多视图会提高报告质量”，没有自动留下我们的 novelty。
- negation 已有 [CVPR 2025 NegBench](https://openaccess.thecvf.com/content/CVPR2025/html/Alhamoud_Vision-Language_Models_Do_Not_Understand_Negation_CVPR_2025_paper.html)；医学侧还有 [CXR-Align](https://physionet.org/content/cxr-align/) 和成熟的 negation-scope NLP。仅证明模型不懂 `no` 不新。
- measurement hallucination 已被 [CVPR 2025 FactCheXcker](https://openaccess.thecvf.com/content/CVPR2025/html/Heiman_FactCheXcker_Mitigating_Measurement_Hallucinations_in_Chest_X-ray_Report_Generation_Models_CVPR_2025_paper.html) 正面覆盖。
- 当前/既往证据缺失与报告清洗已有 [Pragmatic RRG](https://proceedings.mlr.press/v225/nguyen23a.html) 和 [CXR-ReDonE](https://proceedings.mlr.press/v193/ramesh22a.html)。
- phrase grounding 数据并不等于 claim–view incidence，但 [MS-CXR](https://physionet.org/content/ms-cxr/1.1.0/) 提供 1,162 个经放射科医生核验的 image–sentence box pairs，可作为 SISC 的初始独立锚点。

所以可声称的新颖性只能是下列更窄的机制差值，而不能是“multi-view / negation / metadata 有用”。

## 4. 候选一：Study–Image Supervision Collision（SISC）

### 4.1 问题与机制

临床报告的生成过程是：

\[
\{x_{s,1},\ldots,x_{s,m}\}\longrightarrow G_s,
\]

其中一个 study 的全部视图共同对应一个 claim graph。常见数据管线却把它展开成：

\[
(x_{s,1},G_s),\ldots,(x_{s,m},G_s).
\]

当 claim \(c\) 只在侧位片或某个视图可见时，展开后的其它图像也被赋予 \(c\) 的正监督。这不是普通 label noise，而是**有方向、与 view 可见性相关的 supervision collision**。它会训练模型把 study-level 共现先验当成 single-view evidence。

可检验预测：若 \(v^+(c)\) 是独立标注为可见的视图，\(v^-(c)\) 是同 study 中不可见/不可判定的视图，则 exploded supervision 会增加

\[
P(\hat c=1\mid x_{v^-}, c\in G_s),
\]

但不会相应增加该视图上的真实视觉可分性。错误应沿 study 共同出现关系迁移，而不是沿随机患者或 same-finding 图像迁移。

### 4.2 自然可观测量

- 同一 `study_id` 下的多张 DICOM、`ViewPosition` 和共享报告。
- 独立的 claim–image 可见性/grounding 标注；“没有 box”不能直接视为 absent，必须区分 absent 与 unassessable。
- 同 study 中一张图可见、另一张图不可判定的 view-exclusive claims。

MIMIC-CXR 的官方结构本身支持“一个 study 对应一份报告并含一张或多张影像”的定义；这使机制的自然变量存在，而不是人工 style transform。

### 4.3 本地 substrate（只证明能做实验，不证明机制成立）

- `data/mmedrag/test/report/mimic_test.json`：700 rows。
- 两套已完成 native report 输出各有 694 rows：
  - `corrected_runs/unified_eval/full/hulu_mimic_report_greedy_v2/predictions.jsonl`
  - `corrected_runs/unified_eval/full/llava_mimic_report_greedy_v1/predictions.jsonl`
- 694 个评测 image rows 对应 647 个 unique studies；44 个 studies 含多于一张图，共 91 个 image rows，最大 3 views。同 study 的 reference report 字节相同，而单图生成可能不同。

这只能证明本地评测确有“study target 被复制到 image rows”的结构；没有独立 claim–view truth 前，绝不能把两张图回答不同称为错误。

### 4.4 最小因果实验

**阶段 A：自然 gate（无训练）。**

1. 用 MIMIC metadata 取得 view/study mapping；以 MS-CXR、Chest ImaGenome 或新增医生标注建立 claim–view incidence。
2. 冻结至少 100 个 paired-view studies、至少 3 个 findings，每个 finding 至少 30 个 view-exclusive 或 view-unassessable claims。
3. 对 frozen VLM 比较单视图、同 study 全视图和错误配对视图；控制图像数、分辨率与 prompt。
4. 主读数不是平均 RadGraph，而是：wrong-view emission、right-view recall、same-study crossover，以及 matched claim count 下的 precision/recall。

**阶段 B：exact-parent causal children。**

相同初始化、tokens、steps、seed family，只改变监督映射：

- `Exploded`：每张单图配完整 study report；
- `Incidence-aware`：每张图只配独立确认在该图可验证的 claims，并将其余 claim 显式标成 `unassessable-from-this-view`；
- `Study-set`：全部可用视图配完整报告。

关键 prediction 是 `Exploded` child 在错误视图上系统性复制 sibling-view claim，而 `Incidence-aware` 降低 wrong-view emission，同时 right-view recall 下降不超过 1pp。`Study-set` 用于验证信息确实可由完整 evidence set 恢复。输出长度、positive claim 数和 ontology coverage 必须匹配。

### 4.5 若成立，方法为何自然

方法不是“再加一个 multi-view encoder”，而是 **Evidence-Set–Claim Incidence Training**：

- 样本主键从 image 改为 study evidence set；
- 每个 claim 保存 `supported-by-view-set / refuted / unassessable` incidence；
- 缺视图时不把 study-level positive 当作该视图正例；
- 有完整视图时仍生成完整报告，不靠删除 claim 获益。

论文贡献会是：揭示公开 RRG 构造中一个可识别的监督冲突；证明该冲突因果地产生 wrong-view pseudo-evidence；再给出最小的数据/目标修复。

### 4.6 致命碰撞

MCL、MLRG、MAIRA-2 和大量 dual-view RRG 已证明多视图有益。若实验最终只能说“输入两张图比一张图好”，方向立即死亡。唯一可能的新意是同时满足：

1. 有独立 claim–view truth；
2. exploded mapping 造成方向明确的 wrong-view crossover；
3. exact-parent child 只修监督 incidence 就消除该 crossover；
4. 效果不是多给像素、参数或更长输出。

### 4.7 一日 kill gate

一天内只做 manifest/annotation feasibility：

- 至少 100 个 paired-view studies；
- 至少 3 个 findings；
- 每 finding 至少 30 个被独立医生/grounding 证据确认的 view-exclusive 或 unassessable claims；
- patient/study-disjoint split 可建立；
- absence 与 unassessable 可分开。

任一项失败，或唯一可行标签仍是从共享报告反推，就 **NO-GO**，不启动 GPU。

## 5. 候选二：Shared-Scope Evidence Pooling（SSEP）

### 5.1 问题与机制

报告语言常把多个原子 finding 压进一个 operator scope：

- `No A, B, or C.`
- `Possible A or B.`
- `A but not B.`

视觉 encoder 可能为 A、B 分别保留正确证据，但 decoder 在 realization 时只维护一个 group-level polarity/uncertainty state，随后把它复制给 sibling atoms。于是问题不是不认识 finding，也不只是“不懂 no”，而是：

\[
\text{operator state} \not\leftrightarrow \text{per-claim evidence}.
\]

核心预测是一个可交互效应：在 A 与 B 真值相反时，B 的错误率应受到 sibling A 真值影响；该影响在 shared-scope realization 中显著强于两个独立句子。若只出现 negation 主效应、长度效应或位置效应，机制不成立。

### 5.2 自然可观测量

- 报告中的 coordinated negation、alternative、contrastive scope。
- 每个 sibling atom 的独立视觉真值，尤其 reader-unanimous 的 mixed-state pair（A present、B absent）。
- scope、atom order、句法位置，以及输出后每个 atom 的 polarity/uncertainty。

### 5.3 本地 substrate

对已有 694-row native report 输出做正则 smoke audit：

| corpus | negated coordination | hedged alternative | any disjunction | numeric measure | comparison |
|---|---:|---:|---:|---:|---:|
| Hulu generated | 258 | 32 | 290 | 12 | 126 |
| reference | 217 | 27 | 369 | 46 | 600 |
| LLaVA generated | 6 | 0 | 8 | 0 | 1 |

这些是宽松字符串频数，不是临床 scope labels，也不是效果结果。它们只说明 Hulu 有足够自然语言 substrate；LLaVA 输出过短，不能自动成为第二模型。第二模型 admission 必须先解决，否则该方向不能跨模型立论。

### 5.4 最小因果实验

1. 从 VinDr reader votes / MIMIC 独立 labels 选 mixed-state atom pairs，保证 A、B 都是常见 finding，且图像证据强。
2. 同一图、同一两个 atoms、同一顺序和 claim count，只改变 realization contract：
   - shared scope：一个协调结构；
   - distributive scope：两个独立子句；
   - matched nonclinical syntax：控制纯句法/长度；
   - reversed atom order：区分 scope 与 serial position。
3. 由临床人员先盲审 constructions 的自然性与语义等价性；未通过不得运行模型。此前 AR-lock-in 因不自然 continuation 失败，这里不能重复。
4. 终点是每个 atom 的 polarity、alternative membership 和 location/attribute 保持，不用整句相似度。

预注册检验：

\[
\Delta =
\big(E_{mixed}-E_{same}\big)_{shared}
-
\big(E_{mixed}-E_{same}\big)_{independent}.
\]

要求 \(\Delta>0\)、image-cluster bootstrap 95% CI 排除 0、反转顺序后效应跟随 scope 而非位置，并在至少两个模型、三个 findings 上成立。

### 5.5 若成立，方法为何自然

**Scope-Separated Claim Realization**：先生成 operator-typed claim graph，令 polarity/uncertainty 挂到每个 atom，而不是挂到整段文本，再用受约束 realization 合并为自然报告。

它必须保持：

- atom 身份不变；
- positive/negative claim 数不变；
- 输出长度 matched；
- finding coverage 不变。

并必须优于简单“全部拆成短句”、JSON/模板化生成和 generic negation finetuning，否则贡献只是工程格式化。

### 5.6 致命碰撞

NegBench 已覆盖一般 VLM negation，CXR-Align 覆盖插入式医学 negation，NegBio/临床 NLP 已长期研究 negation scope，structured RRG 也会生成 claim graphs。剩余空间仅是：**共享 scope 会把一个 sibling 的视觉真值因果地传播到另一个 sibling，并可在保持 atom 集合不变时修复**。

若错误在 independent clauses 同样严重，或 extractor 才是 scope 错误源，或只能在一个模板/一个模型上看到，立即判为已有 negation failure 的子例，不写论文。

### 5.7 一日 kill gate

- 从 native outputs/references 中抽取并人工复核至少 100 个自然 scope cases；
- 至少 3 findings、2 models；
- mixed-state siblings 有独立 truth；
- shared 与 distributive 版本均由临床审阅为自然且语义可比；
- parser 对 scope 的人工一致率达到预注册阈值（建议 ≥0.9）。

第二模型没有足够自然多 atom 输出，或无法构造不改变语义的 scope minimal pair，即 **NO-GO**。

## 6. 候选三：Evidence-Sufficiency Type Error（ESTE，默认剪枝）

### 6.1 机制

一份报告混合了不同 evidence signatures：

| claim type | 必需证据 |
|---|---|
| finding presence | 当前图像 |
| physical measurement | 当前图像 + localization + DICOM calibration |
| interval change | 当前 study + 对应 prior study |
| indication/history statement | EHR/context，而非当前像素 |

若接口只给 raster，却要求生成完整报告，某些 claim 在信息论意义上不可识别。模型只能从训练先验补值，造成“来源类型错误”，而不一定是一般 hallucination。

### 6.2 自然可观测量与本地 substrate

- DICOM `PixelSpacing`、Rows/Columns、view metadata；prior/current study identity；report 中 measurement/comparison/history spans。
- `/workspace/vinbigdata/train` 有 15,000 DICOM。对前 1,000 张只读 metadata smoke audit：1,000/1,000 有 Rows/Columns，862/1,000 有 PixelSpacing，示例 0.14–0.175 mm，观察范围 0.125–1.0。
- 本地 Hulu generation 仅 12/694 含 cm/mm 数值，reference 为 46/694；comparison 分别 126/694 与 600/694。

这些数字不能证明错误；尤其 projection radiograph 的 PixelSpacing 未必等于解剖对象的真实尺度。

### 6.3 最小因果实验

把像素严格固定，只切换正确/置换/缺失的证据源：

- measurement：正确 vs permuted PixelSpacing，且 localization 固定；
- temporal：正确 prior vs matched wrong prior；
- history：正确 indication vs matched wrong indication。

要求影响只出现在对应 claim type，finding identity/location 保持不变；错误 metadata 产生可预测的 scale ratio，而不是泛化的 prompt 效应。

### 6.4 致命碰撞与 gate

FactCheXcker 已直接研究 measurement hallucination；Pragmatic RRG、CXR-ReDonE 和 2025–2026 longitudinal RRG 已覆盖缺失 prior/context。故单一 measurement 或单一 prior 结果都不够新。

一天 gate 必须找到：

- 至少两个 source types；
- 每 type 至少 100 个自然 claims；
- 至少两个模型；
- source-specific perturbation 有清楚、方向预注册的效应；
- 一个统一的 typed-observability 机制能解释两类结果。

只剩 measurement 时以 FactCheXcker 碰撞剪枝；只剩 prior 时以前述 longitudinal/pragmatic work 碰撞剪枝。默认不投入 GPU。

## 7. 评分与稳定性

评分沿 mechanism-research rubric：Impact \(I\)、Mechanism clarity \(M\)、Novelty margin \(N\)、Execution readiness \(E\) 各 0–3，基础分

\[
S=0.30I+0.30M+0.25N+0.15E,
\]

再减 collision risk \(R\)。

| candidate | I | M | N | E | base | R | final | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SISC | 3 | 3 | 2 | 2 | 2.60 | 0.20 | **2.40** | 首选，先过 claim–view truth gate |
| SSEP | 3 | 3 | 2 | 2 | 2.60 | 0.25 | **2.35** | 并行 CPU gate；自然性和第二模型是脆弱点 |
| ESTE | 2 | 3 | 1 | 3 | 2.25 | 0.50 | **1.75** | 默认剪枝，仅统一两类 source type 才恢复 |

稳定性分析：提高 novelty 权重时 SISC 仍优于 ESTE，但 SISC 与 SSEP 的次序不稳定；提高执行权重时 SSEP 可能领先，因为本地报告已经有 scope substrate。故不能凭总分冻结主线，真正的 tie-breaker 是一天 gate 后谁取得**独立真值 + 自然干预 + 第二模型**。

## 8. 明天可直接执行的 outcome-blind 交付

### SISC gate artifacts

1. `study_view_manifest.jsonl`：study/image/view/subject，不含模型结果。
2. `claim_view_truth_candidates.jsonl`：只保留独立 grounding/医生来源及 `visible/refuted/unassessable`。
3. `sisc_feasibility.json`：每 finding 的 paired counts、truth source、split leakage audit。
4. 冻结 `GO/NO-GO` 规则后才允许打开现有模型输出。

### SSEP gate artifacts

1. `scope_candidate_spans.jsonl`：原始文本、operator、siblings、来源。
2. 双人盲审表：自然性、scope、semantic equivalence，不含模型 outcome。
3. `scope_admission.json`：每模型/每 finding 数量、parser–human agreement。
4. 只有 admission 通过才生成 minimal-pair prompts。

## 9. 推荐的论文逻辑链（若 SISC 成立）

1. **现象不是起点，数据生成过程才是起点**：临床 truth 属于 evidence set，但训练常把它复制到单图。
2. **可识别机制**：独立 claim–view truth 揭示 view-invisible positives；study 共现决定错误迁移方向。
3. **因果证据**：exact-parent children 只改变 supervision incidence，即改变 wrong-view hallucination。
4. **最小方法**：把 image→report 改为 evidence-set→typed claim graph，不加花哨 decoding。
5. **OE/报告验证**：在 matched claims、长度和 coverage 下，同时降低 wrong-view fabrication 并保持 right-view recall。
6. **边界清楚**：不声称解决知识型、治疗型或无独立来源的 hallucination。

这是比“又一种 hallucination mitigation”站位更高的问题：它可能说明一部分医学 VLM 幻觉是**监督系统把临床 study truth 错写成 image truth 后的必然后果**。

## 10. 当前冻结决策

- **不启动 GPU。**
- **首查 SISC 的独立 claim–view truth 是否足够。**这一步失败，SISC 当天死亡，不靠弱标签续命。
- **同时做 SSEP 的自然 scope admission。**第二模型或自然性不足，当天死亡。
- **ESTE 暂停。**除非 CPU audit 显示至少两种 evidence signature 共享同一 source-specific causal law。
- 任何候选都不得用更短回答、更少 positive claims、统一阴性、大量 hedge、拒答或 evaluator/parser 漏检换取“幻觉下降”。


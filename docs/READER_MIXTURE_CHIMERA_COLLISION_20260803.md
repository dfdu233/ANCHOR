# Reader-Mixture Chimera：碰撞检索、构念审计与一日证伪门

**日期：2026-08-03**  
**Reviewer verdict：REJECT AND PIVOT（不得进入当前医学 VLM 幻觉主线）**  
**允许保留的角色：低成本数据/评测诊断，或未来获得真正 multi-report、multi-reader、jointly adjudicated 数据后的新问题预研。**

## 0. 结论先行

候选观察是真实而有意思的：若把同一张 VinDr-CXR 的多个读者 finding sets 逐 claim 聚合，输出可能不是任何一个读者的完整判断集合。但当前版本有两个足以拒稿的致命缺陷：

1. **`不属于任一读者的 claim set` 不是 `医学上错误的 claim set`。** VinDr 的正 finding 可以共同成立；单个读者未同时提及两个 finding 可能只是漏报、报告意图或敏感度差异。union 中每个 claim 反而至少有一位读者支持。没有独立影像真值或医生对联合陈述的判定，不能把 reader-crossing 称为 hallucination。
2. **label substrate 不能解释现成医学 VLM。** VinDr 提供的是独立多读者标签而非自然语言报告，且没有证据表明 HuatuoGPT-Vision、Hulu 或 LLaVA-Med 用这些相互冲突的同图读者集合训练。对本地 Huatuo CXR source bank 的审计发现 5,846 个 source groups 各有两个 VQA response units，但每组始终只有 **一个**原始 caption；它们是同一 caption 的 alignment/instruction 改写，不是独立读者报告。因而即使受控 fine-tuning 能制造 chimera，也只证明 synthetic training artifact。

此外，作为方法的“latent reader mixture / coherent set decoder”已被多条成熟路线覆盖：多读者医学分割、crowd/annotator modeling、multi-label crowd learning、conditional label-set mixtures 与 multi-reference sequence mixtures。开放式报告中的“把混合分布错误渲染成单一合取陈述”尚有语义增量，但不足以独立支撑 ICLR oral。

## 1. 冻结构念：三个不能混用的事件

设同一图像的读者集合为 \(S_1,S_2,S_3\subseteq\mathcal C\)，模型输出的确定阳性 claims 为 \(O\)。必须分别报告：

1. **Panel-unsupported claim**：
   \[
   c\in O\setminus (S_1\cup S_2\cup S_3).
   \]
   这是 panel-relative fabricated positive 的候选，但仍需考虑三位读者共同漏诊。
2. **Reader-crossing composition**：
   \[
   \nexists r,\;O\subseteq S_r.
   \]
   只说明输出跨越了读者集合；它不是错误真值。
3. **Clinically incompatible joint commitment**：\(O\) 中存在被影像或临床规则反驳的 claim，或多个 claims 的联合陈述不可能/不恰当。这个事件必须由独立 adjudication、可靠外部真值或明确互斥关系定义。

本文候选的 29.4% 属于第 2 类，却被叙事推向了第 3 类。这是核心构念越界。

术语约束：在没有第 3 类证据前，只能写 **reader-crossing set** 或 **cross-reader composition**；`chimera hallucination` 禁用。

## 2. 本地 VinDr CPU 审计

数据文件：`image_labels_train.csv`，SHA256：

```text
30f1b5bd5e9491cb6a0f775ade7fb77fe09de15c3f27d7f52d93d5376819ed93
```

固定 R8/R9/R10，共 5,501 张三位读者都标注的图像。直接采用官方发布的 22 个 local findings（`Aortic enlargement` 至 `Other lesion`；不含 6 个 global diagnosis 和 `No finding`）时，CPU 复核得到：

| 聚合策略 | reader-crossing | 比例 | 非空输出 | 输出平均 claim 数 | `K_output > max K_reader` |
|---|---:|---:|---:|---:|---:|
| union，票数 ≥1 | 1,685 / 5,501 | 30.63% | 4,203 | 2.747 | 1,685 |
| majority，票数 ≥2 | 67 / 5,501 | 1.22% | 4,120 | 1.567 | 9 |

用户提供的 1,618/5,501 与 58/5,501 **不能在同一个已记录的官方 22-finding contract 下同时复现**。它们可能来自另一个筛选 ontology、层级合并或不同的 subset 定义；在 exact script、列清单和 SHA 冻结前不得进入论文。这个差异不改变定性结论：union 容易 reader-crossing，majority 很少。

### 2.1 条件随机化：读者级 claim bundles 是否超过边际票数？

对每个 image-claim 固定原始票数 \(v_c\in\{0,1,2,3\}\)，仅在三位匿名读者之间均匀重排正票归属；1,000 次随机化，seed=20260803。该 null 保留每个 claim 的 reader support，却破坏同一读者内部的 claim 共现。

| 策略 | 实际 crossing 数 | null 均值 | null 95% 区间 | 下尾随机化 p |
|---|---:|---:|---:|---:|
| union | 1,685 | 1,863.6 | [1,825, 1,900] | 0.001 |
| majority | 67 | 79.2 | [65, 94] | 0.058 |

解释：union 的实际 crossing **低于**只由 per-claim 票数决定的随机分配，说明读者内部确有一定 bundle coherence；这支持“读者集合不是完全可交换的独立 Bernoulli 标签”。但它不证明 bundle 是互斥临床解释，更不证明 VLM 学到了或混合了这些 bundles。majority 的 reader-crossing 本来就很少，且相对该 null 不稳健。

### 2.2 一个简单但决定性的事实

对 union，\(U=\cup_r S_r\) 天然包含每个 \(S_r\)。所以 \(U\subseteq S_r\) 当且仅当 \(U=S_r\)。因此所谓 union chimera 率本质上就是“union 比每个单读者都多至少一个 claim”的频率；它主要测 **读者漏报的互补性**，而不是输出错误。

## 3. 为什么标准 CE 并不必然制造 union/chimera

### 3.1 Componentwise BCE

若每个 claim 独立使用 BCE，固定图像上的总体风险最优解是 reader marginal：

\[
p(c=1\mid x)=\frac{1}{3}\sum_r \mathbf 1[c\in S_r].
\]

阈值 \(>0.5\) 对应 majority，而不是 union。union 只在显式 max/OR 聚合、阈值 \(>0\)，或为 recall 极度下调阈值时出现。本地 majority crossing 只有约 1.2%，所以 30% union 不能作为“标准 componentwise training 必然失败”的证据。

### 3.2 Canonical sequence CE

若每个 reader set 被确定性序列化为完整参考序列，且模型对固定 \(x\) 精确拟合经验分布

\[
p(y\mid x)=\tfrac13\sum_r\delta(y=y_r),
\]

则理想 greedy decoding 不会凭空 union：每一步选择具有正条件概率的 next token，所得 prefix 必然仍是至少一个参考序列的 prefix；归纳到 EOS，输出落在一个观测模式中。

有限容量、跨图像泛化、非规范报告顺序、label smoothing 或近似解码当然可能拼接模式，但这已经是**待观测的模型现象**，不是 CE 的数学必然。必须先展示现成 checkpoint 的真实 OE 输出 reader-crossing，随后才有资格研究其来源。

## 4. 机制级碰撞矩阵

| 工作 | 同一现象/对象 | 同一机制 | 同一方法 | 对本候选的影响 |
|---|---|---|---|---|
| [A Probabilistic U-Net for Segmentation of Ambiguous Images, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/473447ac58e1cd7e96172575f48dca3b-Abstract.html) | 多位专家为同一医学图像给出多个合理结构化输出 | 单一平均预测抹掉多模态标注分布 | latent generative model 采样 coherent hypotheses 及其频率 | **核心抽象直接碰撞**；从 mask 换成 claim set 不足以构成新机制 |
| [Learning Calibrated Medical Image Segmentation via Multi-Rater Agreement Modeling, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Ji_Learning_Calibrated_Medical_Image_Segmentation_via_Multi-Rater_Agreement_Modeling_CVPR_2021_paper.html) | 医学 multi-rater disagreement | majority/preferred-rater 丢失 disagreement | expertise-aware、重建多读者 grading | 直接覆盖“不要先聚合、建模 reader expertise” |
| [Pionono, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Schmidt_Probabilistic_Modeling_of_Inter-_and_Intra-observer_Variability_in_Medical_Image_ICCV_2023_paper.html) | 医学专家 inter/intra-observer modes | reader-conditioned variability | 生成模仿各 rater opinion 的 coherent outputs | 与“latent reader-mode decoder”高度重叠 |
| [D-Persona, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Diversified_and_Personalized_Multi-rater_Medical_Image_Segmentation_CVPR_2024_paper.html) | multi-rater 医学输出 | 合并 ground truth 不可达 | diversified + personalized latent reader outputs | 方法主张几乎被覆盖，仅输出从 mask 改为 report |
| [Deep Learning from Crowds, AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11506) | 多 annotator labels | annotator reliability/bias | EM 或 crowd layer；覆盖 classification、regression、sequence labeling | reader heads / confusion model 不是新方法 |
| [You Are What You Annotate, Findings EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.832/) | disagreement 被聚合抹掉 | annotator-specific perspectives | annotator/annotation embeddings | reader identity conditioning 已有 NLP 邻近工作 |
| [Unbiased Multi-Label Learning from Crowdsourced Annotations, ICML 2024](https://proceedings.mlr.press/v235/xia24a.html) | **同一实例、多 annotator、多 label sets** | crowd transition + label correlations | unbiased risk + decoupled autoencoder | 对 claim-set substrate 的直接碰撞，不能声称首个 multi-rater multi-label learning |
| [Conditional Bernoulli Mixtures for Multi-Label Classification, ICML 2016](https://proceedings.mlr.press/v48/lij16.pdf) | 多标签联合分布与 subset prediction | independent marginals 丢失 label dependence | conditional mixture 建模 joint label sets | coherent set mixture 是成熟多标签方法 |
| [Mixture Models for Diverse Machine Translation, ICML 2019](https://proceedings.mlr.press/v97/shen19c.html) | 同一输入多个 sequence references | 单一生成器难覆盖 output modes | EM mixture、质量—多样性多参考评测 | multi-reference sequence mode 建模已直接存在 |
| [Pragmatic Radiology Report Generation, ML4H 2023](https://proceedings.mlr.press/v225/nguyen23a.html) | 图像相同但报告内容取决于沟通意图 | mention/omission 不完全由图像决定 | indication conditioning、清理不可从图像推断的信息 | 反驳“某读者未提及 = 不支持”；也是 report semantics 的强近邻 |
| [Collaboration between clinicians and VLMs in RRG, Nature Medicine 2024](https://www.nature.com/articles/s41591-024-03302-1) | radiologist 对报告质量/错误显著分歧 | 评价与临床场景差异 | multi-reader expert evaluation | 证明多读者评判必要，但没有证明训练时 reader mixtures 造成生成 chimera |

检索结论的精确措辞应是：**没有检索到同时研究“multi-reader claim-set modes 被医学 VLM 渲染成单一合取报告，并用独立 joint truth 证明其造成 hallucination”的工作；但其三个组成机制分别已有强直接近邻。** 这更接近一个尚未被实证的新 failure metric，而不是全新 learning principle。

## 5. 能否解释当前 Huatuo 等医学 VLM？目前答案是否定的

### 5.1 需要什么训练证据

要把 reader-mixture 作为当前 checkpoint 的成因，至少需要证明：

1. 同一个或近乎相同的医学图像在训练中配有多个**独立来源**的完整 claim sets/reports；
2. 这些报告存在不能用同一 caption 改写、模板变化或信息补充解释的真实 mode difference；
3. 训练暴露强度足以影响 checkpoint；
4. 删除/condition reader identity/保持 mode 后，现成模型的 joint error 因果下降。

### 5.2 Huatuo 本地 source audit

对本地 PubMedVision CXR source bank：

```text
source groups                            5,846
assistant response units                11,692
groups with >1 response unit             5,846
response units per group                 exactly 2
groups with >1 original caption              0
captions per group                       exactly 1
stage pair                               alignment + instruction_tuning
```

文件指纹：

```text
source_index.jsonl   77cb3858dc14b24cc72d5e715757f6925fec93a89107d7fcc3ae6e92dc0bbe
source_records.jsonl 7e6467faad807146de6add5d6b057db6d6005ba62b7be1ff188be73880bf589b
```

[HuatuoGPT-Vision](https://arxiv.org/abs/2406.19280) 说明 PubMedVision 是由 PubMed image-text pairs 经 GPT-4V denoise/reformat 得到的 1.3M VQA。上述本地 CXR 子集与此一致：多 VQA target 来自同一 caption，不是多位独立 radiologists。因此它可能产生 **reformatting inconsistency**，但不能支持 **reader-mixture** 机制。

MIMIC-CXR 等主流 report-generation 数据也按 study 配一份临床报告发布，并不提供同一 study 的独立 multi-reader report distribution。[VinDr-CXR](https://www.nature.com/articles/s41597-022-01498-w) 确实为训练图像保留三位独立读者，但提供的是 28 维标签与 boxes，不是完整 OE reports；其测试集反而发布五人 adjudicated consensus。当前没有证据证明所测医学 VLM 用 VinDr 的独立集合训练。

所以：在 VinDr 上观察到现成 VLM 的 reader-crossing 输出，最多说明输出与这组三位读者的集合结构不同；它不能反推训练时 reader mixture，更不能排除视觉误识、语言先验、prompt presupposition 或普通 multi-label error。

## 6. 一日 decisive pre-gate（不把自动 extractor 当真值）

当前自然因果门已经因 §5 失败。若仍想用一天确认该方向是否值得保留为 side diagnostic，固定以下流程；不需要训练 7B 模型。

### Gate A：CPU construct/power audit（约 2 小时）

1. 冻结 exact CSV SHA、reader panel、claim columns、`No finding` 处理、positive/negative/uncertain 规则。
2. 同时报告 panel-unsupported、reader-crossing、joint-incompatible 三种事件，禁止合并。
3. 执行 §2 的 conditional randomization；按 image cluster bootstrap。
4. 统计每个 crossing case 的 witness claims、读者归属、输出 \(K\)、每位 reader 的 \(K_r\)。
5. 若 majority crossing <100 或合格 finding pair <100，不以 majority 作为机制主实验。

这一门已经显示：reader bundle coherence 可测，但 majority crossing 极少，union 主要反映互补漏报。

### Gate B：现成 checkpoint 的真实生成门（4–6 小时小 GPU；本任务未运行）

从 R8/R9/R10 的图像中，image-disjoint 抽取：

- 100 张 union reader-crossing；
- 100 张同 support/同 union-K 但非 crossing；
- 100 张三读者 unanimous；
- 不根据任何模型输出选样。

Huatuo/Hulu 各用同一个长度锁定 OE prompt greedy 生成一次。统一 claim extractor 在看结果前冻结，只负责生成待审候选，**不定义真值**。至少盲审 60 个 crossing 与 60 个 matched non-crossing outputs，两个临床读者独立判定：每个 claim 的 supported/refuted/unobservable，以及整个 claim conjunction 是否 jointly acceptable；分歧 adjudicate。

只在以下条件同时满足时继续：

1. 两个模型都产生足量 multi-claim outputs（每模型至少 50 个 reader-crossing outputs）；
2. reader-crossing 与输出长度、claim 数、union-K、prompt、图像难度匹配后，仍显著预测 physician-adjudicated joint error；
3. crossing 的风险不是只由某一个 panel-unsupported fabricated claim 驱动；
4. 至少出现 20 个被医生确认的“每个原子 claim 看似可辩护，但联合确定陈述不恰当”的 witness cases；
5. bootstrap 95% CI 排除 0，且相对风险至少 1.5。

**只有自动 extractor、union reference 或 LLM judge：直接判定 Gate B 未通过。** 它们可以定位样本，不能建立 clinical joint truth。

### Gate C：受控 training susceptibility（仅在 A/B 通过后；小 GPU）

对同一小模型/同一初始化做四臂：

1. 每位 reader 的 canonical set sequence 各为独立 target（sequence CE）；
2. raw componentwise BCE；
3. 显式 union target；
4. latent coherent-set mixture。

在 image-disjoint test 上比较 actual generated set，不比较训练标签本身。主指标是 physician-adjudicated joint error 和 panel-unsupported claims；reader-crossing 仅为机制中介。若只有显式 union arm 失败，结论是 aggregation policy artifact；若 CE/BCE 也失败且 latent mixture 在 fixed coverage 下修复，才说明模型有 mode mixing susceptibility。

但即便 Gate C 阳性，也只能支持 controlled model organism；没有 §5 的真实训练暴露证据，不能解释 Huatuo 的自然 hallucination。

## 7. Coverage-preserving mitigation：唯一逻辑上成立的形式与其碰撞

### 7.1 不可能性边界

令 raw candidate atoms 为 \(U\)。如果“coverage-preserving”要求单个常规报告继续把 \(U\) 中每个 claim 都作为确定阳性合取断言，那么输出仍是

\[
\bigwedge_{c\in U} c,
\]

它不可能同时消除 reader-crossing。固定原子覆盖、禁止删除 claim、又不改变语义类型时，任何 decoding trick 都无解。

因此只有三条路：

1. 删除/交换 claims（用户已禁止以少说获益）；
2. 获取新视觉证据或可靠 adjudication；
3. **改变逻辑类型**：把一个合取集合改成多个明确的、相互备选的完整解释。

### 7.2 最小可行形式：Disjunctive Mode Rendering（DMR）

学习联合后验 \(p(S\mid x)\)，输出少量完整 modes：

\[
(\bigwedge_{c\in S_1}c)\;\lor\;(\bigwedge_{c\in S_2}c)\;\lor\cdots
\]

并要求：

- **atom coverage 锁定**：\(\cup_m S_m=U\)，raw candidate claim 一个不少；
- 每个 branch 内部是 joint model 的高概率 mode，而非 per-claim marginal 拼接；
- 只对 mode membership 作明确分支，例如“Interpretation A: …; alternative B: …”，不使用“可能存在多种异常”这类泛化 hedge；
- consensus claims 提到一次，差异 claims 放入显式 branches；
- 报告 branch 数、总 claim atoms、长度、拒答率。

这比删 claim 更符合 coverage 约束，但有两个重大问题：

1. VinDr 的 readers 可能只是各自漏报，而不是互斥解释；强行 disjunction 可能比 union 更不真实。
2. latent mixture/coherent hypothesis 已被 Probabilistic U-Net、Pionono、D-Persona、conditional Bernoulli mixtures 与 sequence mixtures 覆盖。新意主要是临床语言的 logical rendering 与相应评测，不是新的学习原理。

必要 baselines：union、majority、independent marginals top-K、classifier chain、conditional Bernoulli mixture、reader-conditioned heads、random branch partition、temperature scaling、固定-K claim exchange。所有比较锁定 atom coverage，并由医生分别评价 branch correctness 与临床可用性。

## 8. Reviewer-style fatal flaws 与 oral ceiling

### Fatal flaw 1（CRITICAL）：构念没有临床真值

“没有任何单读者同时写出全部 claims”不等于 claims 不能同时成立。VinDr positive ontology 缺少足以定义 joint impossibility 的 polarity、laterality、temporal、severity 与 causal attributes。没有独立 adjudication，headline 只是 annotation-set provenance statistic。

### Fatal flaw 2（CRITICAL）：自然模型的训练成因不可识别

当前没有 same-image independent-reader reports 的训练暴露证据；本地 Huatuo CXR 审计反而显示同图多个 VQA 来自一个 caption。对 VinDr 人为 fine-tune 再观察到 mixing 只能证明我们制造了它。

### Major flaw 3：方法碰撞

reader latent、annotator embedding、joint label mixture、coherent hypotheses 和多参考 sequence mixture 都已有顶会直接近邻。若贡献退化为“第一次用于医学 OE”，属于 setting transfer。

### Major flaw 4：固定 coverage 下方法需要改变任务

不删 claims 时必须把单报告合取改成多分支析取。这可能是正确的逻辑设计，但临床工作流、现有 metrics 与医生偏好未必接受；它不再是普通 abnormality listing/report generation。

### Ceiling

| 证据状态 | 可信 ceiling |
|---|---|
| 只有 29.4%/1.1% label-set statistic | 内部诊断；不成论文 |
| 加自动 extractor 的现成模型 crossing | workshop / benchmark note，仍不能叫 hallucination |
| 加医生 joint adjudication，证明 crossing 对临床错误有独立预测力 | MICCAI/ML4H 级 failure analysis 候选 |
| 再有真实 same-image multi-reader report training provenance、因果去除、跨两模型 fixed-coverage 改善 | ICLR main-track 候选 |
| 再提出超越已有 mixture models 的统一理论，跨 RRG/segmentation/NLP 证明“marginal-to-conjunctive overcommitment”并给出非平凡边界 | 才可能讨论 oral ceiling |

以现有证据，**ICLR oral ceiling 不成立**。

## 9. 最终决策

**Reject and Pivot。** 不运行 Reader-Mixture Chimera 的大规模 GPU 实验，不把它作为当前医学 VLM hallucination 的解释，也不实现 latent-reader decoder 作为主方法。

可保留的最小资产只有：

1. 将 reader-crossing 作为评测系统的一个 provenance diagnostic，明确不等同 hallucination；
2. 在未来获得 same-study independent full reports + joint adjudication 后，重启更严格的问题：

> 当训练监督只提供原子边际，却要求模型输出单一合取临床陈述时，模型是否产生 **marginal-to-conjunctive overcommitment**？

这个重写需要真正互斥或不可同时确定的 clinical attributes，而不是仅靠 VinDr positive finding union。当前更可信的主线仍应优先研究已有自然模型中可观测、可因果干预的视觉证据—语言承诺失配。

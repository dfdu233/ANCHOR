# 从竞赛提分到 ICLR 论文：医学 VLM 幻觉研究计划（2026-08-10）

## 2026-08-10 18:12 UTC superseding update

The earlier "patient-aligned intervention code" interpretation is rejected.
Its original placebo changed both patient and question.  Under the correct
placebo, which shuffles the RAG response only within exact question text, the
paired code is not better: full complete-case BAcc is 0.90985 versus 0.91339
for the placebo (delta -0.354pp, image-cluster 95% CI -0.961 to +0.263pp).
On the strictly exchangeable subset (n=1,384), the delta is -0.035pp with a
95% CI of -1.045 to +0.977pp.  The cached gain is therefore ordinary
question-template prior/stacking, not patient-specific clinical evidence.

Fisher-scored Minimum Intervention Basis and DICOM render fingerprints are
also rejected.  Fisher adds only 0.0123 partial R-squared after controlling
arm strength and subset size, is leave-one-arm-out sign-unstable, and does not
pass its permutation gate.  Five admitted DICOM render responses reduced
Huatuo VinDr BAcc from 0.7250 to 0.7125 (delta -1.25pp, 95% CI -7.50 to
+5.01pp).  These remain negative controls, not method contributions.

The only active scientific candidate is now **Cross-Patient Evidence
Transportability**.  A frozen CPU audit finds that raw RAG responses move
toward the present/absent state stated in other patients' reports on
CXR-VisHal: +13.12pp for Huatuo (95% CI +10.21 to +15.46) and +8.26pp for
Hulu (+5.49 to +10.24).  Knowledge-MIMIC is asymmetric: negative/no-state
transport replicates in both models, while positive-state transport does not.
This is observational discovery evidence only; its shuffled causal control is
underpowered.  The candidate survives only if matched positive/negative donor
state swaps establish a signed, claim-specific causal effect in two models and
a source-typed state firewall removes that effect without deleting knowledge,
shortening output, increasing FN, or reducing fixed claim coverage.

The baseline queue remains unchanged and has priority on the single GPU.  A
32-claim CPU-built three-arm canary is ready, target-label blind and hash-bound;
no model result is claimed from it yet.

## 2026-08-10 17:18 UTC superseding update

CMP is no longer the provisional paper method.  QueryBandits directly covers
per-query mitigation selection and V-ITI covers selective visual intervention;
our pre-treatment router also failed its own transfer gate (+0.14pp, CI crossing
zero, 1.90 generations, FP worse).  It remains a required competition baseline.

The strongest new cached result is an **aligned intervention code**: on a
patient-hash blind CXR-VisHal test split (n=734), the validation-selected best
single arm achieved BAcc 0.8650, while paired Huatuo/Hulu plain+RAG responses
achieved 0.9102 (+4.47pp, 95% CI +2.17 to +6.66pp).  Keeping both plain model
outputs aligned but shuffling only their RAG responses across patients reduced
BAcc to 0.8761; the paired-code advantage over this placebo was +3.42pp with CI
excluding zero.  FP fell 63→23 but FN rose 37→42, so the result has not passed
the two-risk gate.

This is not yet an ICLR idea: SAC3, ESI, VGS-Decoding, QueryBandits and BCEA
occupy generic perturbation consistency, intervention uncertainty, distorted
image contrast, per-query routing and active evidence acquisition.  The code is
therefore frozen as a Kaggle-strength instrument.  The only potentially
research-level question is an identifiability boundary on reader-grounded
claims: when does the *patient-aligned response to clinically admissible
interventions* reveal recoverable clinical evidence, and when is it merely a
shared prior?  It must beat same-model stochastic ensembles, patient-misaligned
placebos, VGS/SAC3-style scores, and matched compute, then causally transport the
identified state without increasing omission.

## 2026-08-10 17:05 UTC superseding update

FIN is rejected rather than promoted: across four synthetic clarity strengths
and five seeds, ordinary full consistency matched or beat FIN; the cached
single-axis curvature screen was also negative.  A blind conditional-control
screen then showed that per-image style-matched normal controls did not beat a
fixed per-finding offset on either Huatuo or Hulu.  A blind intermediate-state
readout screen further contradicted a universal early-evidence story: Huatuo's
selected non-final macro AUROC was 0.645 versus 0.754 at the matched final
layer, while Hulu differed by only +0.0005.

The current algorithm candidate is therefore **Clinical Mitigation Portfolio
(CMP)**, derived from the only repeatedly positive observation: mitigation
effects are strongly heterogeneous across claims.  CMP treats each mitigation
as a treatment with two separate adverse events—fabrication and omission—and
learns a Pareto policy from out-of-fold counterfactual outcomes:

\[
\pi^*(x)=\arg\min_{m\in\mathcal M}
\big(\widehat R_{FP}(m\mid x),\widehat R_{FN}(m\mid x),\lambda C_m\big).
\]

Unlike majority voting, a policy is admissible only when it is non-inferior on
both FP and FN across source domains; unlike generic routing, the paper's
scientific object is the **individual treatment effect of hallucination
mitigation** and the failure of average benchmark gains to predict per-claim
benefit.  A first Huatuo cached pilot selected a candidate on fit/tune only and
opened a disjoint 407-sample test once: BAcc rose from 0.7275 to 0.7846 while FP
fell 67→61 and FN fell 45→29.  This is still a development finding, not a paper
result.  A conservative direction-wise certificate failed to transfer cleanly,
which establishes that marginal safety certification alone is insufficient;
the next formal version must use multi-domain Pareto/OOD risk rather than a
single pooled certificate.

CMP remains provisional until it transfers across two models and an unopened
dataset, works on fixed-K OE claims, and survives collision checks against
HALP, generic routers, ensembles, and HalluTrace.  The baseline queue and the
strict stop criteria at the end of this document are unchanged.

## 0. 当前判决

项目**不能停止，也不能宣称已经达到 ICLR 水平**。当前只有两个可靠事实：

1. 竞赛式受控推理组合能提高胸片 CE：Huatuo 从 Knowledge-MIMIC 学到的两探针路由迁移到 CXR-VisHal，BAcc 从最佳单探针 81.87% 提至 84.45%，按图像 bootstrap 的 95% CI 为 +1.62 至 +3.53pp；平均调用 1.67 次。
2. 该结果本身不是论文创新：一致性、反事实探测、多视图解码、模型集成和阶段归因均有直接先例。

因此，组合器只作为“竞赛上限与现象发现仪器”。暂定科学候选是：

> **幻觉可能主要来自域因素之间的交互，而非任何单一域偏移；训练应保留有用的图像呈现、提示和知识主效应，只抑制使临床极性翻转的交互项。**

候选名：**Factorial Interaction Neutralization (FIN)**。它尚未通过机制门槛，不能写成既定贡献。

## 1. Baseline 不受影响

- 唯一正式矩阵：`configs/unified_eval/baseline_matrix_v1.json`。
- 当前覆盖审计：336 cells；24 completed、1 generated-unscored、173 pending、3 running/partial、135 N/A。
- GPU 当前由 baseline 持续占用，所有队列共享 `corrected_runs/detached_jobs/locks/gpu0-vindr-v2.lock`。
- 新实验只通过同一锁排队，不能杀进程、不能更换方法默认参数、不能覆盖输出。
- 正式 baseline 仍坚持 T0→T1→T2→Full，关闭态 32/32 token-exact，官方架构开启态与官方入口 32/32 token-exact；失败则 N/A，不用近似方法冒充。

## 2. 已确认的证据账本

### 2.1 正信号，但只属于竞赛层

| 结果 | 数值 | 能说明什么 | 不能说明什么 |
|---|---:|---|---|
| Huatuo 普通/RAG 响应诊断 | CXR 上两次答案一致错误率 10.6%，不一致错误率 45.7% | 受控响应包含可靠性信息 | 不等于不一致必然是幻觉 |
| 跨数据错误检测 | AUROC 0.696→0.804 | 响应差异比单次置信度多信息 | 不是因果机制；目标集已看过 |
| 跨数据自适应路由 | BAcc 81.87→84.45%，CI 排除 0 | 竞赛式 source-calibrated routing 可提分 | 与 SAC3/SCoOP/HalluCXR 等存在强碰撞 |
| 简单 signed evidence | Knowledge-MIMIC 与 CXR 有正增益，SLAKE 无 | 胸片探针有互补性 | 不能泛化到所有医学模态 |

### 2.2 已证伪或被严格降级

1. “训练域中心”与 FedDG 风格方向不能稳定提高医学 VLM；域变化方向不等于通向训练中心。
2. “正确答案普遍藏在固定中间层”不成立；同疾病同真值随机轨迹比患者自身轨迹更易满足旧 recoverability 定义。
3. reader-clarity 在两模型上没有形成统一的后层擦除机制。
4. style flip 稀少，简单 style drift 对正确性预测弱；不能直接把风格敏感包装为机制。
5. lesion-aware head suppression 没有优于低视觉依赖 head suppression；病灶约束未产生因果增量。
6. 单轴二阶曲率快速筛选为负：确认集 n=16、error=5，scalar AUROC 0.782，一阶响应 0.873，加入单轴曲率降至 0.600。
7. 以上负结果只否定各自受限测量，不证明模型内部完全没有临床信号。

## 3. 综述仓库审计与流程修正

`ZJUMAI/Multimodal-Medical-Reasoning` 适合作为候选文献地图，不是可运行框架：仓库没有数据转换、训练、推理和评测实现。抽查还发现多个错误链接（如 DynamiCare、DiagECG、MOTOR、EIT-1M、EEG-FM-Bench 指向无关论文），所以每条方法必须回到原论文和官方代码核验。

综述的三阶段结构仍有价值：

| 医疗推理阶段 | ANCHOR 当前覆盖 | 近期补充 | 不应混做 |
|---|---|---|---|
| 多模态理解 | VinDr reader votes、CE/OE/report、render/null/swap controls | MediConfusion 小规模 ambiguity 控制；一个 reasoning-trained medical VLM | 为追求数量加入病理、基因、手术、ECG 全家桶 |
| 外部知识利用 | 去污染 shared RAG、no-context/shuffle/image-swap | 仅 knowledge-claim 轨加入 evidence rerank 或 confidence re-retrieval | 用 RAG 文本“修复”纯图像 claim |
| 长程分析 | 当前较弱 | ClinHallu 作为阶段诊断 benchmark/control | 把 Codex 多代理运行当模型方法 |

评测不改成综述中的泛化指标清单。ANCHOR 继续使用 dataset-native 真值、patient/image cluster bootstrap、FP/FN/omission 分开、固定 claim coverage、长度与拒答审计；没有医生时，OE/report 的自动指标只能称 proxy。

## 4. 竞赛实验室：先赢，再解释为什么能赢

### 4.1 数据与验证

1. 每个数据集建立 patient/study-disjoint train/dev/private split；测试标签只开一次。
2. 所有方法保存 OOF 预测、NLL、回答长度、claim 数、错误类型和 provenance。
3. 建立专家错误相关矩阵；优先选择低错误相关而不是单项最高分的方法。
4. 设 public-LB 类开发集与 private-LB 类未开封 cell；阈值、路由和 checkpoint 只在 source/dev 选择。
5. 每个技巧必须同时通过 BAcc/FP/FN、coverage、length、拒答和 compute 审计。

### 4.2 提分阶梯

1. 最佳单模型/单方法。
2. 概率校准与 finding-specific threshold（只用 OOF）。
3. 低相关专家的 greedy forward selection。
4. patient/domain/finding router。
5. source-calibrated 自适应探针：低风险样本单次推理，高风险才追加探针。
6. hard-example mining：优先训练高交互、高分歧且有可靠标签的样本。
7. LoRA checkpoint soup / OOF blend；只能放在竞赛附加模块，不能冒充核心机制。
8. pseudo-label 只用于无标注池，并要求 reader-vote/多模型一致与置信区间过滤；不得污染测试集。

每一步都做单变量增量实验。无 private holdout 增益或增益来自回答变短，则删除。

## 5. 暂定论文主线：Hallucination Is Interaction, Not Shift

### 5.1 为什么它不同于旧 DG

旧方案试图把不同域拉到一个中心，隐含假设是域变化都应被消除。但医学 DICOM window 能揭示病灶，知识上下文能补充非视觉事实，prompt 能规定回答任务；这些主效应可能合理。

对同一 claim，构造图像呈现因素 `a` 和语言/知识因素 `b` 的 2×2 方格，记 claim margin 为 `m_ab`：

```text
                    prompt/context b0     prompt/context b1
render a0                 m00                   m01
render a1                 m10                   m11
```

混合差分：

\[
I_{ab}=m_{11}-m_{10}-m_{01}+m_{00}.
\]

- `a` 有主效应：不同 window 改变证据清晰度，可以保留。
- `b` 有主效应：问题表达或可靠知识改变任务，可以保留。
- `I_ab` 大：图像结论取决于某个不应相关的图像—语言组合，是交互型脆弱性。

### 5.2 训练方法（仅在机制 gate 通过后）

\[
\mathcal L=\mathcal L_{claim}+\lambda_I\lVert I_{ab}\rVert_1
+\lambda_R\mathcal L_{reader\ distribution}.
\]

- `L_claim`：原子 finding 的支持/反驳/无法判断监督。
- `L_interaction`：只压制预注册的、临床语义保持因素之间的混合差分，不强迫各域表示相同。
- `L_reader distribution`：用 VinDr 0/3–3/3 reader votes 保留模糊病例，不把分歧强压为硬标签。
- 训练阶段使用 factorial views；部署仍是原始单图单 prompt，一次推理。

### 5.3 Kaggle 技巧如何服务科学问题

- OOF error correlation 用于挑选最有信息的干预因素。
- hard-example mining 聚焦高交互病例。
- checkpoint soup 检查结果是否依赖单一训练随机性。
- private holdout 防止把测试集阈值拟合当贡献。
- 这些技巧若提高分数但不支持“交互项导致错误”的预测，只留在附录或系统结果，不进入论文标题。

## 6. 最小实验与淘汰顺序

### E0：2×2 interaction pilot（已排队）

- Huatuo、VinDr clear-positive reader-box claims、16 例。
- 两个合法 DICOM center render × 两个语义等价 claim prompts。
- 固定 teacher-forced 评分位置，记录 5 层、每 head 输出、claim margin。
- 只做 feature discovery，不宣称确认。

### E1：独立机制确认

- disjoint dev/confirmation，各至少 100 个 claims，必须含 FP、FN、TP、TN 和 reader disagreement。
- 二模型；第三个 intervention family 完全留出。
- 对照：entropy、prompt consistency、style drift、一阶 delta norm、HALP static probe、随机/pixel-distance-matched transform。
- 通过：交互特征在控制 entropy、一致性、一阶 norm 后，错误 AUROC 增量至少 0.05，image-cluster bootstrap CI 排除 0；至少两模型成立。
- 失败：停止 FIN，不调测试阈值挽救。

### E2：最小训练确认

- 先在 Huatuo/LLaVA-Med 的 LoRA 或 projector adapter 上训练；不做全参。
- 比较 ERM、普通一致性、MixStyle/MatchDG、robust instruction tuning、FIN。
- source selection：VinDr train；private tests：VinDr heldout render、CXR-VisHal、Knowledge-MIMIC；SLAKE 只作跨模态边界。
- 通过：两个模型、两个胸片测试集上 BAcc 提升，FP 与 FN 均不恶化；OE fixed-K claim precision 提升且 omission 不增加。

### E3：因果与组合验证

- 定位高交互层/组件，做 good↔bad 双向 patch。
- 随机、正交、norm-matched、encoder-only patch 对照。
- 需要层局部、双向、剂量单调，并能迁移到未见 prompt/render family。

## 7. 论文逻辑骨架（暂定，未冻结）

| 环节 | 内容 |
|---|---|
| Paper type | Cross-domain Technique：把 factorial design/ANOVA 的“主效应—交互效应”分离迁入医学 VLM DG 与 hallucination |
| Background | 医学 VLM 对 render、prompt、RAG 敏感；现有方法多消除域差异、做一致性或融合答案 |
| Limitation 1 | 域不变方法把可能有用的医学主效应一并压掉 |
| Limitation 2 | 一致性/多视图方法无法区分稳定错误与真正证据，也常需多次部署推理 |
| Limitation 3 | 现有阶段归因说明错误在哪里，却未给出可直接训练、单次部署的交互约束 |
| Key idea | 不消除域变化，只消除使临床 claim 依赖于因素组合的非加性交互 |
| Challenge 1 | 确认哪些 render/prompt/knowledge 操作真的是 label-preserving 因素 |
| Challenge 2 | 证明交互项提供超越主效应、entropy 和 consistency 的信息 |
| Challenge 3 | 在减少 FP 的同时保持 FN、coverage 和 ambiguity calibration |
| Module A | Reader-grounded factorial data cube 与干预审计 |
| Module B | FIN interaction objective 与 hard-interaction curriculum |
| Module C | Claim-level CE/OE/report 统一评测与 fixed-coverage causal controls |

当前四项逻辑检查在概念上连贯，但 Methodology→Contribution 尚未通过，因为 E1/E2/E3 没有结果；因此不能写 Introduction 或锁标题。

## 8. 文献碰撞边界

- [SAC3](https://arxiv.org/abs/2311.01740)：语义等价问题扰动与跨模型一致性检测。
- [Prompt Multiplicity](https://arxiv.org/abs/2602.00723)：一致性不等于正确性，prompt multiplicity 普遍存在。
- [HALP](https://aclanthology.org/2026.eacl-long.287/)：静态内部表征探针与 selective routing。
- [LENS](https://openreview.net/pdf?id=oh3c2ieVab)：医学 counterfactual views、跨视图稳定性与训练/解码干预。
- [RITUAL](https://arxiv.org/abs/2405.17821)：图像增强与多视图输出融合。
- [ClinHallu](https://arxiv.org/abs/2606.14697)：视觉识别、知识回忆、推理整合三阶段替换归因。
- [Diagnosing Modality Interference](https://arxiv.org/abs/2505.19616)：扰动增强与跨模态一致性训练。
- [Neural-ANOVA](https://arxiv.org/abs/2408.12319)：用 ANOVA 分解解释神经网络交互效应，但不是医学 VLM 幻觉训练。
- [Failure Modes of DG](https://openaccess.thecvf.com/content/CVPR2022/html/Galstyan_Failure_Modes_of_Domain_Generalization_Algorithms_CVPR_2022_paper.html)：提醒域不变目标并不自动胜过 ERM。

FIN 的暂存 novelty 仅是：**对预注册的医学语义保持因素构造交叉方格，只正则混合交互而保留主效应，并验证它是否因果地产生 claim hallucination。** 如果检索到直接同构方法，或 E1 失败，立即更名为负结果并 pivot。

## 9. 严格停止标准

只有以下条件全部满足，才允许说“完成了足够 ICLR 主会竞争力的论文”；oral 无法事前保证：

1. baseline 336 cells 全部 completed 或有可验证 N/A，所有数字可追溯。
2. 核心机制在至少 2 个模型、2 个胸片数据集、1 个跨任务/OOD 设定成立。
3. 核心方法在未开封 private holdout 有显著提升；不是测试集调参。
4. CE 与 OE/report 至少两类任务成立，固定 claim coverage 后 FP 降而 omission 不升。
5. 增益不是长度、拒答、统一阴性、温度、随机方向或更强 base model造成。
6. 有因果实验支持核心机制，不只是一张相关性 AUROC 表。
7. 最新碰撞审查仍留下实质 novelty delta，官方 baselines 完整。
8. 论文骨架四项一致性检查全过，代码/配置/答案/评分/provenance 可复现。

在此之前，项目状态始终是 `research_in_progress`，不能为了“必须完成”而降低判断标准。

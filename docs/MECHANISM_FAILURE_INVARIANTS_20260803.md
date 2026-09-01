# 医学 VLM 机制探索的失败不变量

**冻结日期：** 2026-08-03　　**范围：** PPI、Metric Calibration、SISC、SSEP、Reader-Mixture Chimera、Evidence-Set Closure、Spatial Claim Binding 的最终审计。  
**结论：** 七个分支并非主要败于模型能力，而是反复在 GPU 前缺少可识别机制所需的四个对象：独立 truth、双向 counterfactual、未被占据的机制预测、稳定 output contract。下一候选必须先同时取得这四者；不能再用更多模块弥补其中任一缺失。

## 1. 跨分支失败矩阵

| 分支 | 决定性失败 | 不能被写成什么 |
|---|---|---|
| PPI | 人工随机化与 power 合格，但自然 source bridge 仅 2/8 claims；空 cue→claim 又与 RaVL/shortcut/backdoor 直接碰撞 | 自然医学 provenance 机制 |
| Metric | VinDr spacing 无 patient-plane calibration truth；structured failure 强烈模型/格式依赖，clinical direct 跨模型信号为 0；scale law 已被 MedVision/FactCheXcker 覆盖 | patient-mm accuracy 或统一 calibration hallucination |
| SISC | 仅 44 个多图 study；没有逐图 `visible/refuted/unassessable` truth，missing box 不能作 negative | 复制 study report 导致 wrong-view hallucination |
| SSEP | regex 不是 scope truth；同任务第二模型 LLaVA 输出坍缩（13 unique texts、0 candidates），Hulu 又由少数模板主导 | 跨模型 shared-scope 机制 |
| Reader Chimera | reader-crossing set 不等于临床错误；没有自然 multi-reader report 训练暴露；latent reader/joint-mixture 方法已成熟 | “chimera hallucination”或当前 checkpoint 成因 |
| Evidence Closure | VinDr/SLAKE 都不能形成同一 proposition 的双 source `support/refute/unavailable` 四格；LLM-RG4、missing-modality、MIL 已占 broad mismatch | view/prior/history/metadata 的统一神经机制 |
| Spatial Binding | VinDr 有可靠 pixel boxes，但没有 phrase/patient-side/extent truth；反平衡 `A@u,B@v ↔ A@v,B@u` finding pair 为 0；通用 binding 与医学 grounding 均已覆盖 | identity 保留后的 report-level binding erasure |

## 2. 四条失败不变量

### I. Truth 不可由被检验对象自身产生

共享 report、模型 classifier、LLM judge、自动 extractor、缺失 box、单读者未提及、裸 DICOM spacing 和 reader-set crossing 都只能产生候选，不能定义机制真值。反复出现的构念越界是把“未观察到”改写成 `refuted/unassessable`，或把“没有任何读者写出整个集合”改写成“联合陈述错误”。合格 truth 必须与模型和目标文本独立，在**同一原子 proposition**上给出 `supported/refuted/unobservable`（或机制所需的等价三态），并保留 image/ROI/span/metadata pointer 与 patient/study identity。

### II. Perturbation 不是 counterfactual；必须能反转机制预测

一个有效 counterfactual 必须保持 claim identity、polarity、语义指称、输入/输出长度和主要 nuisance 不变，只交换机制变量，并同时拥有两个方向。PPI 有漂亮的人工正负互补，却没有自然桥；Metric 有相同 pixels 的 spacing 变换，却没有 patient truth且已碰撞；Spatial 没有任何反平衡 finding-location pair；SISC/Evidence Closure 没有同 claim 的 source 四格。没有双向 cells 时，layer probe 只能把 anatomy prior、难度、prompt 或 source prevalence 误称为机制。

### III. 新设置、新组合和更严 controls 不等于新机制

七个分支分别撞上 shortcut/backdoor、measurement/tool use、multi-view input-output correspondence、negation scope、multi-rater mixtures、MIL/partial labels、spatial binding/grounding。剩余空间不能靠“医学化”、把多个 source 放进统一框架，或把已有方法加 fixed coverage/norm controls 获得。合格候选必须在检索前写出一个**最近工作未预测、且与至少两个替代解释给出相反结果**的 law；例如干预变量只改变某一承诺维度，同时 identity、polarity、coverage 与 clear-case performance不变。若最近代码已实现同一 intervention/decoder，或修复退化为确定性 if-statement，立即降为 baseline/diagnostic。

### IV. Output contract 是科学变量，不是工程细节

structured JSON、clinical direct、binary explanation 与 native report 不能混合定义一个效应；reference 不能充当第二模型。输出坍缩、模板重复、截断、拒答、少生成 claims 会制造虚假的“低幻觉”。任何 OE 机制结论必须来自至少两个模型的同任务 native outputs，并先通过 nonempty、cap、解析、长度、claim-count、template diversity 和 refusal gate；主比较固定 claim slots 或 matched coverage/polarity。自动 parser 只产生盲审候选，医生/独立 truth 才定义 error。

## 3. 下一候选的硬性设计约束

下一候选必须是**一个自然变量、一个原子临床错误、一个因果路径**，并同时满足：

1. **Truth-first：** 在查看模型 outcome 前冻结 ontology；至少 3 个 claim types，每个关键 state/cell ≥30 个 patient/study-disjoint units；独立双人审计样本的三态一致率 ≥0.90，分歧 adjudicate。任何 absent-as-negative 或 model-as-truth 直接淘汰。
2. **Natural bidirectional support：** 每个 claim 同时存在机制变量的两方向，语义完全相同；每方向 ≥30，且 identity-only/nuisance-only classifier balanced accuracy ≤0.60。若只靠人工 cue/label permutation，必须**先**有自然 occurrence、自然错误与同通路 bridge，不再先训练 model organism。
3. **Mechanism-before-method：** 预注册 causal graph、主 law、两个强替代解释及各自相反预测；必须有一个 selective intervention endpoint，明确禁止以删 claim、统一 hedge、温度、输出缩短或拒答达成。
4. **Collision-before-compute：** 核验至少 5 个最近机制邻居及官方代码。核心 intervention、failure definition、deterministic repair 三者中任一已被直接覆盖，则停止；“更严格组合”不能越过此门。
5. **Native OE admission：** 至少两个模型、每模型 ≥100 个合格 native outputs；nonempty/parse ≥0.95、cap-hit/refusal ≤0.05，最大 exact-template cluster ≤0.20；主结果 matched claim count、coverage、polarity与长度。若问题只在结构化 prompt 或单模型出现，只能报告 prompt/model-specific failure。
6. **Lineage and power：** 若研究训练成因，必须有可验证 exact parent 与只改变单一训练变量的 children；若研究冻结 checkpoint，必须有 paired natural intervention。预先按 patient/image cluster 做 power，目标效应的 95% CI 必须可排除 0，而 clear-case 允许损失 ≤1pp。

## 4. 一日 fail-closed 淘汰门（GPU 前）

| 时段 | 必交 artifact | 当日 KILL 条件 |
|---|---|---|
| 0–2h：collision | 最近工作/代码矩阵；一句机制 law 与两种反事实预测 | 摘要级同义、代码已有同 intervention、或修复只是 rule/tool |
| 2–5h：truth census | 哈希冻结的 proposition-level manifest；truth provenance、patient split、state counts | <3 claims、任一关键 cell <30、缺 patient/study ID、truth 来自 report/模型/absence |
| 5–7h：counterfactual census | 两方向配对、nuisance table、identity-only test | 单方向、改变 proposition、balanced accuracy >0.60、无法固定长度/coverage |
| 7–9h：output admission | 两模型同任务 census；format/cap/template/claim-count audit | 任一模型 <100 outputs、parse/nonempty <0.95、cap/refusal >0.05、单模板 >0.20 |
| 9–12h：falsification pack | causal DAG、estimand、negative controls、cluster power、不可调阈值 | 替代解释不给出可区分预测，或目标 MDE 无 80% power |

只有五门全部通过才允许最小 GPU probe；任一失败即冻结可信负结果，不换 truth、不降 cell 数、不追加异任务模型、不把多个弱方向拼成“统一框架”。这套合同的目的不是提高实验门槛，而是确保下一次 GPU forward 真正在区分一个自然医学 VLM 机制。

## 审计来源

`PPI_V3_COLLISION_AUDIT_20260803.md`、`PROVENANCE_TWO_PLANE_BINDING_PROTOCOL_V3_2.md`、`corrected_runs/ppi_v31/CPU_FALSIFICATION_REPORT.md`、`METRIC_GAUGE_HALLUCINATION_COLLISION_20260803.md`、`METRIC_CALIBRATION_SIDE_PROBE_N8_RESULT_20260803.md`、`SISC_OUTCOME_BLIND_TRUTH_GATE_20260803.md`、`SSEP_SCOPE_ADMISSION_GATE_20260803.md`、`READER_MIXTURE_CHIMERA_COLLISION_20260803.md`、`EVIDENCE_SET_CLOSURE_COLLISION_20260803.md`、`SPATIAL_CLAIM_BINDING_COLLISION_20260803.md`。

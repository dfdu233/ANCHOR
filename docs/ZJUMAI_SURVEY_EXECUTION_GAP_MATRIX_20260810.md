# ZJUMAI Survey → ANCHOR 执行差距矩阵

**冻结日期：** 2026-08-10  
**输入：** `Multimodal-Medical-Reasoning/README.md`（625 行）与 ANCHOR 的 `configs/unified_eval/baseline_matrix_v1.json`。  
**目标：** 只回答四件事：已经覆盖什么、还缺什么强 baseline、最近工作是谁、新 benchmark 何时才值得加入。

## 0. 来源准入规则

ZJUMAI 仓库是文献目录，不是可执行评测框架。README 中存在错链或占位式链接，例如 ECG Foundation Model 与 PathChat 指向同一篇 Nature PathChat 论文，`DiagECG` 指向通用占位编号。因此下表只保留已经回到**原论文、官方仓库或官方数据页**核验的条目；survey 中的规模和功能描述不单独作为证据。

## 1. 当前 baseline 已覆盖

“已覆盖”仅表示已经进入冻结矩阵或数据协议，不等于已经得到临床有效性结果。

| Survey 对应能力 | ANCHOR 当前对应项 | 当前状态 | 执行决定 |
|---|---|---|---|
| 基础生成与搜索 | Greedy、Beam | 已进入统一矩阵 | 保留为所有任务必跑下界；Beam 必须做同算力说明 |
| Training-free 解码 | VCD、DoLa、OPERA、PAI、AvisC、VISTA、SECOND | 已进入统一矩阵；部分只完成代码/激活资格检查 | 不再重复接入；只有通过官方语义和共同样本协议的 cell 才进主表 |
| 外部知识利用 | `shared_medical_rag` | 已进入统一矩阵 | 作为最朴素 raw-RAG 对照；不能代表 RULE/MMed-RAG |
| 医学 VLM 主干 | Huatuo、Hulu、LLaVA-Med；Qwen2.5-VL 通用对照 | 已配置 | 继续作为 frozen-backbone 主矩阵 |
| CE / OE / report 三类任务 | CXR-VisHal、Knowledge-MIMIC、SLAKE；VQA-RAD/Visual-MIMIC；IU-Xray/MIMIC-CXR | 已配置但完成度不同 | 不再用新增同质胸片 CE 扩数量；优先补现有 OE/report 的合格评分 |
| 幻觉成因拆分 | [MedHEval](https://arxiv.org/abs/2503.02157) 的 visual / knowledge / context 任务思想 | CXR-VisHal 等切片已经部分采用 | 下一步是接入其完整官方 OE/成因切片，不另造同义 benchmark |

**2026-08-10 task audit correction:** 上表中的 `Visual-MIMIC` 490 不是短答案 OE。冻结 prompt
逐例要求生成 medical report，reference 也是多 claim 放射学报告；现有生成可复用，但必须转入
report clinical scoring，旧 `evaluate_oe_vqa` 分数不得进论文。当前矩阵另外缺少本地已经存在的
MedHEval Knowledge open-ended 2,318 例与 Context Misalignment 2,000 例；为避免运行中改变冻结矩阵，
它们在当前七集完成后作为 P0 extension，而不是现在插队。

## 2. 应补的强 baseline

| 优先级 | Baseline | 为什么必须补 | 可比方式 | 接入条件 / fail-closed 条件 |
|---|---|---|---|---|
| P0 | [RULE，EMNLP 2024](https://aclanthology.org/2024.emnlp-main.62/) / [官方代码](https://github.com/richard-peng-xia/RULE) | 直接研究医学多模态 RAG 的“检索过多”和“错误上下文过度依赖” | 先复现 calibrated top-k inference；DPO 部分单列为 trained baseline | 官方代码能跑通、checkpoint/数据许可齐全且关闭态与 native 一致；否则标 N/A，不用本地启发式冒充 |
| P0 | [MMed-RAG，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a559a5a8aa5ae6682ced009ad97cdb16-Abstract-Conference.html) / [官方代码](https://github.com/richard-peng-xia/MMed-RAG) | 已包含 domain-aware retrieval、adaptive context selection 和针对错误检索的 preference training，是最强医学 RAG 对照 | paper-native full system；另报告 retrieval-only 与无 preference tuning 消融 | 先重新审计当前仓库的数据、retriever 和可用 checkpoint；不能完整复现时只报告明确命名的组件消融，不能写成 MMed-RAG |
| P0 | [Med-R1](https://arxiv.org/abs/2503.13939) / [官方代码与 checkpoint](https://github.com/Yuxiang-Lai117/Med-R1) | Survey 中最直接、已开放的多模态医学 reasoning-trained 模型，可检验现象是否只属于普通 instruction-tuned 医学 VLM | X-ray checkpoint 作为**模型边界对照**，不是 hallucination mitigation | 统一 prompt/解析通过，且图像处理与官方 processor 一致；先跑 32 例资格检查，再进入 CE/OE |
| P1 | [FactMM-RAG，NAACL 2025](https://aclanthology.org/2025.naacl-long.28/) / [官方代码](https://github.com/cxcscmu/FactMM-RAG) | 用 RadGraph factual pairs 训练报告检索与生成，是 report-RAG 的直接强基线 | 仅在 IU-Xray/MIMIC-CXR report 轨比较；同时保留 raw BM25/CLIP RAG | 官方仓库虽给出训练流程和 checkpoint 链接，仍须下载校验并完成原生推理复现；缺语料或 checkpoint 时 N/A |
| P1 | [RADAR，ACL 2025](https://aclanthology.org/2025.acl-long.1279/) / [官方代码](https://github.com/wjhou/Radar) | 先用 image expert 确认内部发现，再只检索补充知识，与“过滤其他患者状态、保留通用知识”高度接近 | 只在报告生成轨按 paper-native 模型比较；额外做 fixed-claim-coverage | 若新方法不进入报告生成，不占当前 GPU；进入报告主张后必须补 |
| P2 | [LVMed-R2](https://arxiv.org/abs/2504.02885) | Survey 中与报告自反思最接近，但属于训练式 report generator | 仅作 trained reflection 对照 | 未核验到稳定官方代码入口前保持 N/A；Med-REFL 是文本 MedQA 方法，不作为图像幻觉 baseline |

## 3. 新方法必须击败的 nearest work

Survey 的知识利用章节没有覆盖下列最近工作，但它们决定可发表的新颖性边界。

| 新方法准备声称什么 | 最接近工作 | 已经覆盖的机制 | 新方法仍必须证明的增量 |
|---|---|---|---|
| “防止其他患者检索报告把阳性/阴性带入当前回答” | [RULE](https://aclanthology.org/2024.emnlp-main.62/)、[MMed-RAG](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a559a5a8aa5ae6682ced009ad97cdb16-Abstract-Conference.html) | 错误/过量检索导致过度依赖；校准检索数、adaptive selection、preference alignment | 在**同一患者图像和问题、只交换 donor clinical state**的双向实验中证明 signed contamination；再在固定 coverage 下同时降低 FP/FN，且优于 calibrated top-k 和 adaptive selection |
| “把检索内容去极性化，只保留通用影像知识” | [RADAR](https://aclanthology.org/2025.acl-long.1279/)、[FactMM-RAG](https://aclanthology.org/2025.naacl-long.28/) | image-expert gating、supplementary knowledge injection、事实感知报告检索 | 证明被删除的是 patient-state 而不是有用知识；需要 raw-RAG、term-only、length-matched、state-shuffled 四个对照，并报告知识 claim 不退化 |
| “利用反事实证据判断检索是否可信” | [CF-RAG，ICLR 2026](https://openreview.net/forum?id=9U51rOnGko) / [代码](https://github.com/CF-RAG/CF-RAG) | counterfactual query、dialectical retrieval、evidence arbitration | 医学化本身不新；必须有 reader/claim-grounded 的新因果变量，并在同算力下优于 CF-RAG arbitration |
| “training-free 地验证/纠正图像 claim” | [CounterVHD](https://arxiv.org/abs/2606.28520) / [代码](https://github.com/Agentic-CliniAI/CounterVHD)、[CoEV](https://arxiv.org/abs/2606.18609) | claim/entity 抽取、反事实视觉 grounding、counter-evidence verification 和纠错 | 必须在检测 AUROC 之外证明真实生成端的 hallucination 降低，且 omission、claim 数和拒答不恶化；否则只是另一种 verifier |
| “报告检索提高事实性” | FactMM-RAG、RADAR、MMed-RAG | 三者已覆盖事实检索、补充知识和跨域 RAG | 至少在同一 MIMIC/IU split 上超过它们的 paper-native 或可复现最强组件；仅超过 BM25 不足 |
| “真实信息来自错误来源，普通 factuality 看不出” | [Ghost Context，TrustNLP 2026](https://aclanthology.org/2026.trustnlp-main.19/) | misattributed grounding、source-blind metric 不完备、mask-and-rerun causal attribution、Fix@k/CDR/CIR 与 post-hoc remediation | 必须证明 image-grounded patient owner × clinical predicate 的内部变量绑定及衰减，而不只是 wrong-span influence；mask/rerun 本身只作 baseline |
| “按生成 token 分解 RAG 信息来源” | [TPA，ACL 2026](https://aclanthology.org/2026.acl-long.1159/) | 将下一 token 概率归因给 query、RAG context、past/self token、FFN、LayerNorm 和 embedding | source-ownership 方法若做 component attribution，必须进一步细分同一 external context 内的 patient owner，而非重复 context-vs-FFN 分解 |
| “用文本/图像扰动评价医学 VLM 推理归因” | [Moll et al.，ML4H 2026](https://proceedings.mlr.press/v297/moll26a.html) | 受控多模态扰动评估 clinical fidelity、causal attribution 与 confidence；指出文本 cue 比视觉 cue 更易改变解释 | 当前四格不能只报告答案 margin；必须区分最终答案、解释归因和图像真值，并避免把模型复述 cue 当作 grounding |
| “恢复晚层 context/source routing” | [CoDA，ACL Findings 2026](https://aclanthology.org/2026.findings-acl.576/)、[Taming Knowledge Conflicts / JuICE，ICML 2025 Spotlight](https://openreview.net/forum?id=0cEZyhHEks) | context-selective routing 衰减、context/parametric conflict 的共享高影响 heads 与 test-time attention steering | 必须定位 patient-owner × predicate binding，而非泛称 context dominance；正式对照需包含 CoDA/JuICE-style steering |
| “对某一生成 statement 找关键 context 并剪除” | [ContextCite，NeurIPS 2024](https://openreview.net/forum?id=7CMNSqsZJt) / [官方代码](https://github.com/MadryLab/context-cite) | 黑箱、可扩展的 context attribution 与 pruning | span attribution、leave-one-out、mask/delete 都只能作 baseline；剩余增量需进入跨模态 token/path binding mechanism |

**最小 must-beat 集：** 若当前候选保持“source-typed polarity firewall”，主表至少需要 `no-context / raw RAG / RULE / MMed-RAG / firewall`；进入报告任务后再加入 `FactMM-RAG / RADAR`；使用反事实检索后再加入 `CF-RAG`。

## 4. 新增 benchmark 的条件式队列

| 优先级 | Benchmark（官方来源） | 修复的当前缺口 | 只有满足以下条件才加入 | 主指标 / 必须控制 |
|---|---|---|---|---|
| P0 | [MedHEval](https://github.com/Aofei-Chang/MedHEval) 完整 CE+OE | 从胸片二元 CE 扩展到 visual / knowledge / context 三种成因和开放回答 | 使用官方 split、prompt 和 task definition；图像权限完整；OE 自动 judge 只称 proxy | CE: BAcc、FP/FN；OE: claim precision/recall、长度、claim 数、拒答；按成因分层 |
| P0 | [MedVH v1.0.1](https://physionet.org/content/medvh/1.0.1/) / [官方代码](https://github.com/dongzizhu/MedVH) | 错图、缺正确选项、错误前提、诱导回答及长报告等 stress tests | 使用最新 v1.0.1；六个原始任务都能保持输入语义；两模型先过 nonempty/parse/template gate | 任务原生准确率；错误前提拒绝/纠正率；报告只报 benchmark-native 与自动 proxy，不宣称医生级真值 |
| P0（数据恢复后） | [MediConfusion，ICLR 2025](https://openreview.net/pdf?id=H9UnNgdq0g) / [官方代码](https://github.com/mshahabsepehri/mediconfusion) | 同一问题、两张图像、相反答案的最干净视觉依赖测试 | 352 cases 图像全部按官方 source 下载并通过 checksum；必须保留 176 个 pair，不拆成独立样本 | paired/set accuracy、两图同时正确率、rescue/harm；禁止只报单图平均 accuracy |
| P1 | [PadChest-GR](https://arxiv.org/abs/2411.05085) / [官方数据申请页](https://bimcv.cipf.es/bimcv-projects/padchest-gr/) | 把 finding 是否存在扩展到部位、属性和空间 grounding；含双 reader boxes | 数据申请获批；研究主张包含 location/attribute hallucination；模型能输出或映射 bbox | claim P/R、location/attribute error、IoU/pointing、双 reader 分别与并集敏感性；可参考 [CURE 代码](https://github.com/PabloMessina/CURE) 适配 |
| P1 | [ReXVQA](https://arxiv.org/abs/2506.04353) / [官方数据](https://huggingface.co/datasets/rajpurkarlab/ReXVQA) | 大规模胸片 OE/多类型 QA 与独立来源数据 | study-disjoint 子集；先做 text-only、null-image 和 image-swap，视觉增量不成立则不作为 image-grounding 证据 | task-native accuracy/F1 + image-use gap；按 question family 报告，不能让模板频率主导 |
| P1 | [MedXpertQA](https://openreview.net/pdf?id=IyVcxU0RKI) / [官方数据](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) | 增加非胸片、多模态专家推理边界 | 仅当论文主张跨模态/知识推理；只用公开 multimodal split，冻结 exact-option scorer | MCQ accuracy、模态分层；不得把 MCQ 错误直接称为视觉 hallucination |
| P2 | [EHRXQA，NeurIPS 2023](https://proceedings.nips.cc/paper_files/paper/2023/file/0c007ebef1d11fd48da6ce4f54687db6-Paper-Datasets_and_Benchmarks.pdf) / [PhysioNet](https://physionet.org/content/ehrxqa/1.0.0/) / [代码](https://github.com/baeseongsu/mimic-cxr-vqa) | 区分 image、table、image+table 的证据来源，适合 source-erasure 问题 | 只有研究问题升级为“证据来源身份”；完成 MIMIC 权限、患者链接和泄漏审计 | 三种 source task accuracy、逐源 ablation、错误来源矩阵；不能与纯视觉 hallucination 合并平均 |
| P2 | [ReXGroundingCT](https://arxiv.org/abs/2507.22030) / [官方代码数据](https://github.com/rajpurkarlab/ReXGroundingCT) | 从 2D CXR 扩展到 3D CT 自由文本 finding grounding | 至少一个主模型原生支持 3D/多切片输入，且已有正向核心结果；否则工程变量大于科学增量 | Global Dice/HIT、instance precision/recall；单独成轨 |
| P2 | [PathMMU](https://arxiv.org/abs/2401.16355) / [官方 benchmark](https://pathmmu-benchmark.github.io/) | 病理跨模态边界 | 只有主张跨影像模态；模型支持病理 patch，且 text-only shortcut 已审计 | MCQ accuracy、image-minus-text gain、corruption robustness；不是 report hallucination 主指标 |

### 暂不加入

- `DeepTumorVQA`、`U-MRG-14K`、超声、ECG/EEG、手术视频：在核心机制未通过两模型两任务前，它们只增加适配成本，不能修复当前最关键的 CE→OE 与 RAG 因果缺口。
- `Med-REFL`、MDAgents、TxGemma 等文本/agent 系统：任务对象不是 frozen medical VLM 的 image-grounded claim hallucination，不作为必跑 baseline。
- BLEU、ROUGE、单一 LLM judge：不能单独定义 hallucination 真值。

## 5. 最短执行顺序

1. **不新增数据先补 nearest work：** 资格审计 RULE、MMed-RAG、Med-R1；分别输出 `paper-native / component-only / N/A`。
2. **补现有评测的宽度：** 接入 MedHEval 完整 OE 与 MedVH；先做每模型 32 例输出质量门，再扩到正式样本。
3. **恢复最干净的配对视觉测试：** 修复 MediConfusion 官方图像下载后，以 pair 为单位接入。
4. **仅在报告主线存活后：** 申请 PadChest-GR，并接 FactMM-RAG、RADAR、CURE evaluator。
5. **只有产生跨来源机制时：** 再启用 EHRXQA；只有产生跨模态主张时，再启用 MedXpertQA、PathMMU 或 ReXGroundingCT。

**停止规则：** 新方法若不能在 `raw RAG + RULE/MMed-RAG` 上证明固定 coverage 的增益，或增益只来自缩短、拒答、统一阴性/hedge，则不再用更多 benchmark 包装；直接降为诊断性现象。

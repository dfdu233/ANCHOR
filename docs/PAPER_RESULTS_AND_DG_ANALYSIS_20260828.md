# Baseline 结果图与 DG 风格迁移分析（2026-08-28）

## 1. 论文可用的 Baseline 图

结果来自统一覆盖审计 `corrected_runs/paper_baselines_v1/full_matrix_v1/coverage_audit.json`。图中严格区分：

- `OK`：生成、qualification 和评分均存在；
- `N/A`：协议上不适用、方法 gate 失败，或没有论文定义的数值指标；
- `PEND`：尚未完成；
- `PART`：生成/评分不完整。

未测试格子没有填 0，也没有被当作失败。

- [Figure 1: Baseline coverage/status](../corrected_runs/paper_baselines_v1/full_matrix_v1/paper_figures_v1/figure_baseline_coverage.png)
- [Figure 2: Baseline primary score](../corrected_runs/paper_baselines_v1/full_matrix_v1/paper_figures_v1/figure_baseline_primary_score.png)
- [Machine-readable result table](../corrected_runs/paper_baselines_v1/full_matrix_v1/paper_figures_v1/paper_results.json)

当前审计快照：

| Track | Cells | Completed | Running/partial | Pending | N/A |
|---|---:|---:|---:|---:|---:|
| training-free / trained matrix | 336 | 82 | 2 | 8 | 244 |
| auxiliary controls | 40 | 16 | 1 | 8 | 15 |

主矩阵的 82 个 completed 是论文结果候选；`PART/PEND` 不进入主表；`N/A` 在图中保留，以便审稿人看到覆盖边界。

## 2. 已尝试的方法总表

### 2.1 必须保留的 Baseline/Comparator

| 家族 | 方法 | 当前用途/结论 |
|---|---|---|
| Native decoding | greedy, beam | 必须报告的原始工作点；beam 可能只改变 precision/recall 平衡 |
| Visual contrast | VCD | 医学迁移的重要负/对照基线；当前严格质量 gate 仍需完成 |
| Layer contrast | DoLa | 与早层证据假设对照；跨任务无稳定增益 |
| Self-logit / visual steering | VISTA, OPERA, PAI, AvisC | 官方/社区常用缓解基线；逐模型逐数据集报告，不能混用评分口径 |
| Local grounding | SECOND | 当前 LLaVA-Med backend 不兼容，按 N/A 记录，不能写成算法失败 |
| Retrieval | shared-medical-RAG, no-context, shuffled-context, image-swap | 用于区分病例证据、问题先验和检索文本极性 |
| Trained LLaVA-1.5 | base, DA-DPO, HA-DPO, FactMM-generator, Less-is-More, OPA-DPO, SENTINEL, VHR | 适用性/权重/官方入口逐项审计；不适用项标 N/A |

### 2.2 DG / 风格迁移路线

| 方向 | 具体尝试 | 结果 |
|---|---|---|
| Fourier/FedDG | low-frequency interpolation、source-ratio mixing、SGTA source views | 视图可生成，但结构与临床证据未分离；selected style 在 128 cases 上比 native 低 2.34pp |
| Source center | dataset/source mean、frequency center、FedDG center | source identity 可识别，但不能稳定预测 hallucination；CT/CXR center provenance 未对齐 |
| Full-spectrum residual | release2/source-spectrum alignment | 参数名与 low-frequency implementation 语义不一致，导致实际变换强度不可解释 |
| Cosmetic controls | gamma、DICOM render、multi-style response fingerprint | 大多只改变低 margin 样本；未证明存在共同 source direction |
| Decoder DG | KL barycenter、style-view NLL selection、anchor consistency、strict-then-NLL、switch gate、OE risk ordering | 可能改善 NLL 或阳性率，但没有稳定降低 hallucination；容易发生 operating-point shift |

### 2.3 视觉证据与局部结构路线

| 方向 | 结果 |
|---|---|
| Evidence addressability / hidden-state summary | 视觉信息存在，但最终 claim 增量小且跨模型不稳定，双模型 gate 关闭 |
| CEB / visual-edge label-free score | pair AUROC 0.542–0.646，未过 0.70 门；简单 entropy/margin/support 不能直接部署 |
| Sparse lesion boundary | 小病灶与较弱正确支持在两模型复现，是可信现象；但 spatial scan 增量仅 +0.004 AUROC |
| Lesion delete-relocate | 删除/搬运不满足守恒律；编辑响应混有位置先验与伪证据 |
| Anatomy-conditional null / context completion | 当前数据缺 ViewPosition 等关键元数据；crop/context 主要造成 criterion shift |
| Patch relational/evidence capacity | 邻接共变、容量匹配、局部 pooling 未产生稳定病例级增量 |

### 2.4 输出、语言和评测路线

| 方向 | 结果 |
|---|---|
| LET / intermediate-layer transport | 历史 CE 增益伴随 FN→FP 工作点漂移；不能当作通用医学幻觉方法 |
| VISTA/LET window comparison | 主要是 operator collision 和 operating-point difference，不能声称 VISTA 失败 |
| Prompt rewrite / criterion audit | 同一输出在不同解析规则下会改变方法排名；已确认为必须控制的评测原则 |
| RAG response code / Fisher selector | 有局部竞赛增益，但受 question prior、成员强弱和相关性影响，不是独立机制 |
| Specialist/XRV expert | 能补充 Huatuo 的部分病例信息，但跨模型迁移不稳定；监督 stacking 上限，不是通用无训练算法 |
| Claim exchange / one-bit veto | 可降低部分 FP，但同时误伤 TP 或增加总体错误 |

完整候选注册和每个候选的 GO/NO-GO 证据见 [candidate_registry_v0.md](daylong_idea_search/candidate_registry_v0.md) 与 [METHOD_ZOO.md](METHOD_ZOO.md)。

## 3. 为什么当前 DG 结果不好：相信 DG，但修正流程

当前结果不应解释为“DG 没用”。更准确的解释是：DG 的域变量、临床证据变量和 decoder 语言先验在当前流程中被混在一起，导致域对齐没有作用到真正的 hallucination mechanism。

### 3.1 变换定义层：同名参数不是同一操作

`methods.py` 的 `low_frequency_ratio` 表示低频窗口半径，`source_ratio` 表示窗口内插值；而 `frequency_alignment_source_spectrum.py` / `release2.py` 将同名参数当作全频谱 residual alpha，并且强制 `source_ratio=0`。因此实验名写的是“低频 DG”，实现可能执行的是“全频幅度混合”。这会直接破坏可解释性和超参数可迁移性。

### 3.2 Source center 层：域中心没有与成像机制对齐

当前默认 bank 路径为 `PubMedVision/train/ct__chest.npy`，目标任务是 CXR。即使它能提供可用的频谱先验，也同时携带 modality、重建和导出差异。DG 需要对齐的是可迁移的 acquisition nuisance，而不是把 CT source statistics 当成 CXR source truth。

### 3.3 Evidence preservation 层：PSNR/edge 不是临床等价

已有 128-case 可视化显示：

- LF median PSNR≈19、edge≈0.996；
- source-ratio 0/0.5 的 median PSNR 约 5/11、edge 约 0.73/0.97；
- source-ratio 0.8 的 edge≈0.997，但 PSNR 仍约 18.8。

高 edge 只说明边缘排列相似，不保证病灶对比度、设备线索、心影测量或外部反证保持。当前结构 gate 因而可能把 evidence-destroying view 当作“安全风格”。

### 3.4 Selector 层：NLL/一致性选择放大 winner's curse

在多个 style view 中选择最低 NLL 或最一致的候选，相当于先搜索再验证；候选越多，越容易挑到偶然有利的 view。选择结果还受到输出长度、模板和阳性率影响，而不是只受到视觉 evidence 影响。

### 3.5 Decoder 层：全句重新生成，没有 native anchor

当前 view-based DG 让模型在变换图上完整重新生成一份答案。任何 token 都可能变化，无法区分：

1. visual evidence 真正改变；
2. 语言历史导致的 claim commitment 改变；
3. EOS、长度和模板改变。

因此即便 DG 变换能纠正某个 claim，也可能同时引入新的 unsupported claims。

### 3.6 评测层：工作点变化被误报为 hallucination reduction

已有 LET、crop、style 和多种 decoding 实验反复显示：FP 降低可能只是 recall 下降，或者反之。DG 必须在固定 coverage、固定 claim 数 K、相同解析规则和 paired bootstrap 下报告，不能只比较总体 accuracy 或 positive rate。

## 4. DG 的修正版流程

保持 DG 主张不变，但把流程改成五个独立模块：

```text
modality-matched source center
          ↓
explicit transform mask / amplitude audit
          ↓
claim-specific evidence preservation
          ↓
native-anchored decoder delta
          ↓
fixed-coverage clinical evaluation
```

具体修复：

1. 使用 CXR/CXR、同 projection、同 preprocessing 的 source center，并保存 provenance；
2. 明确区分 low-frequency window、full-spectrum residual 和 gamma，不再复用含义冲突的参数名；
3. 在 PSNR/edge 之外加入 claim-specific evidence survival 和 pathology-preserving controls；
4. 不让 DG view 直接接管整句生成，只对 `DG_interaction` 为正的 claim token 做 native-anchored calibration；
5. 预注册 view 数量、selector、coverage、长度和拒答规则；
6. 用可视化 dashboard 检查每个 claim 是被 evidence 支持、被 style 推动，还是被 language history 推动。

## 5. 下一步计划

### P0：Baseline 并行收口

继续现有 GPU lock 队列，不为 DG 重新分配 GPU。所有主矩阵格子以 coverage audit 为完成真相。

### P1：已有数据的 DG 直接对比

当前已启动 `dg_paired_validation_v1`，等待 Baseline 释放共享 GPU lock。它复用已完成的 Huatuo `visual_mimic_oe/greedy` native answers，先做 64 个同图像/同问题/同 seed 的 FEDDG paired canary。

### P2：四格 interaction 验证

对同一 claim 构造：

```text
原始域 / DG 域 × 证据保留 / 证据削弱
```

计算：

```text
DG_interaction = (B - A) - (D - C)
```

只有 interaction 在两个模型、两个数据集上可复现，且不影响 control claims，才继续做推理时 calibration。

### P3：修正 source center 后的消融

在 canary 通过后，仅比较：

- CXR-matched center；
- 当前 CT-named center；
- 无 source center 的 identity control。

这一步用来定位当前负结果究竟来自 center provenance 还是 decoder binding，而不是继续扩大 style bank。

### P4：论文决策门

进入主方法需要同时满足：

- hallucination risk AUROC 增量 ≥0.03；
- paired bootstrap CI 下界 >0；
- 两个模型复现；
- geometry-insensitive control claim 不同步漂移；
- clear-case accuracy 下降不超过 1pp；
- fixed coverage 下 FP 降低且 FN 不显著增加。

如果未通过，DG 不被否定，而是保留为可靠的 mechanism/diagnostic 模块，并继续修复 evidence preservation 和 decoder binding，而不是继续调风格强度。

## 6. LaTeX/PDF 交付

已将 Baseline 覆盖图、primary score 图、N/A 规则和代表性结果编译为论文可引用的 PDF：

- [paper_results.pdf](../corrected_runs/paper_baselines_v1/full_matrix_v1/paper_figures_v1/paper_results.pdf)
- [paper_results.tex](../corrected_runs/paper_baselines_v1/full_matrix_v1/paper_figures_v1/paper_results.tex)

PDF 采用 landscape appendix figure，覆盖矩阵拆成两页，避免 48 行压缩到不可读；未测试格子统一显示 N/A。

## 7. 已完成的 DG CPU 问题探索

新增 `anchor/corrected_sgta/analyze_dg_failure_modes_v1.py`，并对现有 128-case style phenomenon 与 FEDDG raw generations 运行。机器可读结果：

- [style failure modes](../corrected_runs/style_phenomenon/huatuo_rule_mimic_n128_v1/dg_visual_audit/failure_modes.json)
- [FEDD-G failure modes](../corrected_runs/huatuo_rule_mimic_feddg/stage_n128_reselect_v3_switch/dg_visual_audit/failure_modes.json)

这一步确认当前负结果可由流程错误解释，而非 DG 假设被否定：

1. 低频参数在不同实现中含义不同，实验名与实际频谱 mask 不一致；
2. 默认 source bank 的命名为 `ct__chest`，尚未证明与 CXR acquisition mechanism 对齐；
3. edge correlation 高但 PSNR/幅度变化仍大，结构 gate 不能充当临床证据 gate；
4. flip 样本的 native margin 约为 0.02--0.05，而稳定样本约为 0.59--0.61，说明变换主要触发边界敏感性；
5. 当前候选选择与整句重生成没有 native anchor，容易把 DG 效果和长度、EOS、语言先验混在一起。

下一轮只验证修正后的模块链：CXR-matched center → 明确低频 mask → claim-specific evidence survival → native-anchored decoder delta → fixed-coverage evaluation。

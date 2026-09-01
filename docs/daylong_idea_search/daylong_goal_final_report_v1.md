# 全天候医学 VLM 幻觉研究冲刺：最终证据报告 v1

日期：2026-08-12。本文区分“实验已确认”“探索性信号”和“论文叙事”，不把响应变化当作临床证据，也不把未通过门槛的候选重新命名。

## 1. 最终裁决

**本轮没有发现一个已经达到 ICLR Oral 水平的 hallucination mitigation 算法。**

这不是因为候选数量不足：共登记 24 个机制不同的候选，完成超过 6 个低成本证伪、6 个真实模型/真实数据致死实验，并对有表面正信号的方向做了强基线、配对 bootstrap、竞争机制和文献碰撞。没有候选同时满足：病例级增量信息、降低 FP、不增加 FN、跨模型复现、且不退化为工作点移动或已有 global–local/crop/calibration 方法。

当前最可信、最有社区价值的结论是：

> **Response is not evidence.** 医学 VLM 对 mask、crop、prompt、检索或层间干预有明显响应，并不表示干预提供了新的病例级临床证据；大量表面“缓解”只是改变回答标准、阳性率、长度或拒答率。

这个结论足以支撑一篇严格的机制/评测论文骨架，但目前仍不能诚实称为 Oral-ready 的新算法论文。

## 2. 量化完成情况

| Goal 项 | 完成情况 |
|---|---:|
| 机制不同候选 | 24 个，目标至少 12 个 |
| 低成本证伪/等价性审计 | 超过 10 个，目标至少 6 个 |
| 真实模型/真实数据致死实验 | 6 个，目标 3–5 个 |
| 通过正式多模型放量门的候选 | 0 个 |
| Baseline 保护 | 五个持久队列已恢复；生成按 64 例 chunk 落盘 |
| 可审计产物 | 候选 registry、实时 scorecard、脚本、原始分数、bootstrap 结果、碰撞文档齐备 |

## 3. 已经可信的实验结论

### 3.1 小病灶获得更弱支持，但简单局部搜索不能修复

- VinDr 3/3 阳性 bbox 病例，开发集每模型 480 claims、fresh 确认每模型 133 claims。
- 病种内病灶面积与正确 margin 的 Spearman：Huatuo `0.323`、Hulu `0.415`；控制位置、局部对比、纹理、框数量、碎片化和 reader dispersion 后仍为 `0.239` 与 `0.420`，CI 均排除 0。
- 但面积预测 miss 的增量只在 Huatuo 通过；Hulu 增量 AUROC 仅 `+0.015`。
- multiscale sparse scan 相对 `final+mean+max+top5` 强基线只增加 AUROC `+0.0040`，95% CI `[-0.0194,+0.0273]`。

结论：**稀疏病灶边界是可信现象，不是已经可利用的通用修复信号。**

### 3.2 局部 crop 会制造巨大阳性漂移，主因不是放大尺度

Huatuo、62 张七 finding 均为 0/3 的阴性胸片：

| 输入 | FP rate |
|---|---:|
| 完整胸片 | 8.1% |
| ROI 保持原位置和尺度，仅模糊外部 | 71.0% |
| 原尺度 ROI + sham context | 64.5% |
| 放大 ROI + sham context | 67.7% |
| 放大 ROI + true context | 58.1% |

- 移除外部 context：FP `+62.9pp`，95% CI `[+50.0,+74.2]pp`。
- 只改变 native→zoom scale：FP `+3.2pp`，95% CI `[-6.5,+11.3]pp`。
- true context 相对 sham context 救回 `9.7pp` FP，CI `[3.2,17.7]pp`。

但 positive ROI 来自专家 bbox，negative ROI 来自随机网格，因此阳性 recall 与阴性 FP 不能当作同一部署策略的公平 ROC。可信主张只到：**移除完整解剖上下文会使模型强烈向 Yes 漂移。**

### 3.3 “观察策略是 prompt”被同像素实验否定

124 张图、5 种 render、3 种 provenance prompt，共 1,860 个 Huatuo 单步评分；每种 render 的三种 prompt 复用 bit-identical PNG，并核验 SHA。

- “该 crop 是随机产生，选择不是证据”令 crop 阴性 FP 从 `67.7%` 降至 `6.5%`。
- 但阳性 recall 同时从 `91.9%` 降至 `32.3%`，损失 `59.7pp`。
- crop 特异 interaction `Gamma=-0.117`，95% CI `[-0.188,-0.042]`，方向与预注册假设相反。
- full image FP 也下降 `4.84pp`。

结论：它是**全局保守化/criterion shift**，不是去除了 observation-policy evidence。按门槛不扩 Hulu，Endogenous Evidence Recycling 与 Policy-Cut Update 均关闭。

### 3.4 真实上下文响应有标签关联，但无病例级增量

- `delta_ctx = margin(true-context panel) - margin(sham-context panel)` 单独 AUROC `0.6574`，CI `[0.5797,0.7340]`。
- 强基线 `full + crop + finding` AUROC `0.8548`；加入 `delta_ctx` 后为 `0.8431`。
- 增量 `-0.0117`，CI `[-0.0330,+0.0044]`；NLL、Brier 也变差。
- standalone delta 在 7/7 findings 高于随机，但强基线上只有 2/7 改善。

结论：**真实 context 会改变响应与校准，却没有提供超出 full/crop 已有分数的新排序信息。**

### 3.5 搜索、池化与第二观察均未形成可用方法

| 实验 | 关键结果 | 裁决 |
|---|---|---|
| Lesion delete–relocate，n=128 | 删除效应 `-0.025`，CI 跨 0；搬运虽增分但过冲，joint direction 34.4% | 局部 evidence transport 关闭 |
| Raw region search | p95 随 1→7 claims 增 `+1.137`，CI `[.517,1.388]` | 仅内部极值现象 |
| End-to-end selection reuse | 最大搜索 selected-random FP `+17.7pp`，但 region 16→361 gap 墰 CI 跨 0 | 搜索相变关闭 |
| Evidence-conserving e-mixture | null 对分区稳定；强基线 AUROC 仅 `+0.0016`，NLL 显著变差 | 算法关闭 |
| IU-Xray 第二真实 view，n=64 | accuracy `+1.56pp`，CI `[-4.69,7.81]`; Brier CI 跨 0 | 不扩模型/数据 |
| Sparse patch scan | AUROC `+0.0040`，CI 跨 0 | 不扩 Hulu |

## 4. 数学上保留什么、不能声称什么

### 4.1 局部见证与全局证书不对称

对 OR 型局灶 finding，把图分成 `M` 个区域。一个病灶区域即可证明阳性，所以阳性 certificate complexity 为 1；要证明阴性则需排除所有区域，所以阴性复杂度为 `M`。

任取不完整 crop，都可构造两张 crop 完全相同但图外真值相反的完整图。任意随机二元规则在该配对上满足：

\[
\mathrm{FPR}+\mathrm{FNR}=1.
\]

这说明 crop-only 方法不可能在所有 completion 上同时降低 FP 与 FN。该结论是标准 certificate/MIL/partial-observation 逻辑在本问题中的清晰表达，不是全新的数学定理。

它只允许三种出路：看完整图或新观察、加入分布假设、输出 Unknown。前两者不是从 crop 免费创造证据，后者是 abstention。因此它不能自然推出所需的非拒答缓解算法。

### 4.2 信息守恒不等于计算可用性

crop 是 full image 的确定性变换，按 data processing/Blackwell order 不可能比 full image 含有更多世界信息。固定 VLM 在 crop 上 recall 上升，只能来自有限 token、注意力或工作点改变，而不是临床信息增加。任何新方法若只展示“模型更响应 crop”，都必须证明病例排序增量并排除阳性率漂移。

在同一批阴性病例中，full→native-context-removed 的 AUROC 仅从 `0.7946` 到 `0.7980`，增量 `+0.0035`，95% CI `[-0.0623,+0.0730]`，但阈值 0 下 FP 增加 `62.9pp`。这正是“排序几乎没变、工作点剧烈移动”，而不是 crop 创造信息。把它写成 bounded-rationality/Blackwell paradox 也不产生新算法：有限模型的表示并不保证保留 Blackwell 序，这是标准 restricted decision-rule 现象；自然修复仍退化为已有 global–local fusion。

### 4.3 不能再当贡献的数学包装

- 双中心化/DID：已覆盖 NCD、ISD、CMEI，且无增量信号。
- Poisson hazard/noisy-OR：等价经典 MIL；线性版退化为 mean，校准版退化为已失败 e-mixture。
- Bayes action-likelihood subtraction：PCU 的识别式标准，且其必要现象已失败。
- 凸层融合与正交投影：标准几何；候选层没有正确极性时不能凭凸组合修复。

## 5. Top-3 研究资产，而非 Top-3 新算法

### Top 1：Mitigation Mirage / Response Is Not Evidence

证据最强。相同输出在 strict 与 official 口径中，10 个方法对有 5 对排序翻转；LLaVA CXR 复核中 15 对有 7 对翻转，invalid rate 与额外得分 Spearman `0.943`。crop prompt、context、RAG placebo、head suppression 与 LET 也反复出现“响应强但无病例级增量”。

适合定位：**New evaluation/mechanism problem**。尚缺完整跨模型 baseline 表和连续 score/ROC 轨迹，当前更像有竞争力的主会论文骨架，不是 Oral-ready 算法。

### Top 2：Sparse Lesion Boundary

自然现象跨两模型、开发与 fresh confirmation、测量 confound 后成立。它说明医学 VLM 的错误与小面积病灶有系统关系，但简单 scan、HC、e-mixture 和面积 gate 都失败。

适合定位：机制发现或未来方法的 admission condition；当前不能声称 mitigation。

### Top 3：Recoverability / Addressability Boundary

answer-position、中间层、风格/DG、权重方向、全局视觉池化、patch scan 等接口反复呈现“可响应/可解码，但不在 final margin 之上提供可利用增量”。这可形成 training-free decoding 能力边界的系统负结果。

适合定位：负结果与统一审计框架；需要把方法 correction rate 与内部 oracle 上限跨模型、跨任务关联，才能达到强主会水准。

## 6. 推荐论文主线与停止边界

当前唯一证据充分的主线不是再发明一个 decoding 名字，而是：

> **When Mitigation Is Not Evidence: Auditing Criterion Shift in Medical Vision–Language Models.**

论文类型应明确为 New Problem/Setting + mechanism/evaluation，而非 Technique：

1. 定义 intervention response、case discrimination、criterion/coverage 三个必须分开的量；
2. 证明现有方法排名会随解析和评分规则反转；
3. 用 matched coverage/ROC/连续 score 检查“提升”是否只是移动工作点；
4. 用 crop、prompt、RAG、layer/head intervention 作为多类证据链；
5. 给出 fail-closed 的 mitigation admission protocol，而不是另一个启发式校正。

它可能成为可信 ICLR 主会投稿，但要达到 Oral 仍需：至少三架构、CE/OE/report 两类以上任务、完整官方 baseline、统一连续分数、一个能预测方法何时失败的跨方法规律。当前不能保证 Oral。

停止边界如下：

- 不再复活风格/DG、NCD/ISD/CMEI、普通层融合、全局视觉 probe、crop max、e-mixture、observation-policy prompt、PCU、stochastic resonance、spatial hazard。
- 不因单一 AUROC、响应幅度或显著 p 值放量；必须在强基线之上有 fresh、病例级增量且 CI 排除 0。
- 任一“降低 hallucination”结果若伴随输出更短、Yes-rate下降、拒答上升或阳性 recall下降，先归入 criterion shift，不能称 mitigation。

## 7. Baseline 当前状态

新方法实验只在完整 64 例 chunk 边界短暂停队。Huatuo 致死实验结束后，`baseline_matrix_v1`、`baseline_llava_methods_v2`、`baseline_cross_methods_v3`、`baseline_shared_rag_v1`、`baseline_vhr_full` 已全部由退出钩子恢复。

LLaVA-Med Visual-MIMIC OE 的 VCD 生成暴露一个真实质量问题：部分 chunk 有空输出，因此被统一质量门标为失败并重跑，而不是把不完整结果写入主表。该问题需要 baseline session 按既定规则继续处理；不能通过放宽门槛伪造完成度。

## 8. 主要产物

- `docs/daylong_idea_search/candidate_registry_v0.md`
- `docs/daylong_idea_search/live_scorecard_v1.md`
- `docs/daylong_idea_search/observation_policy_pragmatics_v1.md`
- `docs/daylong_idea_search/context_completion_signal_v1.md`
- `docs/daylong_idea_search/witness_certificate_logic_v1.md`
- `docs/daylong_idea_search/endogenous_observation_oral_story_v1.md`
- `anchor/corrected_sgta/observation_policy_probe_v1.py`
- `anchor/corrected_sgta/analyze_context_completion_signal_v1.py`
- `corrected_runs/daylong_idea_search_v1/observation_policy_huatuo_v1/analysis.json`
- `corrected_runs/daylong_idea_search_v1/context_completion_signal_huatuo_v1.json`

最终建议：**停止为“必须今天产出一个漂亮算法名”而继续拟合故事；保留所有负结果，把 baseline 跑完，用 Response–Evidence Separation 统一重评现有方法。若未来出现一个 fresh、跨模型的病例级增量信号，再由机制自然推出方法。**

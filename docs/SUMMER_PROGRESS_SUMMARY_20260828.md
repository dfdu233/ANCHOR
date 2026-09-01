# 暑期医学 VLM 幻觉项目阶段总结

日期：2026-08-28

## 一句话结论

暑期工作的主要产出不是已经完成一个最终算法，而是完成了一个较完整的**机制筛选与评测基础设施**：确认医学 VLM 幻觉与视觉证据不足、低 margin、输出历史和域/提示交互有关；同时严格关闭了多个看似有效但不可解释、不可复现或与已有工作碰撞的方向。当前 Baseline 仍在收口，最有希望的论文方向是“视觉证据预算/跨模态 claim binding”，DG/style 则保留为可解释的干预与诊断框架。

## 1. 暑期主要尝试

### 1.1 Baseline 与评测体系

建立了统一的 model × method × dataset 矩阵，覆盖 Huatuo、Hulu、LLaVA、Qwen 等模型，以及 VQA、开放式 VQA、报告生成和 shared-RAG 任务；复核了 VCD、DoLa、OPERA、PAI、AvisC、VISTA、VHR 等方法。

同时补齐了：

- generation qualification、token identity、method activation 和 checkpoint provenance；
- dataset-native 指标、FP/FN、coverage、长度、拒答和 clinical contradiction 审计；
- patient/image-cluster bootstrap；
- generated-but-unscored、partial、pending、N/A 的覆盖真相。

当前最新审计：主矩阵 336 格中 82 completed、3 running/partial、8 pending、243 N/A；辅助控制 40 格中 16 completed、1 running/partial、8 pending、15 N/A。适用主格子的完成比例约 88%。

### 1.2 传统 DG / style migration

尝试过 pixel/频域/vision/decoder source probe、低频替换、FedDG、source center、DICOM render、gamma/window 变换、多风格响应 ensemble，以及 style-aware decoding。

可靠结论是：source/domain identity 可以被强烈识别，但 source identity 不能稳定预测 hallucination；style flip 主要落在低 native-margin 样本；多风格 fingerprint 没有超过 canonical image。因此传统“把所有域拉到共同中心”的方法还没有构成可靠缓解。

这不是 DG 假设被否定，而是当前对齐对象不干净：style 混合了 modality、采集协议、显示变换、病理证据和报告先验。更严重的是，部分实现对 `low_frequency_ratio` 的语义不同：有的表示低频窗口半径，有的实际上是全频谱 residual alpha；默认 source bank 的路径还标为 `ct__chest`，与目标 CXR 的 modality/provenance 需要重新核验。

### 1.3 视觉证据与局部病灶方向

尝试了 lesion deletion/relocation、局部 patch、head suppression、视觉依赖 subset、稀疏病灶边界和局部方向搬运。

较稳定的现象是：小病灶或视觉证据稀疏时，模型的支持更容易被全图表征稀释；但“把局部高响应搬到另一张图”没有满足证据守恒，说明 attention/high response 不能直接当作临床证据。该方向可作为视觉约束不足的现象分析，不宜直接包装为 patch 算法。

### 1.4 Evidence addressability / Causal Evidence Budgeting

研究了 Full、−V（阻断视觉边）、−H（阻断输出历史边）三类前向路径，试图测量视觉支持 `V` 与历史自支持 `H`，并定义 `max(0, H-V)` 形式的证据赤字。

强的配对/真值依赖信号可以区分错误，但简单无标签分数没有达到部署门槛：selected-support pair AUROC 约 0.542，候选 span、history persistence 等最高约 0.646，预设门槛为 0.70。因此当前结论是：视觉约束潜变量很可能存在，但不能直接用单个 entropy/margin/support 分数判断。

### 1.5 RAG、响应码与组合策略

plain/RAG、跨患者 response、模型投票、探针 stacking、Fisher basis、CMP/router 等均被测试。

这些实验得到过明显的竞赛增益，例如 plain+RAG 响应组合在部分 CXR 数据上提升 BAcc；但正确 placebo 后，部分“患者对齐”解释消失，增益更像问题模板先验、响应模式或普通 stacking。它们目前是竞赛工具与机制探测仪器，不是论文核心算法。

### 1.6 Prompt 改写、一致性、多视图与反事实

测试了 prompt paraphrase、multi-view/style consistency、输出历史 masking、null/swap/replacement、DICOM render 等。

这些操作能暴露模型不稳定性，但“输出改变”本身不能证明原答案是幻觉：有些变换会改变病理可见性或测量律，有些只改变回答格式。因此后续必须区分合法的条件性变化、普通低 margin 边界和语言先验越界。

## 2. 当前最重要的科学认识

项目逐渐从“找一个有效变换”收紧为三个层次：

```text
图像/采集因素 → 视觉表示 → 临床证据 → claim commitment
                                      ↑
                              prompt / history prior
```

目前最可信的研究问题是：当视觉证据不足时，语言先验是否跨过了视觉证据边界；DG 的作用不是简单消除所有域差异，而是识别哪些域/视觉因素与错误 claim 发生了非加性 interaction。

对 DG，下一步使用四格反事实：

```text
                 原始域       DG 域
证据保留          A             B
证据削弱          C             D
```

并计算：

```text
DG_interaction = (B-A) - (D-C)
```

只有该 interaction 在跨模型、跨任务上稳定预测 hallucination，且不影响 control claims，才进入推理时 claim calibration。

## 3. 是否来得及 ICLR 2027

ICLR 2027 官方 abstract deadline 是 **2026-09-18 AOE**，paper deadline 是 **2026-09-25 AOE**（[官方时间表](https://iclr.cc/Conferences/2027/Dates)）。从 8 月 28 日算，距离 abstract 约三周，距离 paper 约四周。

### 诚实判断

- **按“Baseline + 现象论文”提交：来得及，但竞争力不足。** 目前结果更像严谨的机制筛选、负结果和评测协议，尚未形成一个跨模型有效的新算法。
- **按完整 ICLR 方法论文提交：时间非常紧，但不是绝对来不及。** 必须在 7–10 天内完成一条单一主线的机制 gate 和最小缓解实验，不能再扩展候选方向。
- **如果 9 月第一周仍没有跨模型、跨任务的正向 interaction/mitigation 结果，应停止追求“硬凑 ICLR 主论文”，转为 workshop、后续投稿或以高质量负结果/benchmark 论文为目标。**

## 4. 接下来四周规划

### 8/28–8/31：Baseline 与可视化

1. 完成当前 LLaVA VCD/native/shared-RAG 队列。
2. 收口所有 qualification、评分和 N/A 证据。
3. 使用新增的 `visualize_dg_alignment_v1.py` 生成 style/domain 诊断图。
4. 不新增 style bank，不改变 Baseline decoding。

### 9/1–9/5：最小 DG interaction canary

1. 仅选两个模型、两个数据集和一个可解释变换。
2. 对每个 claim 计算 A/B/C/D 四格以及 `DG_interaction`。
3. 记录 visual support、history support、native margin、style drift 和 claim correctness。
4. 对比 entropy、prompt consistency、random transform 和 pixel-distance-matched control。

### 9/6：Go/No-Go 决策

进入主论文必须同时满足：

- interaction 对 hallucination 的 AUROC 增量至少 0.03；
- bootstrap CI 下界大于 0；
- 至少两个模型同向；
- control claims 不同步变化；
- 不以降低 coverage、变短或增加拒答换取表面收益。

否则关闭 DG mitigation，保留其诊断价值，转向 CEB 的简化版本或 PCEM（需要真正的 AP/PA paired 数据和独立 heart-size truth）。

### 9/7–9/15：论文收口

只保留一条主线：问题定义、interaction/证据预算机制、推理时算法、Baseline 对照、跨模型验证、失败边界和可视化。9 月 18 日前冻结标题、摘要、作者和主结果；9 月 25 日只做写作与复核，不再开新实验。

## 最终建议

当前最合理的策略不是继续寻找更多变换，而是把暑期积累转化为一条清晰论证：

> 医学 VLM 的幻觉并非单纯 domain shift；真正危险的是当视觉约束不足时，域/提示/输出历史的交互改变了 claim commitment。DG 负责暴露并分解这种交互，推理时方法只校准越界 claim。

这条路线仍有 ICLR 潜力，但必须在 9 月第一周前获得跨模型的正向机制证据；否则应果断降级为严谨的评测与机制论文，而不是继续堆叠新模块。


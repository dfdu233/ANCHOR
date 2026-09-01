# 医学小模型作为“测量仪器”的碰撞与数学审计

> 日期：2026-08-13  
> 结论：**问题重要，系统实用，但当前方案不具备 ICLR Oral 级方法新颖性；不建议占用 GPU 做主线验证。**

## 1. 候选方案

候选系统把冻结 VLM 与专用医学小模型分工：

1. VLM 起草含数量、尺寸、角度或变化幅度的临床 claim；
2. 检测器/分割器返回潜在测量场景及其不确定集合 `Z`；
3. claim 被编译成关于 `Z` 的谓词；
4. token 级约束解码只允许输出由 `Z` 支持的数字和比较词，非数值文本尽量不变。

核心直觉是对的：**通用 VLM 负责语言，专用模型负责它更擅长、且可验证的视觉量。** 问题不在实用性，而在研究贡献是否已被已有工作分割覆盖。

## 2. 严格碰撞矩阵

| 本方案组成 | 最近直接邻近工作 | 已覆盖内容 | 剩余空间 |
|---|---|---|---|
| 胸片定量 claim 修复 | [FactCheXcker, CVPR 2025](https://arxiv.org/abs/2411.18672) | 已正式定义 measurement hallucination；用专用检测/分割、代码和确定性换算更新报告；在 11 个报告模型上验证 | 不确定集合和 report-wide guarantee |
| 医学尺寸、距离、角度 | [MedVision, 2025/2026](https://openreview.net/forum?id=IdPZh1CACh) | 22 个数据集、检测/病灶尺寸/角度距离；坐标与 pixel size 换算；公开 `scaledPS` 单位缩放测试 | 幻觉约束而非能力训练；但量化任务本身不新 |
| 计数专用小模型协助 | [GroundCount, 2026](https://arxiv.org/abs/2603.10978) | 用 YOLO/object detector 显式 grounding 来减少 counting hallucination，报告约 6.6pp 提升 | 医学专用化不能构成核心新意 |
| 小模型/程序组合 | [VisProg, CVPR 2023 Best Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.html) | LLM 生成程序，调用 detector、segmenter、count、算术模块，training-free | 把模块换成医学模型属于应用迁移 |
| 语义谓词约束生成 | [ChopChop, 2025/2026](https://arxiv.org/abs/2509.00360) | 将 token 生成与抽象程序语义连接，保证输出满足任意丰富语义性质 | 医学 AST 是一个实例，不是新的约束解码原理 |
| VLM factuality 统计保证 | [CONFLVLM, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.576/) | 对 VLM claim 给出有限样本、分布无关 factuality guarantee；已经包含 MIMIC-CXR 报告、LLaVA-Med 和 MAIRA-2 | 使用专用几何 latent set，而非启发式 claim score |
| 分割不确定集合 | [Conformal Prediction Sets for Instance Segmentation, 2026](https://arxiv.org/abs/2602.10045) | 为 instance segmentation 构造带覆盖保证的预测集合 | 与下游文本约束的组合 |
| 下游任务感知 conformal | [Utility-Directed Conformal Prediction, ICLR 2025](https://openreview.net/forum?id=iOMnn1hSBO) | 让 conformal set 考虑下游决策损失并保留覆盖保证 | 临床数值语言是一个具体效用实例 |

因此，当前方案可以被审稿人准确概括为：

> FactCheXcker/GroundCount 式专用视觉工具 + conformal prediction set + ChopChop 式语义约束解码。

这是一套合理的 A+B+C 系统，但还不是新原理；也不能通过“measurement instrument”重新命名成新范式。

## 3. 数学保证：正确版本与原版本缺口

### 3.1 必要背景

`Z(x)` 不是一个点预测，而是一组仍可能为真的视觉场景。例如，对一个病灶，集合同时包含“直径 4.8–5.4 mm、边界在若干相近 mask 内”的所有可能状态。

若真实场景记为 `z*`，校准后有联合覆盖保证：

\[
\Pr[z^*\in Z(x)]\ge 1-\alpha.
\]

若解码器只输出对集合中**每一种可能场景都成立**的 claim：

\[
\forall z\in Z(x),\quad c(z)=\mathrm{true},
\]

那么只要 `z*` 落在集合中，输出 claim 就一定正确。因此：

\[
\Pr[\text{至少一个被认证的定量 claim 错误}]\le \alpha.
\]

这不是新的证明，而是 conformal coverage 与 sound constrained decoding 的一步合成。

### 3.2 原方案何时不成立

上述 report-wide `≤ α` 只有在以下条件全部成立时才成立：

1. `Z` 对整张图、所有对象、所有将被自适应选择的数量是**联合覆盖**，而非每个对象各自覆盖；
2. claim parser/编译器不会把“左侧病灶”错误绑定到“右侧病灶”；
3. 单位、pixel spacing、成像平面和校准来源也包含在 latent scene 中；
4. 解码采用对 `Z` 的全称蕴含，而不是取中心值或采样一个 mask；
5. conformal 所需的 exchangeability/校准假设成立。

若每个 claim 只有边际错误率 `α`，一份报告生成 `m` 个 claim，最坏情况下只能由 union bound 得到：

\[
\Pr[\text{报告中任一 claim 错}]\le m\alpha,
\]

而不是 `α`。VLM 又会根据图像自适应选择“最想说的对象”，会进一步破坏朴素的边际保证。

若 claim 编译器本身出错概率为 `δ`，最乐观的整体界也变成 `α+δ`。所以“只约束数字 token”不能认证整句话：`5 mm` 可以合法，但“左侧病灶 5 mm”仍可能绑定错对象。

### 3.3 不变性主张也需收缩

- 单位变换协变是标准 dimensional analysis；MedVision `scaledPS` 已直接实现，不能作为理论贡献。
- 面积/体积对不相交分块可加；对 union 后测量的直径可以保持。
- **计数不具有 subdivision invariance**：一个病灶被错误切成两个实例时，count 会从 1 变 2。
- 周长、角度也依赖边界重建和 landmark identity，并不天然对分块不变。

因此原来的“统一 subdivision invariance”是错误命题。若要严格成立，必须定义 instance identity 的等价类和合法 merge/split 规则；这会让方法从简洁约束迅速变成复杂场景图验证。

## 4. 最致命的系统问题：数字正确不等于 claim 正确

真实例子：草稿写“左肺结节直径 5 mm”，小模型测到的是右肺另一个结节，约束器允许了 `5 mm`。

- 数字合法；
- 单位合法；
- 但实体绑定错误，整个 claim 仍是假话。

类似问题还包括 current vs prior、长轴 vs 短轴、patient plane vs detector plane。要给整句保证，系统必须把自然语言先解析成带实体身份的 AST，并认证 AST；只保留原有非数值文字无法实现 factuality guarantee。到这一步，方案与视觉程序、formal verification、semantic constrained decoding 的碰撞更强，而不再是一个 VCD/SECOND 式简洁解码规则。

## 5. 本地真实 effect mass

审计当前论文 baseline 中四组完整 greedy 报告：

| 模型/数据 | 报告数 | reference 含精确 mm/cm | model 输出含精确 mm/cm | 输出类型 |
|---|---:|---:|---:|---|
| Huatuo / IU-Xray | 590 | 10（12 mentions） | 1（2 mentions） | 一例虚构 `3 cm × 2 cm` opacity |
| Huatuo / MIMIC-CXR | 694 | 46（60 mentions） | 1（1 mention） | ET tube 距 carina |
| Hulu / IU-Xray | 590 | 10（12 mentions） | 0 | — |
| Hulu / MIMIC-CXR | 694 | 46（60 mentions） | 15（19 mentions） | 全部主要为导管/气管管尖端距离 |

结论：当前本地报告 substrate 中，模型精确尺寸输出极少；主要可修复对象是 device tip distance，而不是广泛病灶尺寸。四组共 2,568 份报告中只有 17 份生成精确 mm/cm，无法支撑主论文的定量幻觉结论。

已有数据审计还显示：

- VQA-RAD 只有 3 个独立测量图、4 个相关问题；统计功效不足。
- VinDr 有大量 bbox，但 15,000 个 train DICOM 中 2,152 个缺 `PixelSpacing`，其余也全部缺可验证的 calibration provenance；不能把 bbox × spacing 冒充 patient-space mm 真值。
- 本地没有可直接提供病灶测量 uncertainty set 的专用 detector/segmenter checkpoint；现有 XRV expert 是分类器，不是测量仪器。

## 6. ICLR Oral 判定

| 维度 | 判定 |
|---|---|
| 问题重要性 | 高：错误尺寸、计数、角度确有临床风险 |
| 方法直观性 | 中高：大模型写，小模型测，容易理解 |
| 数学新颖性 | 低：coverage + sound decoder 的标准合成；原 report-wide 界还缺联合覆盖 |
| 方法新颖性 | 低到中：三个模块各有直接先验，组合性强 |
| 通用性 | 表面广，实际上需为每类 quantity 构造 detector、identity、单位和 AST |
| 本地正例与规模 | 弱：当前输出中目标 claim 极少，且缺合法 patient-mm truth |
| Oral-ready | **否** |

严格结论：它可能成为可靠的医学量化安全系统或应用论文，但不能作为当前 ICLR Oral 主线。继续跑 GPU 只能验证系统有效，无法修复核心创新碰撞。

## 7. 如果坚持做，最便宜且诚实的 L0

不使用当前 CXR 报告；应直接使用 MedVision 的 calibrated size/angle 子集。冻结 128 个单实例样本，比较：

1. 原生 VLM；
2. 将工具点估计放入 prompt；
3. 确定性 slot replacement；
4. joint conformal set + universal-entailment constrained decoder；
5. shuffled tool 与错误 calibration placebo。

指标必须同时报告：定量误差、report-wide any-error、保留数字比例、区间宽度、非数字 token 一致率和时延。继续门槛：相对工具 prompt / deterministic replacement 至少降低 20% 定量错误，保留 ≥90% 数字 claim，size 与 angle 均成立。

但这个 L0 的最可能结果是：确定性 slot replacement 已足够好，conformal 版本只是以更宽区间或少说换保证。由于即使成功也不能解决新颖性，本轮建议 **不启动 L0，不占 baseline GPU**。

## 8. 可保留的研究启发

真正值得带到下一轮 idea 搜索的不是本方案本身，而是一个设计原则：

> 专用小模型不应只是给 VLM 一个额外答案；它应提供一种 VLM 原本无法伪造、且能改变生成可行域的结构化证据。

下一候选若仍采用大小模型合作，必须回答一个比“工具值替换”更深的问题，例如：小模型产生的哪种结构使某类幻觉在数学上变成不可生成，同时又不依赖为每种医学 quantity 手写一个 verifier。否则仍会落回 VisProg / tool use / formal constrained decoding 的既有范式。


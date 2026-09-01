# `Pixels Have No Units` / Metric-Gauge Hallucination：碰撞、可识别性与最小验证审计

> 日期：2026-08-03（UTC）  
> 结论状态：**原始版本 Reject-and-Pivot；校准状态版本值得做一天内淘汰 probe，但不应直接升为 ICLR 主线。**  
> 运行边界：本审计没有调用或占用 GPU，没有修改共享评测代码。

## 0. 结论先行

原始候选包含三个直觉：

1. 同一视觉 tensor 配不同 `PixelSpacing`，mm/cm 输出应按比例协变；
2. 图像缩放并反向修改 spacing 后，物理测量应不变；
3. 缺失 spacing/scale bar 时，模型不应凭训练先验生成确定 mm/cm。

其中前两项已经被最近工作高度覆盖，甚至出现近乎逐行相同的公开实现：

- **MedVision** 已将图像尺寸和 pixel size 输入 prompt，通过坐标定位再换算物理单位；其 2026-04-26 公开的 `scaledPS` 任务保持图像像素不变、只随机缩放 prompt 中的 pixel size，并同步缩放物理真值，明确测试模型是否真正使用 spacing。
- **FactCheXcker（CVPR 2025）** 已把胸片报告中的错误数值命名为 measurement hallucination，并用 `query → code/tool → report update` 修复；其 API 将检测坐标通过 `pixel_spacing` 确定性换算为 cm。
- **MeasureBench（CVPR 2026）** 已系统评估 VLM 的视觉测量读取，定位到 indicator localization 是主要瓶颈。
- **Dimensionless Machine Learning（JMLR 2023）** 已从理论上把单位变换写成 exact units equivariance。

因此，不能再把“测 mm”“spacing counterfactual”或“坐标后接计算器”写成核心新颖性。

真正尚未被上述工作解决的是更窄但更准确的问题：

> **Calibration-state hallucination**：系统拥有一个数值 spacing，并不等于它有权把像素距离称为患者解剖距离。模型和工具链是否区分 `patient-plane metric`、`detector-plane metric` 与 `unidentified/pixel-only` 三种 measurement type？

这是一个真实缺口。FactCheXcker 的公开 API 接受一个裸 `pixel_spacing` 元组并直接输出 cm，没有 calibration provenance/type；MedVision 则从一开始只纳入具有 physical spacing header 的数据，并默认该数值足以定义物理目标。当前检索没有发现医学 VLM 工作对 **missing/ambiguous calibration state + counterfactual units consistency** 做系统研究。

但这个剩余缺口的直接修复是一个确定性 type guard，方法复杂度太低；单独做至多是一篇扎实的 safety benchmark / short paper，而非可信的 ICLR oral 主贡献。它只有扩展为跨模态的 **identifiability contract**，再发现“归一化几何仍在、单位承诺由后层先验补全”的内部机制，才可能重新获得较高 ceiling。

## 1. 冻结研究问题

- **RQ1 — Collision**：医学 VLM 的 counterfactual scale equivariance、missing-calibration third state、deterministic coordinate-to-unit decoding 分别是否已被覆盖？
- **RQ2 — Identifiability**：当 `PixelSpacing` 缺失或校准来源不明时，图像是否能唯一确定 patient-space mm/cm？如果不能，正确输出状态是什么？
- **RQ3 — Minimal falsification**：本地 `/workspace/vinbigdata` 的 DICOM 与多读者 bbox 能否在一天内完成无需新标注的最小 probe？
- **RQ4 — Priority**：该方向是否值得压过 reader-grounded two-plane / 其他机制分支成为主线？

## 2. 最近工作碰撞矩阵

| 工作 | 已覆盖 | 未覆盖 | 对本候选的影响 |
|---|---|---|---|
| [FactCheXcker, CVPR 2025](https://arxiv.org/abs/2411.18672) | 正式命名并缓解 CXR report measurement hallucination；query-code-update；坐标/分割后确定性换算 cm；11 个报告模型 | calibration provenance；spacing 缺失/歧义；单位协变反事实 | “measurement hallucination + modular tool repair” 已被直接占领 |
| [MedVision, arXiv 2025 / ICLR 2026 submission](https://arxiv.org/abs/2511.18676) | 22+ 医学数据集；detection、T/L size、angle/distance；prompt 显式给 image size 与 pixel size；坐标到 mm 的推理 | 论文正文未研究 calibration missingness；不区分 detector-mm 与 patient-mm | “医学 VLM 定量测量”已是现成大基准 |
| [MedVision public code `scaledPS`](https://github.com/YongchengYAO/MedVision) | **同一图像像素不变，只缩放 prompt pixel size**；TL 用 uniform scale，AD 用 anisotropic scale；真值同步重算 | 不要求缺失 spacing 时拒绝；不审计 calibration provenance | 与第一条 counterfactual 几乎完全同构；这是最致命碰撞 |
| MedVision tool-use SFT（同一仓库） | 两阶段生成代码并由受限 Python executor 执行，再生成答案 | 没有 typed calibration guard | “让 VLM 预测坐标、外部确定性计算单位”也非全新 |
| [MeasureBench, CVPR 2026](https://arxiv.org/abs/2510.26865) | 通用 VLM 视觉测量 benchmark；合成/真实仪表；定位错误分析 | 非医学 DICOM；没有 spacing provenance | 把候选的 broader visual-measurement claim 进一步压低 |
| [Dimensionless Machine Learning, JMLR 2023](https://www.jmlr.org/papers/v24/22-0680.html) | exact units equivariance；先构造 dimensionless inputs，再在无量纲空间推理 | 非 VLM、非医学、非不确定表达 | 可作为理论语言，不能当新理论贡献 |
| [Scale-Equivariant Deep Learning for 3D Data, 2023](https://arxiv.org/abs/2304.05864) | 3D 医学图像的 scale-equivariant layer / U-Net | 目标是 segmentation feature scale，而非单位可识别性 | “医学 scale equivariance”本身也不新 |
| [On Scale Invariance in CNNs, 2021](https://doi.org/10.3390/make3020019) | 发现 ImageNet CNN 后层丢失医学任务需要的 scale 信息 | 非 VLM、无 metadata intervention | 若做层级机制必须正面比较 |
| [Seeing Isn't Knowing, arXiv 2026](https://arxiv.org/abs/2605.30557) | 信息不足的 spatial question 应 abstain；受控 occlusion/perspective ambiguity | 非医学；不是 metric calibration | missing-evidence third state 的一般思想已有邻近工作 |
| [BCEA, arXiv 2026](https://arxiv.org/abs/2606.16667) | answer / abstain / acquire evidence 三态与 conformal guarantee | 非医学量纲 | “第三态”本身不足以构成新颖性 |
| [DeepTumorVQA, arXiv 2026](https://arxiv.org/abs/2605.09679) | 3D CT 的 recognition → measurement → reasoning 分层 benchmark；tool agent 改善测量 | 未检索到 calibration counterfactual | 定量肿瘤测量与 tool augmentation 也已有强邻居 |

### 2.1 最接近碰撞不是摘要层面，而是代码层面

审计的 MedVision commit：

- 仓库快照：`9fff1788cfbe41c22b63da07f986954024c0d184`
- `aa708db93c43cc61a4ceab73a344321452e53d60`，2026-04-26：add `-CoT-scaledPS` variants
- `2bfec2760ddc508517bfa61956e3bb612a07ed08`，2026-04-26：同一功能的后续提交
- 该功能进入公开 tag `v1.1.0` 与 `v1.1.1`

其注释明确写明：对每个样本从区间采样 deterministic scale factor，只改变 prompt 中展示的 `pixel_size`，图像不 resize；TL 的 major/minor truth 随同一个标量缩放，AD 用二维 anisotropic factors 从 landmark 重新计算 distance/angle。

这与“same visual tensor + different spacing should change linearly”的原始核心测试没有实质差异。即便论文正文没有主打该实验，提交前已经存在、公开且带测试的代码 artifact，不能忽略。

### 2.2 Deterministic decoder 也被强覆盖

FactCheXcker 的公开代码快照 `c2e1025efb9290408fe5d1e0eed78ad97ca0cc81`：

- `CXRImage` 构造器强制接受 `pixel_spacing: Tuple[float, float]`；
- `convert_pixel_dx_to_cm = pixel_dx * pixel_spacing[0] / 10`；
- `distance`、`diameter`、`dimensions` 都调用该确定性转换；
- 没有 `CalibrationType`、`ImagerPixelSpacing`、magnification 或 missingness validation。

因此，deterministic coordinate-to-unit decoder 的剩余新意只能是：**在运算前做 measurement-type checking，并在不可识别时禁止 patient-unit commitment。**

## 3. 真正可保留的问题：不是 scale error，而是 metric identifiability

### 3.1 DICOM 已经定义了这个第三态

[DICOM Basic Pixel Spacing Calibration Macro](https://dicom.nema.org/medical/DICOM/current/output/chtml/part03/sect_10.7.html) 规定：

- `PixelSpacing (0028,0030)` 表示患者平面内相邻像素中心的物理距离；
- 若它与 `ImagerPixelSpacing (0018,1164)` 不同，则说明做过几何放大校正或用已知物体校准；
- 若 `PixelSpacingCalibrationType`、`ImagerPixelSpacing`、`NominalScannedPixelSpacing` 都缺失，则**无法确定是否进行过校正/校准**；
- projection X-ray 的 `ImagerPixelSpacing` 是 detector plane spacing，不能自动等同于患者解剖尺度。

可把输出类型写成：

\[
T_m \in \{\texttt{patient-mm},\ \texttt{detector-mm},\ \texttt{pixel-only/unknown}\}.
\]

只有第一种允许生成“病灶直径为 12 mm”这类 patient-space claim。第二种必须标记为 `mm at imager plane`；第三种只能报告 pixels、relative coordinates 或明确说明需要 calibration metadata / scale bar。

### 3.2 为什么 pixels 本身不能决定 mm

令归一化图像为 \(x\)，视觉定位得到两个无量纲端点 \(u_1,u_2\)，spacing 为 \(s\)。物理距离是

\[
d = \|A(s)(u_1-u_2)\|.
\]

如果 \(s\) 或它对应的物理平面未知，对任意 \(a>0\)，世界 \((x,s)\) 与 \((x,as)\) 具有同一像素观测，却对应不同 \(d\)。所以 image-only patient-mm 是不可识别的，不是“置信度低但仍可猜”的普通预测问题。

投影胸片还多一层不可识别性：detector spacing 已知也不代表患者内不同深度物体的真实尺寸已知；需要 magnification geometry 或同深度 fiducial。这个限制不能通过更大的 VLM 消除。

## 4. 本地 VinDr/VinBigData 数据审计

路径：`/workspace/vinbigdata`

### 4.1 数据规模

- 15,000 张 train DICOM；3,000 张 test DICOM；总计 18,000。
- `train.csv`：67,914 行、15,000 张独立图像、17 个 `rad_id`。
- 36,096 个 positive bbox rows，覆盖 4,394 张图像。
- 全量 15,000 个 train headers 的 `stop_before_pixels` 扫描耗时 **2.91 秒**。

### 4.2 Calibration metadata 完整性

| 状态 | 图像数 |
|---|---:|
| 有 `PixelSpacing` | 12,848 |
| 无 `PixelSpacing` | 2,152 |
| 有 `ImagerPixelSpacing` | 0 |
| 有 `NominalScannedPixelSpacing` | 0 |
| 有 `PixelSpacingCalibrationType` | 0 |
| 有 `PixelSpacingCalibrationDescription` | 0 |
| 有 source-detector / source-patient distance 或 ERMF | 0 |

因此：

- 2,152 张显然不能从头信息换算 mm；
- 其余 12,848 张只有裸 `PixelSpacing`。按 DICOM 标准，无法判定是否对 projection magnification 做过校正；
- **不能把 bbox × PixelSpacing 当成真实 patient-space lesion size ground truth。** 最多称为 `header-implied nominal length`，且必须保留校准状态不明的限定。

这既是方向最有价值的发现，也是使用该数据写 measurement accuracy paper 的致命限制。

### 4.3 可用于无真值 equivariance test 的 reader-consensus 子集

要求同一 finding 每位 reader 仅一个 box、至少两位 reader、reader boxes 的 median pairwise IoU ≥ 0.5：

| Finding | 合格图像 | 有 PixelSpacing | 至少 3 readers |
|---|---:|---:|---:|
| Nodule/Mass | 213 | 135 | 97 |
| Cardiomegaly | 1,752 | 1,153 | 1,256 |
| Pleural effusion | 222 | 138 | 167 |
| Pulmonary fibrosis | 269 | 198 | 143 |
| Lung opacity | 206 | 129 | 50 |

`Nodule/Mass` 是一天内 probe 的首选：对象边界和“直径”语义比 cardiomegaly bbox 更合理；97 张三读者一致样本足以做约 100-case paired test。注意 bbox 仍不是 RECIST/临床 caliper truth，只用于确保对象真实存在并降低 localization ambiguity。

### 4.4 现有本地筛选结果：现象很尖锐，但 benchmark mass 很小

另一会话已经对本地开放 VQA / report substrate 做过 outcome screening，结果必须与本方向一起解释：

- **VQA-RAD 只有 3 个独立带单位测量图、4 个问题**；其中 question 125/126 还是同一张图。原始 JPEG 没有可见 scale bar。
- 一个 GT 为 **5 mm** 的 case：Huatuo 回答约 `5.0 × 4.8 cm`，Hulu 回答 `10.5 cm`，LLaVA 回答 `10 × 10 cm`，出现约一个数量级的单位/尺度过度承诺。
- 一个 GT 为 **2.5 cm** 的 case：Huatuo 回答 `1.2 cm`，Hulu `1.5 cm`，LLaVA `2.5 cm`。
- VinDr 随机 500 个 DICOM 中 442 个有 `PixelSpacing`，但当前 VLM 的 JPEG 输入路径没有传递该 metadata。这与本次全量 header 审计一致，也说明“模型看到了 DICOM spacing”不能被默认假设。
- MIMIC 本地 694 个 unique reports 中，46 个（6.6%）包含 mm/cm，共 60 个 measurement mentions。

这些观察适合作为 motivating cases，不能作为正式统计结论：VQA-RAD 的独立样本数过小；问题可能来自训练记忆、单位混淆、视觉定位或 prompt presupposition，无法单凭 3 张图归因于 metric-gauge mechanism。MIMIC 的 6.6% 则表明错误具有真实临床 effect mass，但 measurement claim 仍只是报告幻觉的一部分，不能据此声称解决通用医学 VLM 幻觉。

最重要的 benchmark-quality caveat 是：GT 自身是否拥有可追踪 calibration provenance。若只有 JPEG、问题与一个 mm/cm 答案，而无 DICOM geometry、scale bar 或 measurement protocol，那么模型答错可以被计数，但该样本不能支撑严谨的 counterfactual scale 或 patient-space measurement 机制分析。

## 5. 一天内最小 probe（不把 nominal mm 当临床真值）

### 5.1 目标

这个 probe 只回答两个问题：

1. 模型的数值输出是否遵守 spacing 的 counterfactual transformation law？
2. spacing 缺失或 calibration state 明确未知时，模型是否仍过度承诺 patient mm/cm？

它**不**声称评估真实患者病灶尺寸精度。

### 5.2 样本与模型

- 96–97 个 Nodule/Mass 三读者一致图像，固定 manifest；
- 本地已具备：
  - `/home/dbw/models/HuatuoGPT-Vision-7B-Qwen2.5VL`（约 13 GB）；
  - `/home/dbw/models/Qwen2.5-VL-7B-Instruct`（约 13 GB）；
- 一张 48 GB RTX 4090 当前空闲；短结构化回答足以 batch inference。

### 5.3 配对条件

对每张完全相同的 rendered PNG，随机隐藏实际 spacing，并用去标识的 scale symbols 生成：

1. `S1`: 告知 spacing \((s_h,s_w)\)；
2. `S0.5`: 告知 \((0.5s_h,0.5s_w)\)；
3. `S2`: 告知 \((2s_h,2s_w)\)；
4. `MISSING`: 不给 spacing/scale bar；
5. `AMBIGUOUS`: 明确说明只有 detector pixel pitch，patient magnification/calibration 未知；
6. 可选 `UNIT`: 将同一合法 spacing 从 mm/pixel 改写为 cm/pixel，检验 unit covariance。

第二个 invariance family 可用同一 DICOM 构造不同 raster resolution：内容缩小 \(r\) 倍、spacing 放大 \(1/r\) 倍，使 nominal physical extent 不变。它不像前三个条件那样具有完全相同的 vision tensor，必须单独报告 interpolation/processor effects，不能混为一个主 DID。

### 5.4 Prompt 与输出

同一个问题模板，仅 metadata 行变化：

```text
Locate the single pulmonary nodule/mass identified in this radiograph.
Return:
1) whether the object is visible;
2) normalized endpoints of its longest apparent diameter;
3) measurement type: patient-mm, detector-mm, or pixel-only/unknown;
4) a physical value only if patient-space calibration is identifiable.
```

同时跑一个自然语言版本：“What is the maximum diameter ... in millimeters?”。前者测 capability/type compliance；后者测 clinical presupposition 是否诱导错误承诺。两者不能混成一个分数。

### 5.5 主指标

对合法 counterfactual spacing，令模型数值为 \(y_a\)，scale factor 为 \(a\)：

\[
E_{eq}=\left|\log\frac{y_a}{y_1}-\log a\right|,
\qquad
\hat\beta = \mathrm{slope}(\log y_a,\log a).
\]

正确 unit-sensitive decoder 应有 \(\hat\beta\approx1\)。另报告：

- endpoint stability：spacing 改变不应改变 normalized visual endpoints；
- `MISSING` patient-unit overcommitment rate；
- `AMBIGUOUS` patient-unit overcommitment rate；
- detector-mm / pixel-only 类型准确率；
- unit conversion consistency（mm ↔ cm）；
- resize + inverse-spacing physical invariance error；
- exact paired bootstrap CI，以 image 为 cluster。

不要用绝对 mm MAE 作为 VinDr 主指标。若需 accuracy，仅比较 pixel/normalized bbox 与 reader consensus；物理单位只评价 transformation law 与 commitment legality。

### 5.6 必须加入的 oracle 分解

否则会把 localization failure 错当 gauge failure：

- **Oracle-coordinate arm**：把 reader-consensus endpoints 直接作为文字输入，只考 spacing → unit arithmetic/type；
- **Vision-coordinate arm**：模型自己定位，但由外部 deterministic decoder 换算；
- **End-to-end arm**：模型定位并自己输出单位。

三者把错误分成 perception、coordinate expression、arithmetic、calibration commitment 四段。这比只看最终 mm 更可信。

### 5.7 基线

- direct VLM；
- explicit CoT；
- calculator/tool prompt；
- FactCheXcker-style deterministic conversion；
- MedVision-V0（若决定下载，仅作为 strongest quantitative baseline）；
- **Typed decoder**：无量纲 endpoints + metadata validator + deterministic unit transform。

Typed decoder 的规则非常简单：

```text
if patient_plane_calibrated:
    emit patient-mm
elif detector_spacing_known:
    emit detector-mm (never patient size)
else:
    emit pixel-only/unknown and request calibration
```

由于这条规则按构造就能把 illegal unit commitment 降为 0，不能把这一数字包装成 learned-method breakthrough。真正有信息量的是它是否保留 localization/coverage，以及已有系统在现实 headers 上多频繁违反 type contract。

### 5.8 时间预算

| 工作 | 预计时间 |
|---|---:|
| header + reader-consensus manifest | 10–30 分钟（核心统计已完成） |
| 约 100 张 DICOM render + paired prompts + hash audit | 30–90 分钟 |
| 两个 7B、约 5–6 conditions、短输出推理 | 2–6 GPU 小时，取决于 backend/batch |
| parsing、paired bootstrap、case audit | 1–3 小时 |
| 总计 | **同一天可完成** |

## 6. Fatal-flaw audit

### F1 — 原始新颖性被直接先验覆盖（CRITICAL for original version）

检测依据：MedVision `scaledPS` 已实现 same pixels + scaled spacing + scaled ground truth；FactCheXcker 已定义并缓解 CXR measurement hallucination，且以 deterministic coordinate-to-cm API 为核心。

结论：原始 `Metric Gauge Hallucination` 不能以“首次 counterfactual scale equivariance / 首次 deterministic decoder / 首次医学 measurement hallucination”为题继续。

### F2 — VinDr 不能提供 claimed patient-mm truth（MAJOR，若误用则 CRITICAL）

15,000 个 train DICOM 全部缺少可验证 calibration provenance；2,152 个连 PixelSpacing 也没有。projection CXR 的几何放大进一步使 detector-plane 距离与 patient anatomy 距离不同。

防御：把实验限定为 transformation-law / illegal-commitment test；绝对 patient-mm accuracy 必须换到 calibration 可验证的数据，或引入同平面 fiducial / geometry。

### F3 — “修复”可能只是 if-statement（MAJOR ceiling flaw）

只要 metadata validator 判定三种状态，illegal unit 输出按构造消失。reviewer 会合理地问：为何需要 VLM/ICLR 方法？

防御不是加复杂网络，而是扩大科学问题：证明多个医学 quantity 都存在相同的 identifiability boundary，并定位模型为何在视觉几何仍可用时由语言先验补出单位承诺。

## 7. Reviewer-style novelty ceiling

### 7.1 原始版本

- Paper type：incremental benchmark/diagnostic。
- Verdict：**Reject and Pivot**。
- Ceiling：workshop / short empirical note；无法支撑 ICLR 2027 主会新颖性，更不用说 oral。

### 7.2 校准状态修正版

一句话故事：

> Existing quantitative medical VLMs can be numerically consistent yet physically unjustified, because they collapse patient-calibrated, detector-calibrated, and uncalibrated measurements into the same unit-bearing language.

这个故事有真正的 hidden assumption：领域普遍把 header 中的一个 spacing 数字当作充分物理证据，FactCheXcker 与 MedVision 都呈现这一倾向。但目前它仍是 **Novel Problem / New Setting**，不是足够强的方法贡献。

机制未验证前的五维判断：

| 维度 | 分数 | 理由 |
|---|---:|---|
| Higher | 5/10 | 能消除非法单位 claim，但规则按构造成功，不能证明整体 clinical quality 提升 |
| Faster | 7/10 | metadata type-check + deterministic conversion 极便宜，无需训练 |
| Stronger | 7/10 | 对 missing/corrupt/ambiguous metadata 有明确契约；需跨模态验证才可再提高 |
| Cheaper | 8/10 | transformation law 不需昂贵 patient-mm 标注；可用 paired counterfactual 自监督审计 |
| Broader | 5/10 | 当前只落在 2D CXR scale；若扩展 intensity/SUV、time、orientation 才能成为统一框架 |

当前 ceiling：有望成为有辨识度的 benchmark/safety paper；若只做 VinDr + 两模型，约为主会 borderline，远非 oral。

本地 VQA-RAD 的数量级错误提升了 motivation，却不提升这个 ceiling：3 个独立图像没有统计功效，且缺少可见标尺与 calibration provenance。MIMIC 抽样中 6.6% 的 measurement-report prevalence 足以说明临床相关性，但也要求论文把 claim 严格限定为 quantitative claims，而非全部医学幻觉。

### 7.3 获得更高 ceiling 所需的升级

必须从 `pixel scale` 扩成 **Medical Measurement Identifiability Contract**：

- geometry：patient-mm / detector-mm / pixels；
- CT intensity：HU 需要 rescale slope/intercept，普通 PNG 灰度无 HU；
- PET：SUV 需要剂量、时间、体重等 metadata，像素本身不够；
- temporal claims：单张 current image 无法支持 interval change；
- orientation/laterality：去掉 orientation metadata 后不得确定 patient coordinate claim。

统一机制可写为：模型先输出 dimensionless visual primitive，再由一个 typed evidence compiler 绑定合法单位；缺失必要 metadata 时生成 `unidentifiable`，而不是从训练源域先验补全。

只有再加入下列机制证据，才可能接近高水平 ICLR：

1. layerwise probe 发现 normalized geometry 在某层可解码，但 calibration-state/unknown 在生成层被抹除；
2. activation patching 只改变 unit commitment，不改变 endpoints/object identity；
3. 跨至少三种 physical quantities 共享同一 factorization；
4. OE/report 中在 matched content/claim count 下减少 illegal quantitative claims，不以删除所有数字获益。

这已经是一篇明显更大的新论文，不能假装是一天 probe 的自然结论。

## 8. 与当前其他分支的优先级

### 8.1 作为下一步实验

**值得优先做一天 probe。** 原因不是它已足够强，而是：

- 现象由严格 transformation law 定义，不依赖 LLM judge；
- 配对设计强、无需 patient-mm ground truth；
- 本地数据、模型、GPU 都已齐备；
- 一天即可获得明确 kill/continue signal；
- 相比训练多个 synthetic provenance children，它更临床、更便宜、lineage 风险更低。

### 8.2 作为论文主线

**目前不值得压过 reader-grounded two-plane mechanism。** Reader disagreement → language commitment 仍有更高机制深度和通用 hallucination relevance；Metric Gauge 当前只覆盖数字 measurement claim，并被 MedVision/FactCheXcker 强包围。

推荐排序：

1. 用一天完成 metric-gauge falsification probe；
2. 若 off-the-shelf 与 MedVision-style 模型都已对 missing/ambiguous calibration 正确 abstain，立即停止；
3. 若跨两模型出现强烈非法 patient-unit commitment，只把它保留为一个高质量 phenomenon / diagnostic；
4. 只有当该行为能扩展到多个 metadata-defined quantities，并出现共享 layerwise causal mechanism，才考虑替代当前主线。

## 9. 预注册淘汰标准

一天 probe 继续的最低条件：

- 至少两个模型在 `MISSING` 或 `AMBIGUOUS` 条件中持续生成 patient mm/cm，而不是偶发 parsing error；
- 该效应在 matched prompt length、无 measurement presupposition 的中性 prompt 下仍存在；
- oracle-coordinate arm 表明问题不是纯 localization failure；
- spacing intervention 改变 unit value/commitment，但 object presence 与 normalized endpoints 基本保持；
- 现象可在至少第二种 quantity 或第二种 modality 快速复现。

任一以下条件触发停止：

- 模型已普遍拒绝 missing/ambiguous calibration；
- 所谓增益完全来自明确 if-statement，且没有模型内部可解释机制；
- 只有自然语言强制“in mm”时才出现错误，改成中性问题即消失；
- resize 结果主要由 interpolation/vision processor 改变，而非 gauge law；
- 需要把 VinDr nominal spacing 冒充 patient-mm truth 才能得到结论。

## 10. 最终回答 RQ

- **RQ1**：counterfactual spacing covariance 被 MedVision `scaledPS` 几乎精确覆盖；deterministic coordinate-to-unit decoding 被 FactCheXcker 和 MedVision tool-use 强覆盖；missing/ambiguous calibration third state 尚未发现直接医学 VLM 先验，但邻近的 spatial abstention 已存在。
- **RQ2**：缺失或来源不明的 spacing 时 patient-mm 不可识别；projection X-ray 还必须区分 detector plane 与 patient plane。正确状态不是估计一个低置信度数字，而是改变 measurement type。
- **RQ3**：一天内 probe 可行。本地有 97 张三读者一致且有 spacing 的 Nodule/Mass、两套 7B 和空闲 48GB 4090；CPU 数据审计已完成，GPU 不在本次审计中运行。
- **RQ4**：它值得作为极低成本、强可证伪的旁路 probe；原始版本不值得成为主线。只有升级为跨 quantity 的 identifiability mechanism，才可能与当前机制分支竞争。

## 11. 可复核来源

### 论文与标准

1. Heiman et al., [FactCheXcker: Mitigating Measurement Hallucinations in Chest X-ray Report Generation Models](https://arxiv.org/abs/2411.18672), CVPR 2025.
2. Yao et al., [MedVision: Benchmarking Quantitative Medical Image Analysis](https://arxiv.org/abs/2511.18676), 2025/2026.
3. Lin et al., [Do Vision-Language Models Measure Up? MeasureBench](https://arxiv.org/abs/2510.26865), CVPR 2026.
4. Villar et al., [Dimensionless Machine Learning: Imposing Exact Units Equivariance](https://www.jmlr.org/papers/v24/22-0680.html), JMLR 2023.
5. Wimmer et al., [Scale-Equivariant Deep Learning for 3D Data](https://arxiv.org/abs/2304.05864), 2023.
6. Graziani et al., [On the Scale Invariance in State of the Art CNNs Trained on ImageNet](https://doi.org/10.3390/make3020019), 2021.
7. Zhang et al., [Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions?](https://arxiv.org/abs/2605.30557), 2026.
8. Xu et al., [Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition](https://arxiv.org/abs/2606.16667), 2026.
9. Chen et al., [DeepTumorVQA](https://arxiv.org/abs/2605.09679), 2026.
10. DICOM, [Basic Pixel Spacing Calibration Macro](https://dicom.nema.org/medical/DICOM/current/output/chtml/part03/sect_10.7.html).
11. DICOM, [X-Ray Projection Pixel Calibration Macro](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.19.6.9.html).

### 公开代码快照

- MedVision：`9fff1788cfbe41c22b63da07f986954024c0d184`
- FactCheXcker：`c2e1025efb9290408fe5d1e0eed78ad97ca0cc81`

### 检索边界

检索截至 2026-08-03，覆盖 arXiv、OpenReview/CVPR/CVF、JMLR、PMLR、DICOM 官方标准、项目官网与公开 GitHub 代码。对“calibration-state hallucination”没有检索到直接重合工作只能表述为 **no directly overlapping work retrieved under the audited queries**，不能据此宣称绝对首次。

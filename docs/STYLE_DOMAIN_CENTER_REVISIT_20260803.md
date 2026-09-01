# “风格”不是一个机制：source-domain center 严格回访

Date: 2026-08-03  
Scope: CPU-only construct/substrate audit; no new model outcomes and no GPU.

## 结论先行

**当前 source-domain-center 主张是 NO-GO，既有 render sensitivity 不得复活。**

用户最初观察到“同一医学图像换风格后 VLM 回答变化”是真实的现象线索，但它没有识别变化来自哪里。现有本地证据反而排除了最直观的共同中心解释：Huatuo 的严格 paired DICOM-render pilot 在四个冻结 findings 上 `0/4` 通过 held-out signed progression gate；每个 half-A 选出的方向在 half-B 的 95% CI 都跨零。低强度合成变换的所有六个 decision flips 又都发生在低 margin 半区，原始 margin 的错误 AUROC 为 `0.798`，显著优于 raw style drift。

因此，不再问“哪一种风格让答案更好”，而先问：**被改变的量究竟是纯表示、真实成像协议、病理证据，还是带有训练报告先验的来源身份？** 在这四者没有分开以前，任何 source center、style calibration 或 style-aware decoding 都不可解释。

当前只保留一个值得另找数据验证的机制：**Projection-Conditioned Evidence Misbinding（PCEM）**。它不是 source center，也不是 DG：AP/PA 等 acquisition geometry 改变“像素形态 → 临床证据”的测量律，VLM 可能识别出了 projection，却没有在生成 cardiomegaly 等 claim 时使用该条件。现有 VinDr 镜像不含 projection/study identity，故本地仍是 substrate NO-GO。

## 1. 四种不能混称为 style 的因果对象

设真实病理为 `Y`，采集协议为 `A`，呈现/导出映射为 `R`，数据来源及其报告先验为 `S`，模型回答 claim 为 `C`。

| 类别 | 合法操作 | 正确预期 | 回答变化能否证明 source prior |
|---|---|---|---|
| **像素可逆重参数化** | 对同一 detector array 做一一映射，且可从输出精确恢复输入 | 理想系统应在共同 support 上等价；实际差异也可能仅来自 resize、量化、tokenization | **不能**。它只测试实现/表示的 gauge dependence |
| **成像协议变化** | 改 AP/PA、portable/fixed、曝光、探测器、重建/后处理 | 可能改变放大率、噪声、对比度和可见性；有些 claim 应变化、有些不应变化 | **不能直接**。这不是 benign nuisance，必须显式条件化测量律 |
| **病理证据破坏** | blur、mask、window、crop、频谱或 diffusion edit 触及病灶/设备 cue | 若 evidence 下降，输出改变可能完全正确 | **不能**。高 PSNR、edge correlation、feature proximity 都不是临床等价证明 |
| **真正 source-style prior** | 在病理证据和 acquisition law 固定时，只改变可验证的来源线索 `S` | claim commitment 按该模型真实训练来源的 claim prior 有符号地移动 | **可以，但必须有真实 source tag、同图反事实、source-only semantic prior 与因果 mediation** |

关键识别式不是 `C(T(x)) != C(x)`，而是：

\[
\Delta_c^{source}=
\big[C_c(x,S=A)-C_c(x,S=B)\big]
-\Delta_c^{generic\ nuisance},
\]

并要求 `sign(Delta_c_source)` 在 claims 间对齐训练语料中独立估计的 source reporting fingerprint `b_c`。若只有总体 affirmative shift、熵变化或低 margin 翻转，source prior 不成立。

## 2. CPU-only 本地 substrate audit

新增可复现审计：

- code: `anchor/corrected_sgta/audit_style_domain_substrate_v1.py`;
- artifact: `corrected_runs/style_domain_revisit_v1/substrate_audit.json`;
- DICOMs: `/workspace/vinbigdata/train/*.dicom`;
- annotations: `/workspace/vinbigdata/train.csv`;
- artifact SHA-256: `460b79229219dd4192da3b2f366ef684d0296fba0e7164308357c618c55e7617`.

审计只读 headers，`15,000/15,000` 成功，覆盖 annotation 中全部 15,000 unique images。结果是：

| 资格变量 | 非缺失、可分组的自然 source/protocol groups（每组 >=100） |
|---|---:|
| Manufacturer / model | 0 / 0 |
| Institution / Station / Detector / Software | 0 / 0 / 0 / 0 |
| Study / Series / SOP UID | 0 / 0 / 0 |
| ViewPosition / PatientPosition | 0 / 0 |
| VOI LUT / Presentation LUT / intensity relationship | 0 / 0 / 0 |

文件保留了大量**图像表示属性**：`MONOCHROME2=12,357`、`MONOCHROME1=2,643`；BitsStored 为 10/12/14/16 四类；PixelSpacing 有 11 个非缺失值组且 2,152 缺失；WindowCenter/Width、Rows/Columns 也高度多样。组合后有 7,928 个 exact header signatures，最大组为 2,201 张。

这些数字不能反过来命名为 scanner、hospital 或 source。Rows/Columns、spacing、bit depth、window 与 pathology、body habitus、crop、equipment 和 export pipeline 混合；没有 source tag 就没有自然 source truth。更致命的是没有 UID 与 projection metadata，无法构造同 acquisition 或同 study 的 paired counterfactual。

官方 VinDr-CXR 说明原始数据来自两家医院并提供多读者标注，但当前发布/镜像的匿名 header 不提供逐图 hospital assignment。因此“整个 VinDr 是 target domain”合法，“从 raster signature 猜每张图来自哪个 source”不合法。

## 3. 既有本地结果如何约束新假设

### 3.1 数据集可识别不等于 source prior

MIMIC/IU-Xray/CheXpert-proxy 各 500 张的旧 CPU audit 中，公共 dataset identity 在统一 resize 后仍可被强烈区分：full-image intensity balanced accuracy `82.73%`，center-80% radial spectrum `87.60%`。这证明 institution+population+export bundle 可识别，**不证明**纯 acquisition style，也不证明 VLM 使用该坐标生成某一临床 claim。

FedDG 低频 replacement 仅弱地把 source classifier 推向目标域，且破坏临床输出；低频不是主要 source signal。source RGB statistics 在 CE 上出现单例 rescue，但 OE holdout 很快新增 cardiomegaly/effusion 矛盾，触发 catastrophic-harm stop。故“更像 source”与“更临床正确”没有单调关系。

### 3.2 Mild transform flips 主要是 boundary susceptibility

128-case Huatuo CE-D 冻结结果：三种 mild transforms 的 flip rates 仅 `3.13% / 2.34% / 1.56%`，均未达到 preregistered 5%；所有 6 个 unique flips 都在低-margin 半区。错误识别中：

- negative absolute native margin AUROC `0.798`;
- mean style drift AUROC `0.425`;
- max style drift AUROC `0.446`;
- relative drift AUROC `0.711`，仍比 margin 低 `0.087`，paired 90% CI `[-0.167,-0.011]`。

这支持“普通决策边界易感性”，不支持 hallucination-specific style channel。

### 3.3 Paired DICOM render 已正式否定共同方向

必须纳入而不能绕开的 artifact 是：

`corrected_runs/vindr_v2/dicom_render_huatuo_pilot_v1/analysis_v1.json`。

160 claims、160/160 完成、zero errors、lossless duplicate control 通过。这里的 transform admission 是 label-independent computational guard；后续 CECD 所要求的两位真实临床读者 render-equivalence gate 尚未完成。因此该实验足以否定“已有输出中存在可复现共同方向”，即使出现正结果也不足以证明所有变换临床等价。尽管每例 render orbit 不为零、flip rate 可达 17.5%，half-A transform selection 后的 half-B signed reader-equivalent effect 为：

| Finding | frozen transform | held-out effect (95% CI) |
|---|---|---:|
| aortic enlargement | center `+0.05W` | `-0.321 [-0.810, 0.023]` |
| cardiomegaly | center `+0.05W` | `0.015 [-0.604, 0.198]` |
| pleural effusion | width `x1.25` | `-0.020 [-0.427, 0.596]` |
| pulmonary fibrosis | center `-0.05W` | `0.126 [-2.065, 2.266]` |

`0/4` 通过，且后三项 sign agreement 接近随机。结论是 heterogeneous render sensitivity，而不是共同 source/display center。任何新 proposal 若只换 transform bank、center estimator、entropy mask 或 decoder weighting，都是在 outcome 后复活已失败假设。

## 4. 2024--2026 collision boundary

以下邻近工作使“风格不稳定 + 修复”本身没有贡献空间：

- [MM-R3, Findings ACL 2025](https://aclanthology.org/2025.findings-acl.246/) 已大规模评估 image restyling 下的 VLM consistency；[Test-Time Consistency](https://arxiv.org/abs/2506.22395) 已直接优化等价输入的一致性。
- [LENS, 2026 submission](https://openreview.net/pdf?id=oh3c2ieVab)、[VGS-Decoding](https://arxiv.org/abs/2603.20314) 与 SPCD 已把原图/扰动图 token drift 用于 medical hallucination decoding；再做 style-contrastive decoding 是直接碰撞。
- [ARDGen, CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/DG-EBF/html/Ahsan_ARDGen_Augmentation_Regularization_for_Domain-Generalized_Medical_Report_Generation_CVPRW_2025_paper.html) 已做 domain-generalized report generation；[NCBT, MICCAI 2025](https://papers.miccai.org/miccai-2025/0630-Paper0720.html) 已试图去除 CXR device texture；[ICMSeg, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Chen_Generalizable_Single-Source_Cross-Modality_Medical_Image_Segmentation_via_Invariant_Causal_Mechanisms_WACV_2025_paper.html) 已把 style 写成 causal intervention-augmentation。普通 DG/style disentanglement 不新。
- [Boland et al., MIDL 2024](https://proceedings.mlr.press/v250/boland24a.html) 已定位 medical shortcut 出现的层；[Nature Communications 2024](https://www.nature.com/articles/s41467-024-52003-3) 已证明 window、field-of-view、view position 与 portable equipment 等 acquisition factors 会影响 CXR 表征与偏差。仅证明可解码 scanner/style 或找到一个 layer 不够。
- [CORAL, 2026 preprint](https://arxiv.org/abs/2607.03647)、[CounterVHD, 2026 preprint](https://arxiv.org/abs/2606.28520) 已覆盖 image substitution 与 counterfactual grounding。null/shuffle/style view 本身不能构成新机制。

因此唯一可能的 delta 必须研究**一个具体临床测量律如何被 acquisition state 条件化，却在 evidence-to-claim binding 中丢失**；不是 artefact robustness、benign perturbation stability、DG、DICOM rendering consistency 或泛化 counterfactual decoding。

## 5. 唯一值得另找数据的机制：PCEM

### 5.1 Problem

AP 与 PA 不是同一图的“不同风格”。它们改变投影几何：AP 可放大心影，supine/AP 还会改变肺容积和血管外观。因此相同 apparent cardiothoracic ratio 在 AP 与 PA 下不代表相同 cardiomegaly evidence。

PCEM 问：

> 医学 VLM 是否能在线性表征中识别 projection state，却在形成 cardiomegaly commitment 时把 `morphology × projection` 条件证据律压成一个无条件 morphology threshold？

这比 source-domain-center 更接近真实问题。它预测的不是跨 view 完全一致，而是**正确的条件性非一致**。

### 5.2 独立真值与严格反事实

需要新数据，最低标准为：

1. 同一患者、无中间干预、短时间窗的 AP/PA frontal pairs；每张有可信 `ViewPosition`、Study/Acquisition identity；
2. cardiomegaly 的独立参考来自近时点 echo/CT size measurement 或盲法多读者 adjudication，不能从任一待测 CXR 的原报告自动抽取；
3. 自动注册后估计 apparent heart/thorax geometry，但它只作 mediator，不作 truth；
4. 至少一个 geometry-sensitive claim（cardiomegaly）与两个 geometry-insensitive control claims（如 pneumothorax、明显 device presence）；
5. patient-disjoint dev/test，OE 评估固定 positive claim count `K`、长度、拒答和 uncertainty burden。

### 5.3 一日 CPU gate

GPU 前只做 metadata/truth count。需要至少：

- `>=300` qualified paired episodes；
- 独立 heart-size truth 的 positive/negative 各 `>=100`；
- AP/PA 两侧均覆盖 clear 与 borderline apparent-size strata；
- 无 intervention/time-order confounding 的 blind audit sample `>=60`；
- control claims 每个至少 `100` 支持和 `100` 反驳。

当前 VinDr count 是 **0**：`ViewPosition=0`、StudyUID=0、SeriesUID=0，不能进入 model scoring。

### 5.4 层级与因果预测

行为上先拟合 held-out interaction：

```text
cardiomegaly_commitment ~ independent_truth + apparent_CTR
                         + projection + apparent_CTR:projection
                         + image_quality + (1|patient)
```

PCEM 要求模型错误不是由 image quality、native margin 或 projection prevalence 单独解释，而是缺失/反向的 `CTR:projection` 条件项；同一机制不得出现在 geometry-insensitive controls。

层级上学习两个控制 truth 后的坐标：projection `v_l` 与 apparent morphology `g_l`。必要预测是：

- `v_l` 与 `g_l` 都可解码，但正确的 `g_l × v_l` conditional readout 在 projector-to-decoder transition 下降；
- 对 matched borderline pairs 做 residual-view activation patch，只改变 cardiomegaly commitment 的 calibration，不改清晰病理 polarity，也不系统改变 control claims；
- norm-matched random、CTR direction、temperature 和 answer-length controls 不能复制；
- layer 可以模型特异，不主张统一 early layer。

若成立，最简 mitigation 是 **projection-conditioned evidence calibration**：只校准 geometry-sensitive claim 的 evidence-to-commitment mapping；OE 中不删除 claims，不改变 `K`。它不是图像风格标准化。

### 5.5 淘汰线

- 模型根本不能识别 projection：perception/metadata-limited，不是 misbinding；
- 加 interaction 对 held-out clinical error AUROC 提升 `<0.03` 或 CI 跨零：机制失败；
- effect 同样出现在 pneumothorax/device controls：只是 generic domain shift；
- patch 同时改变 claim identity/coverage 或 clear accuracy 下降 `>1pp`：方法失败；
- 两模型不能复现：降级为 architecture-specific diagnostic，不进主线。

## 6. Genuine source-prior crossover：概念上干净，当前数据上彻底 NO-GO

第二个、也是 source-domain-center 唯一合法版本，是 **paired provenance-prior crossover**：同一 detector acquisition 经过两个真实、可验证的 institution/vendor presentation pipelines，临床读者确认 finding support 不变；目标模型的**实际训练 corpus**又能按同两个 source 分组，独立估计每个 claim 的 reporting prior。只有当 paired presentation effect 跨 claims 对齐该 fingerprint，并由 polarity-orthogonal source coordinate 的 activation patch 因果中介，才能称 source-style prior。

它不同于普通 shortcut detection的必要条件是：

- same-acquisition paired pixels，而非跨数据集 matching；
- true source metadata，而非从 image classifier 反推 source；
- claim-vector signed crossover，而非 global abnormality/yes bias；
- exact training lineage 与 source-specific responses，而非模型卡猜测；
- fixed coverage OE 与 clear-case safety。

本地 Gate 失败原因是联合的：VinDr source tags 为 0、paired acquisition identity 为 0；Huatuo 与 raw Qwen 的 lineage audit 只能支持 family-level lineage，不能命名 exact parent；PubMedVision source semantic gate 又只有 2/8 claims 达标。现有 PPI 的自然 bridge 已因此正确停止。继续从 raster clusters、Fourier centers 或 public dataset identity 猜 source 会把 population、disease prevalence、report style 与 export pipeline重新混在一起。

若未来取得数据，最小重开条件是 `>=500` same-acquisition dual-presentation pairs、`>=8` source-semantic-qualified claims、每 claim 的 source prior 两侧各 `>=50` 独立 assertions、两位 blinded clinicians 对 presentation equivalence 的 disagreement `<5%`，以及 exact-parent continuation。达不到就不运行 GPU。

## 7. 最终研究决定

1. **不再追 source-domain center mitigation**：本地已有 held-out signed negative，且当前数据无 source truth。
2. DICOM render orbit、style flips、dataset source classifier 只保留为 exclusion controls；不能当 mechanism outcome。
3. 若能补到带 ViewPosition、paired episode 与独立 echo/CT truth 的数据，PCEM 是唯一高审美的 style-origin pivot：研究 acquisition-conditioned evidence law 的 binding failure，而不是“域偏移导致性能差”。
4. 若补不到上述 paired substrate，这条线到此结束；可信负结论比用更多 transforms 调出正结果更有价值。

**Paper ceiling:** 当前材料适合作为主论文的严谨 negative appendix / hypothesis-pruning evidence；单独不足以投稿。PCEM 若在两模型、独立 patient-paired 数据和固定 coverage OE 中获得行为 interaction、层级定位和因果 patch 三重证据，才可能成为 ICLR main idea。source-prior crossover 即使成立，也需 exact lineage 与自然 source bridge 才有 oral 站位。

# Right Region, Wrong Direction：72小时致死门阶段结果

日期：2026-08-06  
当前决定：**纵向主线 NO-GO；静态 VinDr fallback NO-GO；不启动 adapter、LoRA、KD 或 frontier 拟合。**

## 1. 冻结问题

目标是验证：真实同患者病情变化是否在冻结视觉状态中形成局部、有符号的变化信号；这种监督能否训练一个推理时只看单张图像的小型接口，并在未见 VinDr 上改善准确率而不制造新的 FP/FN 交换。

必须依次通过：

1. Availability：正确变化信号存在；
2. Selective controllability：目标 claim 可改善而其他 claim 不明显漂移；
3. Single-image transfer：纵向监督能够迁移到单图推理。

数据完整性门失败时，按预先计划转入一次静态 VinDr fallback；静态门再失败则不训练模型。

## 2. CheXTemporal 数据门

### 2.1 公开表不能构造唯一的 finding-level 有符号目标

`gold_progression_pairs.parquet` 实际包含：

- 1,787 行、197名患者、738个前后检查对；
- 以 `dataset + patient + prior + current + finding` 为键，只有1,497个唯一键；
- 258/1,497（17.23%）键同时出现多个互斥 progression，涉及548/1,787（30.67%）行；
- 把 `New/Worse` 映射为 `+1`、`Stable` 为 `0`、`Improved/Resolved` 为 `-1` 后，仍有225/1,497（15.03%）键同时包含不同符号；
- schema 没有 annotator、reader、adjudication、consensus 或 lesion-location-to-progression 映射。

这不应简单解释成“医生标错”。论文说明 progression 可以按病灶位置标注，所以同一 finding 在不同肺区可能同时改善和加重。真正的问题是：公开的 finding-level 表丢失了位置索引，无法知道哪个方向属于哪个区域。

`gold_bboxes.parquet` 也不能恢复映射：258个冲突键中251个出现在 bbox 表，且251/251在每个互斥 progression 行中重复完全相同的 prior/current 全部框列表。也就是说，表中存在 `Box1...BoxN` 和多个方向，但没有 `BoxK -> direction` 的对应关系。

因此不能：

- 将每行当独立 finding claim；
- 对互斥方向多数投票；
- 用全部框的并集监督某一个方向；
- 把这些行称为专家 consensus。

### 2.2 其他完整性结果

- Silver：282,214行即282,214个唯一键，内部冲突为0；与 gold 的 patient、study、ordered pair 和 pair×finding overlap 均为0。
- 本机 parent images：CheXTemporal gold prior/current 双侧解析为0/1,787，失败预设的≥90%门。
- 本地 Medical-Diff-VQA SFT：train仅248/78,292对、val仅3/4,064对图像可解析。
- 旧 C³ 两个纵向 cohort：137 claims、68 patients、274/274图像可读，但只有 new/resolved，没有 stable，也没有医生复核。
- 223个 MIMIC gold pair 可连接采集元数据；仅82.1% view相同、28.7%尺寸相同，且有1个时间顺序反向；没有device/exposure字段。

正式数据决定为 `FAIL_AND_ENTER_STATIC_VINDR_FALLBACK`。权威产物见：

- `corrected_runs/right_region_wrong_direction/data_audit_agent/audit_hour0_6_v3.json`
- `corrected_runs/right_region_wrong_direction/data_audit_agent/README_v1.md`
- `corrected_runs/right_region_wrong_direction/data_audit_agent/README_v2_ADDENDUM.md`

## 3. 现有纵向缓存的复用边界

旧 Huatuo longitudinal cache 只保存逐层 Yes−No 标量，可复用做 current/prior/difference、wrong-prior、time reversal 和 layer24/native 的 CPU 烟测；它没有 vision patch、projector、LLM第一层或 ROI 向量，不能回答新的 Availability 问题。

旧 final margin 全为0.125的倍数，存在明显 BF16 读出量化。未来若数据修复，必须新跑 vision/projector/LLM1 张量并用 FP32 verbalizer readout，不能把旧 scalar cache 改名为视觉表征。

详细代码路径审计：

- `corrected_runs/right_region_wrong_direction/representation_path_agent/AUDIT_AND_MINIMAL_PATCH_MAP.md`

## 4. 静态 VinDr fallback

### 4.1 冻结设计

- HuatuoGPT-Vision-7B；模型冻结。
- Dev：640 claims；只用于finding-specific方向拟合和层选择。
- Confirmation：1,920 claims，与dev有0 image overlap；0/3和3/3 reader votes形成960个明确claims。
- 视觉读出：每个finding固定的cosine nearest-centroid方向，无超参数搜索。
- 层选择：只看dev，在7/14/21中选择第21层。
- 极简融合：dev标准化后的第21层视觉分数与最终Yes−No margin等权平均。
- ROI-control：在79条有医生框的阳性claim上，比较同剂量医生框token干预与等数量背景token干预的方向性下降。

门槛在分析前冻结：视觉macro AUROC≥0.70、ROI-control paired AUC≥0.70、融合macro balanced accuracy提升≥3pp，三项必须全过。

### 4.2 结果

| 门 | 结果 | 决定 |
|---|---:|---|
| 第21层视觉状态 macro AUROC | **0.7433** | PASS |
| ROI相对背景 paired AUC | **0.5449**，95% CI [0.4615, 0.6282] | FAIL |
| 最终margin macro BA | 0.6875 | — |
| 等权融合 macro BA | 0.7167 | — |
| 融合增益 | **+0.0292**，image-bootstrap 95% CI [+0.0059,+0.0519] | FAIL（门槛+0.0300） |

融合增益有正的置信区间，是值得保存的静态可解码性线索；但它距离门槛只差0.0008，不能因此事后放宽门槛。更致命的是ROI指标接近随机，未证明分数来自正确病灶区域。

第21层的 `visual_mean` 是LLM内部、经过prompt上下文的视觉token状态，不是原始vision encoder或projector输出。因此AUROC通过只支持“中间视觉token可解码”，不支持“正确区域产生正确方向的因果作用”。

可复现产物：

- `anchor/corrected_sgta/analyze_static_vindr_fallback_v1.py`
- `corrected_runs/right_region_wrong_direction/static_vindr_fallback_v1/preregistered_gate.json`
- `corrected_runs/right_region_wrong_direction/static_vindr_fallback_v1/analysis.json`

## 5. 当前能说与不能说

能说：

1. Huatuo第21层视觉token状态在未见VinDr影像上包含finding polarity信息。
2. 一个无超参数的等权融合在独立confirmation上带来约2.9pp macro BA增益，并非纯dev过拟合。
3. 当前ROI干预无法证明该信息来自医生标注病灶；效果与背景控制不可可靠区分。
4. CheXTemporal当前公开序列化不能把局部方向绑定到具体框，因此无法形成冻结计划需要的全局 `d_c` 或局部 `d_{c,r}` 真值。

不能说：

1. 正确病灶已被模型看到但沿错误方向影响回答；
2. 第21层是因果证据层；
3. 纵向监督能够改善单图推理；
4. 当前方法减少开放式医学VLM幻觉；
5. 训练小adapter会保留2.9pp线性读出增益。

## 6. 决定与唯一重开条件

不运行 Static/Temporal/Static+Temporal、LoRA、KD 或 frontier 实验，因为它们会在未通过 Availability/locality 门时把标签或阈值效应包装成机制。

纵向路线只在以下资产同时满足时重开：

1. 官方提供每个 progression 到具体 lesion location/bbox 的映射，或给出可验证的finding-level adjudicated direction；
2. 受许可 parent images 的prior/current解析率≥90%；
3. 至少三个finding在 `-1/0/+1` 三档具有足够独立患者；
4. patient/image/pair在gold、silver和VinDr间严格隔离；
5. 新采集vision/projector/LLM1状态和FP32 margin，而非复用旧BF16标量。

当前最有价值的新问题不是“如何清洗258个冲突”，而是：**医学VLM的一个全局finding claim能否表示同一疾病在多个区域同时发生相反变化？** 这是数据揭露的真实空间绑定问题，但在公开release丢失方向—位置对应关系时只能作为下一轮数据请求，不得先写成方法贡献。

## 7. 最新工作与代码碰撞

方向性纵向表征本身已经拥挤：

- [TILA, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ko_Temporal_Inversion_for_Learning_Interval_Change_in_Chest_X-Rays_CVPR_2026_paper.html) 已用时间反转学习方向敏感性；
- [ProTrans, 2026](https://arxiv.org/abs/2606.15938) 已把进展写成 `finding-position-progression` directional semantic transition；
- [Med-ST, ICML 2024](https://arxiv.org/abs/2405.19654)、[PLURAL, MIDL 2024](https://arxiv.org/abs/2402.08966) 与 [BioViL-T, CVPR 2023](https://arxiv.org/abs/2301.04558) 已覆盖纵向训练向单图任务迁移；
- [VTI, ICLR 2025](https://openreview.net/forum?id=LBl7Hez0fF)、[PND, CVPR 2026](https://arxiv.org/abs/2605.06679) 与 [CORAL, 2026](https://arxiv.org/abs/2607.03647) 已分别占据latent steering、正负视觉路径解码和医学hard-negative LoRA。

所以不能再声称“首次学习变化方向”“首次做时间反转”或“首次将纵向训练迁移到单图”。目前未检索到的最窄合取delta仅是：**真实pair监督训练、current-only单图推理、冻结自回归医学VLM，并在固定claim coverage下同时控制FP、FN和off-target harm。** 当前数据门失败，所以这一delta仍是未验证命题，不是已有贡献。

代码审计中，Med-ST训练管线最完整；PLURAL有Apache-2.0 OFA代码；TILA只有模型、权重和推理；VTI的LLaVA hook可复用；ProTrans当前入口缺源码且反向路径被注释；PND仅README；CORAL、TempA-VLP和TAMM未找到可验证的官方训练实现。

完整机制碰撞矩阵与官方资源状态：

- `corrected_runs/right_region_wrong_direction/collision_agent/COLLISION_AND_CODE_AUDIT.md`
- `corrected_runs/right_region_wrong_direction/collision_agent/SOURCE_URLS.tsv`

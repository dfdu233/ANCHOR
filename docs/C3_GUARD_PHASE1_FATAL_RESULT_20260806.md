# C³-Guard 第一阶段致死实验

日期：2026-08-06  
结论：**NO-GO；不进入 evidence adapter、Anchor-DG、GNN 或 CRC。**

## 问题

C³-Guard 希望用三类额外信号识别错误医学 claim：

1. 同一张片轻微改变成像风格后，回答是否不稳定；
2. 图像整体是否像模型熟悉的源医院；
3. 删除 claim 对应病灶后，模型证据是否下降。

若这些信号不能比模型原始置信度更可靠，就不应继续训练复杂模块。

## 实验 A：风格稳定性和源域分数能否识别错误

数据是 HuatuoGPT-Vision-7B 在 128 名不同 MIMIC 患者上的二元问题；64 个真值为 Yes，64 个为 No，模型错 30 个。所有错误风险分数均来自 10×5 折患者级嵌套交叉验证；模型选择只发生在训练折。

| 风险特征 | OOF AUPRC | 70% coverage 保留错误率 | 筛掉错误 / 误伤正确 |
|---|---:|---:|---:|
| 原始置信度：`|Yes-No| + entropy` | 0.567 | 12.22% | 19 / 19 |
| + 三种安全风格下的分数方差 | 0.502 | 11.11% | 20 / 18 |
| + 外部 MIMIC-vs-IU 源域分数 | 0.446 | 12.22% | 19 / 19 |

最终可用特征相对基础置信度的 AUPRC 下降 0.121，患者 bootstrap 95% CI 为 `[-0.208, -0.006]`。同覆盖率风险改善为 0，correction/harm 为 1.0，全部 Go 条件失败。

外部源探针本身能区分 MIMIC 与 IU-Xray（分组 CV AUROC 0.938），但加入后更难识别错误。这说明“来自哪个域”与“当前 claim 是否有正确证据”不是同一件事。

可信性限制：这些 128 条是二元问题，不是自然开放报告；其中有复合和历史比较问题。现有 354 张公开肺炎/气胸框标注与 128 张影像交集为 0，因此没有伪造 grounding 或 lesion 特征。

## 实验 B：医生病灶框擦除

在 VinDr 上选择 128 个由至少两名医生标框的阳性 claim：64 个结节/肿块、64 个钙化。比较原图、模糊医生框、模糊对侧同形区域。

| 指标 | 均值 | image-cluster bootstrap 95% CI |
|---|---:|---:|
| 删除医生框后的分数下降 | +0.029 | `[-0.049, +0.109]` |
| 删除对侧框后的分数下降 | −0.100 | `[−0.153, −0.049]` |
| 医生框相对对侧框差值 | +0.129 | `[+0.063, +0.199]` |

结节/肿块的医生框擦除均值反而为 −0.047；只有钙化为正。视觉审查发现模糊操作会产生明显灰色斑块，因此“医生框与对侧框不同”不能解释为可靠病灶因果证据。

## 实验 C：更严格的结节删除—搬运

进一步只保留 64 个小型、单发、至少两名医生位置重合的结节/肿块。用同一患者的对侧组织替换病灶，再把原病灶搬到对侧。预期是：删除降分，搬运后恢复。

| 图像 | 平均 Yes−No |
|---|---:|
| 原图 | 0.002 |
| 删除病灶 | 0.086 |
| 病灶搬到对侧 | 0.293 |

- 删除后的分数变化为 `原图−删除 = −0.084`，95% CI `[-0.227, +0.047]`，方向错误；
- 仅 22/64（34.4%）同时满足“删除降分、搬运恢复”；
- 搬运结果显著超过原图 `+0.291`，更像位置或编辑响应，而非恢复原临床证据。

## 决定

当前三项核心信号都未通过：

- 风格方差没有超过基础置信度；
- 全局源域分数能识别医院，却不能识别 claim 错误；
- 两种病灶干预均未得到跨 finding、方向一致的因果敏感性。

因此当前 C³-Guard 的 feature ladder 不能用于论文主方法，也不能据此训练 evidence adapter。保留的科学结论是：**域可识别性、区域敏感性和 claim 正确性不能相互替代；未经双向因果验证的 counterfactual score 很容易测到编辑伪影或位置先验。**

下一步只允许改变核心测量问题，而不是调阈值：使用真实配对的临床前后状态、合成时有像素级生成真值的病灶，或具有独立阳性/阴性局部证据的资料，重新定义可识别的 claim 因果效应。

## 后续审计：真实患者前后片仍未通过 claim 特异性门槛

为排除人工删病灶的编辑伪影，后续使用同一患者真实的 prior/current
MIMIC 胸片。数据审计先发现两个限制：公开 SFT 镜像实际不含说明页声称的
MS-CXR-T 条目；在排除旧患者、匹配 AP/PA 投照并要求两份原始报告明确呈现
相反极性后，严格可用病例为 0。因此正式 expert-gold 验证不可执行，只允许
把 Medical-Diff-VQA 自动报告差分标签作为可证伪的 silver canary。

Silver discovery 包含 58 条成功 claim、29 名此前未见患者。最终层的
`direction × (current−prior)` 均值为 `+0.233`，患者 bootstrap 95% CI
`[+0.024,+0.466]`，表面上像真实临床信号。然而，同一对片上询问未被变化
列表提及的 off-claim，均值为 `+0.269`；目标 claim 相对 off-claim 的净优势
为 `−0.037`，95% CI `[-0.267,+0.183]`。所以最终层的弱阳性可由全局时间、
病情或采集漂移解释，不是 claim-specific evidence。

层级探索中，五个患者级训练折都选择第 24 层。其 OOF claim-specific
premium 为 `+0.582`，但 95% CI `[-0.215,+1.365]`、`p=0.185`，未成立。
探索性分层显示该现象主要来自 resolved 而非 new claim。随后在读取复现输出
前冻结 `layer=24 + resolved + off-claim common-mode subtraction`：

| 冻结复现（46 claims / 26 patients） | 结果 |
|---|---:|
| Layer-24 specificity premium | +0.615 |
| 患者 bootstrap 95% CI | [-0.192, +1.305] |
| patient sign-flip p | 0.143 |
| 正方向比例 | 60.9% |
| Layer-24 相对最终层增益 | +0.484 [-0.187, +1.065] |

均值方向重复，但三个预注册统计门槛均失败。因此不能声称“正确临床信号稳定
滞留在第 24 层”，也不授权训练纵向 evidence head。保留的弱线索只能用于将来
在独立专家进展标签上做一次性复现，不能进入当前论文结论。

## 可复现产物

- `corrected_runs/c3_guard/phase1_available_features_v1/result.json`
- `corrected_runs/c3_guard/phase1_external_source_score_v1/result.json`
- `corrected_runs/c3_guard/vindr_focal_erasure_n128_v2/analysis.json`
- `corrected_runs/c3_guard/vindr_focal_erasure_n128_v2/visual_audit_extremes.jpg`
- `corrected_runs/c3_guard/vindr_nodule_relocation_n64_v2/analysis.json`
- `corrected_runs/c3_guard/mimic_natural_counterfactual_manifest_v1/audit.json`
- `corrected_runs/c3_guard/huatuo_natural_counterfactual_silver_v1/analysis.json`
- `corrected_runs/c3_guard/huatuo_natural_counterfactual_specificity_v1/analysis.json`
- `corrected_runs/c3_guard/huatuo_natural_counterfactual_layerwise_v1/analysis.json`
- `corrected_runs/c3_guard/mimic_natural_counterfactual_replication_manifest_v1/preregistered_layer24_resolved_gate.json`
- `corrected_runs/c3_guard/huatuo_natural_counterfactual_layer24_replication_v1/analysis.json`

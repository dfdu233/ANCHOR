# 医学 VLM 幻觉项目：实验事实账本

冻结日期：2026-08-14。本文只记录当前证据允许的结论；“有响应”“可解码”和“能缓解幻觉”严格分开。

## 1. 当前总判断

- 尚未确定 ICLR Oral 级方法，也没有一个新方法已经通过多模型、开放生成和无遗漏交换三项门槛。
- 最强的正信号来自专用医学小模型：完整的 18 维疾病状态能补充部分 VLM 的病例信息。
- 最稳定的负规律是：许多干预改变了回答倾向，却没有增加病例级判别信息，即 `response is not evidence`。
- 风格/DG、中间层融合、局部 crop、RAG 和一般阈值修补均不能作为当前方法主线。

## 2. 已确认事实

### A. 层间解码

1. LET 在 3,470 个 CE claims 上将 FN `270 -> 64`，但 FP `420 -> 542`；BAcc 仅 `77.96% -> 78.23%`。它主要把工作点推向阳性，不是干净的幻觉缓解。
2. 盲确认中，非最终层相对最终层的 macro AUROC 差值为 `-0.109`，95% CI `[-0.165,-0.079]`。因此“正确临床证据普遍藏在更早层”不成立。
3. observer update、邻居几何、secant stitching、全局视觉池化增量均未通过病例级增量门。

### B. 风格与 DG

1. 128 例 CE-D 中，三种风格变换的答案翻转率为 `3.13% / 2.34% / 1.56%`，全部低于预注册的 5% 现象门。
2. 风格漂移预测错误 AUROC 为 `.425`，原始 margin 为 `.798`。
3. FedDG oracle headroom 仅 `4/27`；DICOM rendering `0/4` findings 通过。

结论：风格/源域中心不是本项目中可复现的主因，不能靠继续调超参数复活。

### C. 局部病灶与 crop

1. 病灶面积越小，VLM 的正确支持越弱；病种内 Spearman 为 Huatuo `.323`、Hulu `.415`，控制可测混杂后仍为 `.239/.420`。
2. 但 sparse scan 相对 `final+mean+max+top5` 仅增加 `.0040` AUROC，CI `[-.0194,.0273]`。
3. 在 62 张七 finding 全阴性胸片上，完整图 FP 为 `8.1%`，移除 ROI 外上下文后升至 `71.0%`；单纯放大尺度只增加 `3.2pp`，CI 跨 0。
4. 告诉模型“crop 是随机选择的”虽把 crop FP 从 `67.7%` 降到 `6.5%`，却把阳性 recall 从 `91.9%` 降到 `32.3%`。

结论：小病灶弱支持是真实现象；crop 主要改变判定标准并破坏完整解剖反证，尚无可用修复算法。

### D. RAG 与 claim exchange

1. 现有 RAG n=200 可作为 baseline，但真实检索没有稳定超过 shuffled/placebo 控制。
2. 固定 K 的 specialist one-swap 在确认集使总错误增加：Huatuo `+3.35%`，Hulu `+19.43%`。

结论：现有检索和“拿高分 claim 换低分 claim”都不是主线。

### E. 专用医学小模型

VinDr 七个 finding、image-disjoint dev/test、每模型 280/840 claims：

| 输入 | Huatuo AUROC | Hulu AUROC |
|---|---:|---:|
| VLM final margin | .7667 | .8606 |
| + XRV 对应 finding 标量 | .8264 | .8708 |
| + XRV 完整 18D 状态（模型特定监督融合） | .8599 | .8873 |

- 完整状态相对 VLM 的增益为 Huatuo `+.0934 [.0620,.1251]`、Hulu `+.0265 [.0033,.0496]`。
- Huatuo FP/FN `124/123 -> 77/104`；Hulu `75/98 -> 60/92`。
- 但 18D 方向跨 VLM 迁移只保留相对 scalar 的 `27.7%/15.3%` 增量，CI 均跨 0。
- 专家独立 likelihood-ratio state 相对 scalar 的增量仅在 Huatuo 显著：`+.0338`；Hulu `+.0023`，不显著。
- one-bit veto 可去掉约 17.4% FP，但 TP 损失超过冻结门槛。
- conditional likelihood-ratio / residual innovation 在数学上等于相关分类器的 Bayes stacking；它不能作为新融合原语。
- “write-protected visual memory”不改变现有三种 decoder-only VLM：causal prefix 与 append-only KV cache 本就禁止后续语言改写视觉前缀；纯保护必然与 greedy token-exact。

结论：小模型确实携带互补病例信息，尤其能帮助较弱 Huatuo；但现有强结果是监督 stacking 上限，不是通用算法。下一步的核心是寻找 VLM-agnostic 的专家消息，而不是继续训练融合头。

### F. 开放报告的语义接口

- IU-Xray 参考报告中 1,215 个疾病词提及有 `90.62%` 是否定语境，只有 `8.31%` 是明确阳性；MIMIC 中否定提及占 `28.86%`。
- 因此提升单个疾病 token（如 `effusion`）无法区分 “pleural effusion” 与 “no pleural effusion”。

结论：token 不是临床 claim；任何专家引导必须至少识别 finding、极性和不确定性。但这只是接口约束，不等于已有新方法。

## 3. 仅属弱信号

- 小病灶面积是可靠相关因素，但未证明因果，简单局部扫描无效。
- 完整专家状态优于单一专家票在 Huatuo 上显著，在 Hulu 上新增部分不显著。
- 报告中的设备、骨折、结节等错误呈模型特异模式；当前自动词法审计不能替代临床真值。

## 4. 已关闭方向

- 风格一致性、训练域中心、FedDG envelope/LoRA。
- 统一早层证据丢失、固定或动态层融合、Evidence Addressability 全局池化。
- NCD/ISD/CMEI 双中心化与 common-mode subtraction 作为缓解算法。
- crop max、sparse patch scan、context completion、observation-policy prompt。
- raw score transport、固定 K claim exchange、one-bit veto、syndrome/checksum、相对疾病排名。
- clinical-null residual/diffusion amplification：健康反事实残差和 counterfactual diffusion 已有强邻近工作，且输入增强存在制造伪证据风险。
- specialist likelihood-ratio、innovation residual、通用 18D operator 与纯 write-protected memory。
- luma/chroma、Riesz/Haar 或 high-bit residual 通道编码：本地 L0 未过临床增量/特异性门，且数学属于标准多通道编码与 multiresolution 表示。

## 5. 可信度边界

- **高可信**：VinDr reader vote/box、image-disjoint split、配对或 image/patient bootstrap 的 CE 结论。
- **中可信**：自动标签支持的报告 finding 统计，可作筛选与相对比较。
- **低可信**：OE/report 的词面分数或正则表达式审计；没有临床医生时不能称真实 hallucination rate。
- 任何 test 上选超参数、无 matched coverage、或只改变输出长度/阳性率的结果不得进入主结论。

## 6. Baseline 现场状态

- 最新 coverage audit：336 个主矩阵 cells 中 `64 completed`、`171 N/A`、`97 pending`、`3 running_or_partial`、`1 generated_unscored`。
- 当前无 GPU 推理进程。Hulu–DoLa IU-Xray 已保存 `274/590`，可从断点恢复。
- baseline 的监控外壳仍在；在新主线完成低成本门控后恢复生成，不删除任何现有产物。

# 医学 VLM 幻觉：本地证据再审计、领域复盘与 Negative-Control Decoding

日期：2026-08-11

> **关键更正（同日）：** 本文最初把 `visual_mean:21` 的提示性融合结果写成了主要正支柱。
> 重新核对后，正式 evidence-admission blind gate 为 `claim:7` 对 `claim:28`，macro AUROC
> `delta=-0.1091`，明确 NO-GO；等权融合 `+2.9167pp` 也未达到预注册 `+3pp` 且 ROI 门失败。
> 两个数字来自不同 readout/指标，并非算术矛盾，但原先的正向叙事过强。NCD-as-mitigation
> 已暂停，权威后续协议改为 [Intervention Specificity Protocol v2](./INTERVENTION_SPECIFICITY_PROTOCOL_V2_20260811.md)。

## 0. 结论先行

当前还没有得到一个已验证的、跨模型降低医学 VLM 幻觉的新算法。最可靠的发现不是
“风格中心”“病灶掩码”“统一早层”或“reader clarity erasure”，而是：

> 许多干预确实让模型改变答案，但变化中相当大一部分是对所有病种共同的回答工作点
> 移动，而不是针对目标病种的正确临床证据。

因此后续不再直接验证 NCD 提分，而是先做 **Intervention Specificity existence gate**：把
干预响应分解为 method、image、claim 和 patient×claim interaction，先证明最后一项包含
baseline 之外的条件标签信息。若不存在该信息，任何负对照 decoder 都没有方法论文价值。

## 1. 本地结果：哪些已经可信，哪些不能继续讲

### 1.1 提示性结果与正式门的边界

| 结果 | 规模和效应 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| 中间 readout 的提示性互补 | `visual_mean:21` AUROC 0.7433；与 final margin 等权融合 BAcc 0.6875→0.7167，+2.92pp，95% CI [+0.59,+5.19] | 不同 readout 可能有不同误差 | **未通过正式门**：blind-selected `claim:7` 相对 `claim:28` AUROC −0.1091；融合未达到预注册 +3pp；ROI AUC 0.5449。不能作为 mitigation 支柱 |
| Source-margin calibration 能提 CE 分数 | Huatuo，3,470 题，Acc +3.69pp；但 192 题外部确认 CI 跨 0 | 强 calibration baseline | 不是 DG，也不是幻觉机制 |
| RAG/router/stacking 能提 benchmark proxy | CXR router BAcc +2.58pp，CI [+1.62,+3.53]；Knowledge stack +1.44pp，CI [+0.19,+2.70] | 竞赛式融合有互补预测信息 | exact-question placebo 不弱，因此不是患者特异临床 evidence code |

### 1.2 可信负结果

| 被否定的方向 | 核心结果 | 判决 |
|---|---|---|
| FedDG “训练域中心” | 128 题，多组 lambda/selector；最好只打平 baseline，无 wrong→correct，部分 correct→wrong | 不是单一超参数问题；域变化方向不等于通往正确中心 |
| 多风格平均/轨道不变量 | 单风格 flip 1.56%–3.13%，全部集中在低 margin；style drift 预测错误 AUROC 0.425–0.446，原始 margin 0.798；VinDr 5 renders BAcc 0.725→0.7125 | 风格能改答案，但当前变换没有增加可用真值信息 |
| Domain-Orbit 投影 | rank-2 解释约 89% 风格方差、可衰减 86%–92%，但所有 rank rescue=0 且有临床泄漏 | “低维 nuisance”不等于“可安全删除”；不能重新包装成 robust median/quantile |
| 病灶 mask/搬运 | bbox 相对对侧响应为正，但删除方向错误；严格 relocation 仅 22/64 同时通过删除与恢复，搬运图甚至强于原图 | 响应主要受编辑伪影/位置先验影响，不是病灶因果证据 |
| 病灶/低视觉 head suppression | domain+lesion 相对 domain-only 更差；全阳性集上低视觉 head 主要统一推高 Yes | 看更多图不等于看对病灶 |
| 统一 late commitment / reader two-plane | Huatuo early-final clarity 差值 -0.040，CI [-0.198,+0.115]；Hulu 也没有统一机制 | 不存在已证实的跨模型 clarity erasure |
| Evidence Recoverability / ETD | raw 结果被早层 always-Yes 造成；校准后四个 FP/FN 分组均不优于 matched-null | “某层曾出现正确符号”不是正确证据，不能启动 ETD |
| Fixed-K raw claim exchange | pooled precision +1.54pp，CI 跨 0；MIMIC supported recall 26.2%→19.0%，unverified 增加 | 不能靠 ontology rerank 修复开放回答 |
| 纵向/来源/患者特异响应 | target temporal response +0.233，而 off-claim +0.269；OTHER patient 的疾病文本影响不弱于 CURRENT patient；patient code 被 exact-question placebo 解释 | 模型会响应上下文，但响应没有 patient×claim 特异性 |

### 1.3 只能作历史或进度、不能写成科学结论

- 旧 LLaVA-Med 3,466 题表中 Beam +3.55pp、OPERA +2.05pp、VCD -6.03pp 等结果使用旧共同流程，
  只作历史基线；正式 v1 矩阵仍在运行。
- LET 的历史 Acc 可提高 +5.30pp，但主要通过 Yes-rate 上升减少 FN，同时制造 FP；另一 native
  路径 BAcc 仅 +0.27pp。因此它证明“层融合会搬动工作点”，不证明恢复了临床证据。
- 没有临床医生 review 时，OE/report 的词面、RadGraph、RaTEScore、GREEN 等只能称 benchmark
  proxy，不能称真实临床 hallucination rate。

## 2. 能统一多数失败的机制

令模型对图像 `i`、claim `c`、干预 `a` 的有符号 margin 为

\[
m_{ica}=s_{ic}+b_c+g_{ia}+h_{ica}+\epsilon_{ica}.
\]

- \(s_{ic}\)：patient×claim 临床信号；
- \(b_c\)：病种、措辞和答案基率；
- \(g_{ia}\)：干预令大量 claims 共同产生的工作点移动；
- \(h_{ica}\)：目标病种特异的干预响应。

多数方法只观察

\[
\Delta_{ica}=m_{ica}-m_{ic0}=g_{ia}+h_{ica}+\epsilon_{ica},
\]

却把整个 \(\Delta\) 都解释为 evidence。独立反例包括 LET/低视觉 head 的全局 Yes 推动、
target temporal response 不超过 off-claim、recoverability 不超过 matched-null、患者 RAG 不超过
exact-question placebo，以及 OTHER patient priming 不弱于 CURRENT patient。

所以当前最可信的机制结论是：

> **Response is not evidence.** 只有超过同图像 off-claim、same-truth 和 exact-question
> 对照的剩余响应，才有资格被称为临床证据。

## 3. 2024–2026 文献地图与真正空缺

### 3.1 已经拥挤的路线

1. **图像对比/掩码**：[VCD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html)
   使用原图与破坏图对比；[SECOND, ICML 2025](https://openreview.net/forum?id=SbyrpBNNs4)
   已做选择性、多尺度精细掩码；[Med-VCD, 2026](https://arxiv.org/abs/2512.01922) 已迁移到医学。
2. **层/头干预**：[SPIN, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.631/)
   动态抑制低图像注意头；[HALP, EACL 2026](https://aclanthology.org/2026.eacl-long.287/)
   表明最有预测力的层和模态随架构变化。
3. **因果路径与 counterfactual**：[HalluTrace, ALVR 2026](https://aclanthology.org/2026.alvr-main.29/)
   先诊断来源再路由干预；[How Many Counterfactuals, 2026](https://arxiv.org/abs/2606.08777)
   研究反事实影响和样本复杂度；[Causal Decoding, 2026](https://arxiv.org/abs/2602.21441)
   显式建模视觉 object belief。
4. **训练对齐**：[OPA-DPO, CVPR 2025 Oral](https://arxiv.org/abs/2501.09695) 使用 on-policy
   幻觉数据和专家反馈；[DEPO, ACL 2026](https://aclanthology.org/2026.acl-long.1200/)
   用动态多模态扰动构造偏好。
5. **检索、集成和评测**：[MedHEval](https://arxiv.org/abs/2503.02157) 在 11 个模型和 7 种
   方法上发现现有 mitigation 尤其难处理知识/上下文错误；[HalluCXR, 2026](https://arxiv.org/abs/2605.20469)
   的多模型 ensemble 可显著减少 fabrication，但会增加 omission。

### 3.2 对本项目最重要的新证据

- [System-Mediated Attention Imbalances Make VLMs Say Yes, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1940/)
  证明注意力干预会系统性改变 Yes-rate；这与本地 LET/head 结果一致，说明操作点漂移必须被单独控制。
- [Looked but didn't see, 2026 preprint](https://pubmed.ncbi.nlm.nih.gov/42369483/) 指出，判断 VLM
  “看到了某物”必须配 matched-control false-alarm baseline；这支持负对照原则，但该工作没有提出
  claim-level difference-in-differences decoding。
- [HalluCXR](https://arxiv.org/abs/2605.20469) 和本地 fixed-K 结果共同说明，任何只降低输出数量的
  方法都会把 fabrication 转换成 omission，不能当作真正缓解。

在已记录的关键词与相邻工作搜索中，没有找到“同图像 off-claim placebo 响应用于识别并修正
目标 claim 干预效应”的机制等价工作。但这只支持“当前未检索到”，不能写成绝对首创。

## 4. 候选方法：Negative-Control Decoding（NCD）

> 本节保留为历史的一维版本。它只去掉 image-wide response，未去掉 claim-wide response，
> 且尚无 `h>0` 证据；正式方案已升级为 v2 的 two-way Interaction Specificity Decomposition。

### 4.1 一句话

> 若一次干预让“胸腔积液”分数上升 0.8，但也让心大、肺炎、气胸等无关 claims 平均上升
> 0.7，那么可归因于胸腔积液的有效证据只有约 0.1，而不是 0.8。

### 4.2 算法

对目标 claim \(c\)，用任一候选 evidence intervention \(a\) 计算

\[
\Delta_{ic}=m_{ic}^{a}-m_{ic}^{0}.
\]

两路若来自同一 decoder/counterfactual，margin 天然同尺度；若比较中间视觉读出与 final margin，
则只用冻结 dev reader votes 为每个 finding 做单调 location/scale 映射，先落到同一支持尺度，测试集
不得重新校准。

用同图像其他病种作为 placebo outcomes：

\[
\hat g_{i,-c}=\operatorname{median}_{k\in\mathcal C\setminus\{c\}}\Delta_{ik},
\qquad
\hat h_{ic}=\Delta_{ic}-\hat g_{i,-c}.
\]

最终只加入病种特异残差：

\[
\tilde m_{ic}=m_{ic}^{0}+\lambda\hat h_{ic}.
\]

第一版曾冻结 \(\lambda=1\)，但不再默认使用中间视觉读出作为 evidence source；必须先通过
v2 的 patient×claim conditional-information existence gate。

CE 直接按 \(\tilde m\) 判定。OE/report 先把草稿拆成固定 ontology claims，保持原草稿阳性 claim
数 \(K\) 不变，只允许用高 \(\tilde m\) 的遗漏 claim 一进一出替换低 \(\tilde m\) 的草稿 claim；
certainty 单独调整。这样不能靠缩短、全阴性或拒答获益。

### 4.3 非平凡数学结论

**命题 1：单一反事实不可识别。** 对单个观察 \(\Delta=g+h\)，任意 \(\eta\) 都有
\((g+\eta)+(h-\eta)=\Delta\)。因此 VCD/层差/attention response 只凭一个目标 claim，原则上
不能区分“全局工作点变化”和“目标临床证据”。

**定理 1：负对照的稳健识别。** 假设

\[
\Delta_{ik}=g_i+h_{ik}+\epsilon_{ik},
\]

且超过一半 placebo claims 满足 \(h_{ik}=0\)、\(|\epsilon_{ik}|\le\varepsilon\)。则

\[
|\hat g_i-g_i|\le\varepsilon,
\qquad
|\hat h_{ic}-h_{ic}|\le2\varepsilon.
\]

因此当 \(|h_{ic}|>2\varepsilon\) 时，NCD 恢复的病种特异响应符号必然正确。证明只需注意：
超过半数 placebo 响应落在 \([g_i-\varepsilon,g_i+\varepsilon]\)，所以其中位数也落在该区间；
再由三角不等式得到第二个界。中位数同时是 \(L_1\) 位置估计器并具有 50% breakdown point。

**定理 2：任意加性域漂移不变性。** 若域/风格令所有 claim response 同加 \(a_i\)，即
\(\Delta'_{ik}=\Delta_{ik}+a_i\)，则

\[
\Delta'_{ic}-\operatorname{median}_k\Delta'_{ik}
=\Delta_{ic}-\operatorname{median}_k\Delta_{ik}.
\]

这不是声称存在“训练域中心”，而是证明不需要找到中心：只要域影响是 claim-common 加性项，
NCD 对它精确不变。

### 4.4 它和相邻方法的本质区别

| 方法 | 用什么作对比 | 未识别的混杂 |
|---|---|---|
| VCD/SECOND/Med-VCD | 原图 vs 破坏图 | 破坏图引起的全局 Yes/No 或 verbosity shift |
| LET/VISTA | 中间层 vs 最终层 | 层级 base-rate 与 affirmative bias |
| SPIN | 保留头 vs 抑制头 | head suppression 对所有 claims 的共同方向 |
| RAG/ensemble | 无检索/单模型 vs 有检索/多模型 | 问题先验、保守化与少说 |
| NCD | 目标响应 vs 同图像 placebo-claim 响应 | 在“多数 placebo 无 claim-specific effect”假设下识别 common mode |

## 5. 致死验证，而不是继续堆模块

### 5.1 第一轮（两模型、数小时量级）

- 数据：VinDr，冻结 320 张 image-disjoint 图像；8 个 findings，每张图全部评分，不再复用每图只含
  1–5 claims 的旧 cache。
- 模型：Huatuo、Hulu；相同的 claim prompt、answer tokens 和 token budget。
- evidence channels：final、gate-failed 的中间视觉读出（仅作候选）、naive 50/50 fusion、VCD、random/norm-matched。
- 方法：raw intervention、finding-only scalar calibration、mean control、median NCD、shuffled placebo。
- 真值：VinDr 独立 reader votes；0/3、1/3、2/3、3/3 分层；没有医生 review 也可完成受控 CE。
- 指标：BAcc、FP、FN、reader-vote Brier/NLL、每图预测阳性数；按图像 bootstrap 5,000 次。

### 5.2 预注册通过门

NCD 只有同时满足以下条件才进入 OE：

1. 相对 naive fusion 至少 +1pp BAcc，image-bootstrap 95% CI 排除 0；
2. 相对 greedy，FP 与 FN 不能靠一升一降制造总分；clear case 最多下降 1pp；
3. 优于 finding scalar calibration、shuffled placebo、random residual；
4. 两个模型、多数 findings 同方向；
5. fixed-K 后仍改善，reader-vote Brier 相对至少改善 5%。

失败即淘汰。成功后才扩展 LLaVA-Med、Qwen2.5-VL，随后做 VinDr OE listing 和报告 claim；无医生
review 时只把后两者称为自动/benchmark 迁移证据。

### 5.3 已完成的旧缓存烟测（不能替代正式实验）

为了先检查是否存在明显反例，复用了 confirmation 中仅有的 82 张“至少缓存 3 个 claims”的图像，
共 104 个明确 0/3 或 3/3 claims；固定中间层、固定等权融合、固定 \(\lambda=1\)，没有看结果调参。

| 模型 | Native BAcc | Naive fusion | 不完整 ontology NCD | NCD−Naive 95% CI |
|---|---:|---:|---:|---:|
| Huatuo | 0.6436 | 0.6564 | 0.6641 | [-0.0543,+0.0654] |
| Hulu | 0.7564 | 0.7128 | 0.7256 | [-0.0313,+0.0584] |

两模型的 NCD 都只比 naive fusion 略高，置信区间均跨 0；Huatuo 的 FP 没下降，Hulu 的 native
仍明显最好。因此烟测只说明公式和数据路径可运行，**没有提供 NCD 有效的统计证据**。由于每图
只观察 3–5 个、且并非预先固定的 placebo claims，中位数假设也未被满足；正式 8-finding cohort
仍是必要的致死实验。

## 6. ICLR 潜力的客观评价

旧 NCD-as-method 的当前评分改为 **Reject and Pivot**；保留的是 Intervention Specificity
机制问题，其状态为 **Accept with major revisions，等待 h-existence gate**。

- Higher：7/10。若能证明多种现有 mitigation 的大部分 response 是 common mode，并用 NCD
  修复其 FP/FN 交换，这是机制级发现；若只在 LET 上多 1–2 个点，就是普通 calibration。
- Stronger：8/10。定理给出 additively domain-invariant 和 50% contamination robustness。
- Cheaper：7/10。不训练主模型、不需外部大 verifier；代价是每图评分固定 ontology。
- Broader：6/10。适合可规范化 claim 的 VQA/report，不覆盖外部知识、治疗建议和任意开放推理。
- Faster：5/10。可 teacher-force 批量评分，但不是零额外开销。

最大创新风险：审稿人可能把它概括成“robust centering/calibration”。唯一有效防线不是增加公式，
而是完成三层证据：

1. **现象**：跨模型、跨 intervention 证明 raw response 的 common-mode 占比和错误代价；
2. **理论**：证明单一 contrast 不可识别，placebo claims 在明确假设下给出有限样本稳健界；
3. **方法**：在固定覆盖下同时降低 fabrication 和 omission，而不是只让模型少说。

若三层都成立，论文故事可写成：

> Current hallucination decoders mistake responsiveness for evidence. Negative-control claims identify
> and reject the intervention-wide common mode, enabling claim-specific correction without sacrificing coverage.

若第一轮两模型门失败，就承认中间视觉互补信息不是可分离的 claim-specific residual，停止该主线。

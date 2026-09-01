# ANCHOR 本地证据总审计（2026-08-11）

## 0. 审计口径

本报告只总结已经产生真实模型输出、统计量或失败门槛的核心实验。仅有协议、单元测试、
文献构想或等待医生回传的分支不计为实验结论。当前没有临床医生复核，因此自然 OE/报告
实验只能报告 benchmark proxy，不能称为真实临床幻觉率。

可信度分级：

- **A**：独立确认或完整测试集，预先冻结分析，按图像/患者聚类统计，有关键替代对照；
- **B**：完整或较大实验，但只有一个模型、旧评测版本、弱真值或关键外部复现未通过；
- **C**：小样本 discovery/smoke，只能产生下一实验；
- **D**：管线通过、合成验证或无临床标签，不构成效果证据。

## 1. MECE 结果账本

| # | 方向 | 数据、模型与规模 | 主要结果 | 最简单替代解释/对照 | 可信度 | 决定 |
|---|---|---|---|---|---|---|
| 1 | 通用 decoding baseline | LLaVA-Med，RULE/MIMIC，旧共同 3,466 题 | Beam 比 greedy `+3.55pp`；OPERA `+2.05pp`；DoLa `-0.09pp`；VCD `-6.03pp`；M3ID `-7.04pp`；PAI `+0.12pp` | prompt、token budget 和 answer operating point 本身可大于方法增益；该表不是当前 v1 正式矩阵 | B | 保留作历史基线，不作当前论文主表结论 |
| 2 | LET / 中间层正向融合 | LLaVA-Med，RULE/MIMIC，3,470 题 | 历史统一生成路径 `75.45%→80.75%`，`+5.30pp`，patient-cluster CI `[+4.10,+6.55]pp`；但 sensitivity 大升、specificity 降到约 63.9%。另一 native 版本 Acc `+2.42pp`，BAcc 仅 `+0.27pp` | 类别阈值/affirmative bias；算子与 VISTA SLA 直接重合 | B | 证明“层融合会改变工作点”，不能证明普遍恢复正确证据；淘汰为主算法 |
| 3 | Source-margin 校准 | Huatuo，RULE/MIMIC，3,470 题 | 相对 greedy `75.56%→79.25%`，`+3.69pp`，patient CI `[+2.34,+5.01]pp` | 本质是 source-dev 选出的 claim threshold；外部 192 题 reconfirm 为 `+4.69pp` 但 CI `[-1.04,+10.42]`、门槛失败 | B | 保留为强 calibration baseline，不作 DG/幻觉机制 |
| 4 | FedDG / 训练域中心 | Huatuo，RULE/MIMIC，128 题；`lambda={.003,.01,.03}`，多种 selector | baseline `76.56%`；原 selector `75.78%`；重选最好与 baseline 相同；switch 版 `74.22%`。没有 wrong→correct，最多出现 correct→wrong | 不同超参数已改变被选风格但未产生 rescue；不是单一超参数没调好 | B | 训练中心假设淘汰；保留 source shift 作为 stress axis |
| 5 | 轻微风格变化与多 render | Huatuo，MIMIC 128 题；另 VinDr 80 个 unanimous claims、5 renders | 单风格 flip `1.56%–3.13%`，均未过 5% 门；error AUROC：原始 margin `0.798`，style drift `0.425–0.446`。多 render BAcc `0.725→0.7125`，CI `[-7.50,+5.01]pp` | 普通低 margin 边界穿越；全部 6 个 flip 都在低-margin 半区 | A-（现象窄） | 风格会改答案，但未增加可用正确性信息；不再做 cosmetic ensemble |
| 6 | Evidence-DG / 域识别预测错误 | Huatuo；report 48，CE 64 | source 可被像素/视觉特征高准确识别（report 68.8%–81.3%，CE 85.9%–95.3%），但 source-controlled quality 信号弱且不跨 CE/OE；C³ 中 source probe AUROC `0.938`，加入错误检测却使 AUPRC `0.567→0.446` | “像哪个医院”不等于“这个 claim 是否正确” | B | 域身份不是可靠 hallucination risk；淘汰 source distance gate |
| 7 | Domain-orbit SVD / nuisance removal | Huatuo，VinDr 16 claims，DICOM 与 source-radial 两种 orbit | rank-2 解释约 89% orbit 方差，held-out attenuation 约 86%–92%，但所有 rank rescue=0，clear aligned margin 全为负；随机/均值控制不弱于 DOC | 低维可压缩不等于可安全删除；风格切空间也含临床信息 | C | 淘汰 DOC 与 token-stability gate |
| 8 | 病灶框 mask、删除与搬运 | Huatuo，VinDr：bbox blur 128 阳性；严格 nodule relocation 64 | 医生框相对对侧框差值 `+0.129 [+.063,+.199]`，但病灶删除方向错误；严格 relocation 仅 `22/64` 同时满足删除降分/搬运恢复，搬运反而比原图 `+0.291` | 编辑斑块、位置先验和全局图像扰动；不是病灶因果证据 | A- | 淘汰 lesion-mask evidence adapter；任何 ROI 方法必须双向因果+off-region control |
| 9 | Image-guided/lesion-aware head suppression | Huatuo，VinDr，48 个 3/3 bbox positives，6 styles；confirmation 16 | domain+lesion 最差风格 margin `+0.0567 [-.0586,+.1722]`；相对 domain-only `-0.0860 [-.1762,+.0113]`；病灶 head 没有增量 | 低视觉 head 在全阳性集统一推高 Yes margin；缺少阴性安全性 | B- | 淘汰病灶 head 主张；低视觉 head 只作 affirmative-bias control |
| 10 | SITH 权重奇异方向 | Huatuo CLIP，VinDr-like 500 图，4 findings+normal | VO 方向相对随机最大 AUC 优势仅 `+0.0031/+0.0035`；同图 pooled ridge AUC 为 `0.612–0.774` | 文本命名和随机方向同样偏向 device/artifact 词 | A- | 视觉塔有疾病信息，但原始权重奇异方向不是可信临床坐标；淘汰 |
| 11 | Claim Plane / scalar CBD | Huatuo/Hulu，真实 CXR 16–32，grade-C truth | polarity 与 commitment 几何上非一维；但 CBD：Huatuo `87.5%→56.3%/43.8%`，Hulu `100%→75%`；晚层 commitment 单调增长的符号也相反 | final margin/普通 calibration 比 visual-null commitment 更能解释错误 | C | 几何保留为评测语言；CBD 与“统一晚层过度承诺”淘汰 |
| 12 | Reader-Grounded Two-Plane | Huatuo/Hulu，VinDr dev 640/model，8 findings×4 vote bins | Huatuo best early-minus-final clarity AUROC `-0.040`，image CI `[-.198,+.115]`；Virtual Reader 不优于 finding prior/unconstrained evidence model；状态中的方向 screen 也无统一早层优势 | 最终层 polarity/temperature 或 finding prior；未出现独立 clarity-erasure | A- | 核心机制失败；不运行 RCCP，不用 Hulu 事后救 Huatuo |
| 13 | Evidence Recoverability / ETD | Huatuo、Hulu；dev 640/model，独立 confirmation 1,920/model | raw oracle 给出 FP 0% 可恢复、FN 100%，但因两模型最早两层对所有 claim 都偏 Yes。校准后 Huatuo FP/FN `84.5/78.3%`，matched-null `90.6/88.4%`；Hulu `74.3/94.0%`，null `94.6/97.3%`，四格均不优于 null | 层级 base-rate/always-Yes 假象；same-finding same-truth 随机轨迹更“可恢复” | A | sampled answer-position recoverability 被明确证伪；不启动 ETD |
| 14 | 中间视觉 token 的 polarity readout | Huatuo，VinDr dev 640 + image-disjoint confirmation 1,920；明确 claims 960 | `visual_mean:21` macro AUROC `0.7433`；与 final margin 等权融合 BAcc `0.6875→0.7167`，`+2.92pp [+.59,+5.19]`。但独立 evidence-admission gate 中 dev 选出的 `claim:7` 相对 `claim:28` macro AUROC 为 `-0.1091`，CI 明确为负；融合也未达到预注册 `+3pp`，ROI AUC `0.5449 [.4615,.6282]` | 不同 readout/metric 可以产生提示性的互补，但没有任何中间层通过统一 admission+locality gate；不能把 `+2.92pp` 升格为机制支柱 | A-（测量）/NO-GO（机制） | 降级为提示性二级结果；不再作为新方法的正立论基础 |
| 15 | 真实前后片方向 | Huatuo，silver discovery 58 claims/29 patients；冻结复现 46 claims/26 patients | final target change `+.233 [+.024,+.466]`，off-claim `+.269`，净优势 `-.037 [-.267,+.183]`；layer-24 复现 premium `+.615 [-.192,+1.305]` | 全局时间/病情/采集 common mode，而非 claim-specific change | B | 纵向方向 NO-GO；数据也缺 location→progression 真值 |
| 16 | Clinical Selectivity / signed response | 3 模型，32 triplets/96 inputs，report-derived grade-C labels | CSG：Huatuo `-.346 [-.604,-.079]`，Hulu `+.402 [-.133,+.904]`，LLaVA `-.081 [-.103,-.059]`；selectivity mixer 不稳定，signed aligner 三模型三态 Acc `0/.13/0` | image sensitivity 不等于临床方向；简单 calibration/监督 mixer 不弱 | C | 保留“方向性必须验证”的审计原则；方法淘汰 |
| 17 | Fixed-K claim transport / ECCE | SLAKE 第三 cohort 52 images、3 模型；MIMIC natural report dev | pooled precision `+1.54pp [-1.28,+4.46]`；MIMIC supported recall `26.2%→19.0%`，22 个 refuted baseline claims 主要变成 `+27` unverified | visual support 不等于任务 reportability；verified precision 可通过逃到 unknown 人为升高 | B- | 淘汰 raw ontology rerank；fixed-K 仅保留为防作弊评测控制 |
| 18 | RAG/响应路由/stacking | CXR-VisHal blind test 734；Knowledge-MIMIC nested OOF | source-trained router BAcc `81.87→84.45%`，CI `+1.62~+3.53pp`；Knowledge stack `81.35→82.79%`，CI `+.19~+2.70pp`。曾有 paired code `86.50→91.02%`，但 exact-question placebo 为 `91.34%`，反而高于原配对 `90.99%` | 普通 question prior、低相关专家 stacking、阈值保守化；不是患者特异 evidence code | A-（提分）/A（机制否定） | 保留为竞赛 baseline；CMP/pre-treatment router 与患者码故事淘汰 |
| 19 | Evidence source ownership | Huatuo/Hulu，64 VinDr images/model，4 findings，受控 source prompts | CURRENT absent→present 为 `+1.367/+1.757`；OTHER patient 为 `+1.468/+1.866`，OTHER/CURRENT `1.074/1.062`；4/4 findings 同向 | 受控文本 priming；尚无自然报告、层级 erasure 或实际错误率 | C+ | 行为 source blindness 成立，但与主目标距离大；冻结旁支 |
| 20 | 当前正式 baseline 矩阵 | 4 主模型、7 数据集、training-free/RAG/trained tracks，共 336 cells | `32 completed / 3 running-partial / 1 generated-unscored / 151 pending / 149 N/A` | N/A 多来自无法保持官方语义或质量门失败；不能把历史 smoke 填入正式表 | D（进度） | 继续运行；它是论文比较底座，不是新方法证据 |

## 2. 不应被计为“已得到结果”的分支

- CECD、Specificity Ratchet、PCEM、ASCC 等大量文档主要是预注册、碰撞审计或人类
  construct gate；缺少医生回传时，不能因为代码/测试齐全就算临床结果。
- 已发现旧 LLaVA 报告缓存有单词 `The` 或正常模板坍缩；这些只能说明管线失败，不能
  评价方法效果。
- Visual-MIMIC 490 样本已从 OE 短答案更正为报告生成。Huatuo greedy 的
  RaTEScore `0.4891`、RadGraph-simple `0.1498`、CheXbert example-F1 `0.2757`
  是评测修正后的 baseline，不是 hallucination mitigation 增益。

## 3. 能统一解释多数结果的隐藏变量

### 3.1 不是“视觉信息多少”，而是干预引起的 common-mode 工作点移动

将模型对图像 `i`、claim `c`、干预 `a` 的 margin 写成

\[
m_{ica}=s_{ic}+b_c+g_{ia}+h_{ica}+\epsilon_{ica}.
\]

- `s_ic`：真正的 patient×claim 临床信号；
- `b_c`：finding/措辞/答案基率；
- `g_ia`：该图像在干预下对大量 claims 共同产生的工作点移动；
- `h_ica`：我们真正需要的 claim-specific 干预响应。

过去很多方法观察的是

\[
\Delta_a m_{ic}=g_{ia}+h_{ica},
\]

却把整个变化都解释成临床证据。多个独立实验都表明 `g` 往往比 `h` 大：

1. LET 和低视觉 head 主要统一推高阳性，Recall 上升但 FP/Specificity 变坏；
2. temporal target response `+.233`，off-claim common mode 反而 `+.269`；
3. calibrated recoverability 很高，但 same-finding/same-truth 随机轨迹更高；
4. patient-aligned RAG code 在 exact-question placebo 下完全消失；
5. style drift 不如原始 margin，FedDG/DOC 能改变表示却不能 rescue；
6. OTHER patient 的 disease polarity 与 CURRENT patient 一样能移动答案。

因此当前最稳健的跨实验结论是：

> **医学 VLM 很容易对干预作出响应，但“响应”首先是 claim-common operating-point
> shift；只有超过 off-claim、same-truth 和 exact-question controls 的残差，才有资格
> 称为可用临床证据。**

这也解释了为什么漂亮方法反复失败：它们优化了变化幅度、稳定性、层差、病灶响应或
检索响应，却没有识别 `h` 是否存在。

### 3.2 当前没有通过门的内部正证据

`visual_mean:21` 的 `0.7433` AUROC 和等权融合 `+2.9pp` 只能说明一个提示性事实：不同
readout 可能存在误差差异。它不能被写成“中间层优于最终层”：正式 evidence-admission
选择得到 `claim:7`，其确认 AUROC 比同族最终 `claim:28` 低 `0.1091`；等权融合也以
`0.0008` 未达到预注册 `+3pp`，并且 ROI 因果门失败。

因此当前不存在一个已经确认的、可供 mitigation 提取的正向 `h`。后续必须先做
**patient×claim 特异响应存在性检验**；若所有干预在去掉 image-common 和 claim-common
响应后均不含额外标签信息，就没有理由继续构造解码算法。

## 4. 最多两个可继续的问题

### 候选 1（优先）：Clinical Common-Mode Rejection（CCMR）

令 `v_ic` 为共享小型视觉读出器从中间视觉 token 得到的 claim score，`d_ic` 为 VLM
最终 claim margin。定义视觉—语言差值

\[
r_{ic}=v_{ic}-d_{ic}.
\]

在同一张图像的固定 ontology 内，用 robust location 去掉 common mode：

\[
\hat g_i=\operatorname{median}_{k\in\mathcal C} r_{ik},\qquad
e_{ic}=r_{ic}-\hat g_i,
\]

再作最小改动

\[
\tilde d_{ic}=d_{ic}+\lambda e_{ic}.
\]

OE/report 中固定原草稿的正 claim 数 `K`，只允许一进一出；因此不能靠全阳性、全阴性、
缩短或 hedge 获益。

它与旧 DG 的关系不是“找到训练中心”，而是：**域/风格可以任意移动 common mode，
只要它对多数 ontology claims 是共享的，within-image residual 对该移动不变。**

非平凡但简洁的数学结论：

1. **加性域不变性。** 对任意 image/domain shift `a_i`，若
   `v'_ic=v_ic+a_i` 或 `d'_ic=d_ic+a_i`，median-centering 后 `e_ic` 完全不变；
2. **稳健恢复界。** 若少于一半 ontology claims 含 claim-specific corruption，且其余
   noise 的绝对值不超过 `eps`，则 median 对 common mode 的估计误差不超过 `eps`；
   当真实 residual 大于 `2eps` 时，其符号不会被 common mode 翻转；
3. **无全局极性漂移。** 对中心化更新使用 sum-zero/median-zero 投影，统一的 Yes 或 No
   增益被精确消去；固定 `K` 进一步保证内容覆盖不变。

现有 confirmation cache 的 1,920 claims 来自 1,401 张图像，每张只覆盖 1–5 个已评分
findings，并非完整 ontology，因此不能假装已经具备稳健的 within-image median。第一步
只能在重复图像子集做 CPU 可行性检查；正式致死门需另取冻结的小型 image-disjoint
cohort，对每张图一次性评分全部 8 个 findings，再比较 final、visual-only、naive 50/50
fusion、finding calibration、CCMR 和 random residual。VinDr 当前也没有可核验 patient ID，
只能声明 image-disjoint。只有 CCMR 比 naive fusion 再提高至少 1pp、CI 排除 0，且 FP/FN
均不恶化，才进入 OE。

主要风险：疾病可能并不稀疏，或中间视觉 readout 的偏差不是 common-mode；若 Hulu 和
Huatuo cached confirmation 上都无增量，应立即淘汰，不能调 ontology 救结果。

### 候选 2（备选测量，不先包装成算法）：Clinical Difference-in-Differences

对任意干预 `a`（style、mask、RAG、decoder）先计算 target claim 变化，再减同图像的
off-claim 变化：

\[
S_{ica}=\Delta_a m_{ic}-
\operatorname{median}_{k\in\mathcal C_{-c}}\Delta_a m_{ik}.
\]

只有 `S` 在 source/dev 上具有正确方向、跨域复现并超过 matched random intervention，
才允许使用该干预。这个量把过去的 temporal off-claim subtraction、recoverability
matched-null 和 exact-question placebo 统一成一个可执行 admission law。

它更适合作为所有 mitigation 的资格检查，而不是论文方法；若只得到“普通 response
减平均 response”，会与已有 counterfactual/interaction 工作碰撞。只有它能稳定改变
现有方法排名、预测跨域 FP/FN 改善，才升格为贡献。

## 5. 总判决

1. 当前没有一个已验证、通用降低医学 VLM 幻觉的方法；ICLR-ready 状态为 **false**。
2. 已证伪的共同错误是把 domain/style/layer/mask/RAG 引起的“输出变化”直接当作
   “正确临床信号”。
3. 最值得利用的资产不是某个失败方法，而是 `(a)` formal VinDr reader substrate，
   `(b)` 两模型 hidden cache，`(c)` 中间视觉 token 的独立 polarity 信号，和
   `(d)` 一组已经证明必要的 common-mode controls。
4. 后续主线应先验证 CCMR 的 CPU 致死门；它失败就回到 baseline+负结果，不再用新的
   fancy 名称重复风格选择、层融合或干预 stacking。

## 6. 关键证据路径

- `docs/WEEKLY_REPORT_LET_20260727.md`
- `docs/STYLE_HYPOTHESIS_AUDIT.md`
- `docs/EVIDENCE_DG_PILOT.md`
- `docs/DOMAIN_ORBIT_FATAL_AUDIT_20260803.md`
- `docs/EVIDENCE_RECOVERABILITY_FIRST_SCREEN_RESULT_20260803.md`
- `docs/C3_GUARD_PHASE1_FATAL_RESULT_20260806.md`
- `docs/RIGHT_REGION_WRONG_DIRECTION_FATAL_GATE_20260806.md`
- `docs/SITH_MEDICAL_PREFERENCE_PROBE_STATUS_20260806.md`
- `docs/COMPETITION_TO_ICLR_RESEARCH_PROGRAM_20260810.md`
- `corrected_runs/reader_grounded_controlled_source_injection_v1/tristate_margin_v2/`
- `corrected_runs/paper_baselines_v1/full_matrix_v1/coverage_audit.json`

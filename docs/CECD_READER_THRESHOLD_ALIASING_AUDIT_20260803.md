# CECD reader-threshold aliasing：outcome-blind 审计与最小敏感性合同

**日期：** 2026-08-03  
**审计边界：** 未读取任何正式 CECD `dev_fit` / `confirmation_locked` 模型结果、sealed outcome 或 hidden state；未启动 GPU；未改变 CECD 主门槛、finding、sample quota 或确认集规则。

## 结论先行

离散 reader threshold **不能凭空产生**每个 image-claim 内部的 render × wording 二维交互

\[
I_{rp}=m_{rp}-\bar m_{r\cdot}-\bar m_{\cdot p}+\bar m_{\cdot\cdot},
\]

因为该量在任何 reader label 进入前已由完整 product orbit 的模型分数唯一决定。但它可以把一个普通、无临床方向的非加性交互**包装成 reader-grounded CECD residual**，因为当前正式结论还依赖三处 reader-derived 操作：

1. 用 0/3 与 3/3 定义 clear-case polarity error；
2. 用四个 vote-bin clean-score 均值的相邻差中位数 \(\beta\) 把交互换算为 reader equivalents；
3. 只保留 vote count，忽略 1/3、2/3 中具体是哪位 reader 投正票。

因此应把结论拆成两层：

- **threshold-free product instability：** 原始 logit 空间中是否存在非加性；reader threshold 无法伪造。
- **reader-grounded clinical product harm：** 非加性是否稳定增加与 reader support 的 proper loss；这必须通过连续 panel support、leave-one-reader-out、cumulative threshold sweep 与 exact reader-pattern absorption 才能成立。

当前旧 `reader_threshold_aliasing_control_v1` 不能单独回答第二层。它研究的是 clean condition 下“模型是否像某位 reader”，不是“CECD product interaction 是否在替代 reader 定义下仍造成临床误差”。它只能作为一个子控制复用。

## 一、现有 substrate 与合同审计

### 1. Reader manifest 本身合格，但 truth scope 很窄

`reader_vote_manifest_v2.jsonl` 保留了固定 panel `R8/R9/R10` 的 named binary votes、vote count 和 \(q=v/3\)。`summary_v2.json` 显示三位 reader 各覆盖同一 5,501 张图像，三阶段在采样前按 whole image 做 20/20/60 hash split；正式 CECD 的 dev/confirmation quotas 分别是每 finding × vote-bin 20/60。这个设计排除了同一图像通过另一 finding 跨 split 泄漏。

但 \(q\in\{0,1/3,2/3,1\}\) 是**固定三人 panel 的经验支持比例**，不是潜在疾病概率，也不是来自 reader population 的连续真值。没有 patient mapping 时，统计结论只能写成 radiograph-level；不能写 patient-level。只有三位固定 reader，也不能用“bootstrap readers”伪造 reader-population inference。

### 2. Clinical admission 不解决 reader aliasing

临床 admission 的作用是验证每个 render family 是否保持 `supported/refuted/undetermined` 状态、可见性与临床可交换性；语言 admission 验证 proposition、speech act、certainty demand 与 answer space。它是 product orbit 的必要前提。

但 admission pack 的 reviewer sheet 隐藏 reader votes；sealed mapping 只保留 `positive_votes`，不保留具体 reader pattern。其 60 个 clinical claims 来自平衡 vote strata，family-level change-rate 门槛不估计 reader-specific threshold，也不把 transformed view 重新标成独立的 R8/R9/R10 vote。因此 admission 不能替代本文的 alias sensitivity。

### 3. 正式 CECD 只闭合 4 findings，旧 alias control 闭合 8 findings

正式 runner 冻结：

```text
aortic_enlargement
cardiomegaly
pleural_effusion
pulmonary_fibrosis
```

旧 alias preflight 冻结了 8 findings，并要求每个 dev/confirmation vote-bin 恰好 20/60 条。正式 three-stage 不会产生另外四类的 factorial rows，所以旧 preflight 即使未来有 CECD GO，也无法直接由正式输入满足。这不是 power 问题，而是 closure mismatch。

本轮没有篡改旧候选 B 的冻结问题；另建了与正式 CECD 完全一致的 4-finding sensitivity contract：

`configs/cecd_reader_threshold_alias_sensitivity_v1.json`。

### 4. 发现并修复了 reader identity 的 silent permutation 风险

原 manifest 的 reader records 按字符串排序为 `R10,R8,R9`。factorial packer 将其压成无名字的 `individual_reader_votes: [..]`；主 CECD analyzer 随后完全丢弃该字段。旧 alias loader 却允许裸 length-three list，并把位置解释成 `R8,R9,R10`。若直接接线，会静默交换 reader identity。

本轮已将 alias loader 改为 fail-closed：只接受 `{R8,R9,R10}` mapping 或带 `rad_id` 的三条 named records，拒绝裸 positional list。正式 sensitivity 必须通过 `(image_id,finding)` 回连 hash-bound manifest，且同时校验 vote count。该改动不触碰任何统计阈值。

## 二、离散 reader threshold 如何伪造“临床 product residual”

### A. Endpoint aliasing

当前 `polarity_error` 只在 0/3 与 3/3 定义，中间两档全部不进入 AUROC。若 product interaction 主要在低 margin 样本上移动，而 unanimous cells 恰好因 case mix、reader operating point 或疾病谱更容易形成特定 margin 分布，交互可以预测这个二元 endpoint，却不一定连续恶化 reader support。

这不会制造 \(I\)，但会制造“\(I\) 有害”的方向性证据。

### B. Scale aliasing

当前 reader-equivalent scale 是三段 bin-mean difference 的中位数：

\[
\beta=\operatorname{median}(\mu_1-\mu_0,\mu_2-\mu_1,\mu_3-\mu_2).
\]

只要一段非常小、四档关系非线性或非单调，\(|I|/\beta\) 就可能被放大并跨过 0.25 RE MCID。现有 gate 只要求每个 finding 的 bootstrap median slope 为正，没有要求三段差分别单调为正。故 `RMS >= 0.25 RE` 可能是 calibration geometry，而不是 interaction magnitude 的稳定临床尺度。

### C. Reader-identity aliasing

同为 1/3 的 `100/010/001` 和同为 2/3 的 `110/101/011` 并不等价：固定 panel 中 reader 的敏感度、特异度或亚型偏好可能不同。只用 vote count 会把系统性 reader operating point 误写成 image ambiguity。旧 exact-pattern analyzer能检测这个问题，但在 unanimous `000/111` 上 pattern feature 恒为零，所以它结构上不能解释或排除 clear-case CECD harm。

### D. Selection 与 bootstrap 边界

按 finding × vote-bin 平衡采样是合理的 mechanism design，但它改变了自然 prevalence。AUROC 对 prevalence 不敏感，Brier、NLL、harm alignment 与 pooled average 则受目标分布和 case mix 影响。必须报告平衡设计 estimand，不能把它外推为 VinDr population risk。

现有 whole-image bootstrap 单模型内是正确的最小单位；但两模型使用同一图像时应在同一 draw 中同步 resample，才能对模型 conjunction 和跨模型差异保持配对。三位 reader 不应作为 bootstrap cluster。若未来获得 patient mapping，应以 patient 替代 image 为主 cluster；在此之前只声称 radiograph-level。

## 三、最小、可证伪的 sensitivity analysis

所有 calibration、feature transform 和 pattern coefficient 只在 `dev_fit` 学习，`confirmation_locked` 只应用一次。任何 variant 都不修改原 CECD GO/NO-GO；它们决定的是 GO 后能否升级为 **reader-grounded product-specific mechanism claim**。

### S0：Threshold-free existence

直接报告 raw FP32 logit interaction RMS，并以 identity duplicate noise 归一化；不使用 reader votes。若这里只剩 reader-equivalent RMS 而 raw/identity-normalized interaction 不稳定，结论是 scale artifact，而不是临床 product defect。

### S1：Continuous panel support（核心）

令 \(q=(R8+R9+R10)/3\)。在 dev 上按 model × finding 固定拟合两种不调参 calibrator：fractional logistic 与 isotonic；在 confirmation 上计算

\[
\Delta_{soft}=L(q,g(m_{actual}))-L(q,g(m_{actual}-I)),
\]

其中 \(L\) 同时使用 fractional Bernoulli cross-entropy 与 Brier。`m_actual-I` 是保留两边 main effects、只移除 product interaction 的 additive counterfactual。\(\Delta_{soft}>0\) 表示 product interaction 连续增加 reader-support loss，不依赖 0/3 vs 3/3 cutoff。

注意：这里称 **continuous panel support**，不称 continuous truth；四个经验 support level 仍是离散观测。

### S2：Leave-one-reader-out（固定 panel 稳健性）

依次去掉 R8、R9、R10，令 \(q_{-j}\in\{0,1/2,1\}\)，每次仅在 dev 重拟合 calibrator，再一次性应用 confirmation。同步 whole-image bootstrap 三个 omission 版本。

它回答“结果是否由某一位 reader 的 operating point 驱动”，不回答“能否泛化到任意 radiologist”。合格标准：每个模型三个 LORO 点估计同为正，六个 model × omitted-reader cell 的 pooled CI 下界大于零；任何单 reader 删除后稳定反号，则 reader-grounded mechanism claim 失败。

### S3：Cumulative threshold sweep

冻结三个 ordinal cumulative targets：

```text
y1 = 1[votes >= 1]
y2 = 1[votes >= 2]
y3 = 1[votes >= 3]
```

各自在 dev 拟合 calibration/predictor，confirmation 不重拟合。每个模型三个 threshold 的 product-loss point direction 都必须为正，且至少两个 threshold 的 bootstrap CI 下界大于零；不得事后选择最好 threshold。这个分析不替代主 clear-case endpoint，只检验主方向是否仅存在于 unanimity discretization。

### S4：Exact reader-pattern absorption

在原 strongest behavioral baseline 上加入：

```text
vote_count
+ named R8/R9/R10 pattern × model × finding
```

冻结该 baseline 后，再加入 harmful/absolute CECD interaction。若 interaction increment 在两个模型的 paired image bootstrap CI 下界仍高于零，reader identity 未吸收 CECD；否则只保留“固定 panel 的 reader-pattern association”，停止 product-specific clinical claim。

旧 `analyze_reader_threshold_aliasing_v1.py` 的 design matrix、frozen-offset increment 与 named-pattern logic可复用，但其 endpoint、8-finding closure 和 clean-only problem不能原样复用。

## 四、Bootstrap 与统计执行合同

- **主 cluster：** whole `image_id`，同一图像的 findings、15 science cells 与两个模型必须一起进入同一 draw。
- **Confirmation CI：** 5,000 次 paired whole-image bootstrap；dev fit 保持冻结，估计对已冻结开发流程的 held-out uncertainty。
- **Fit uncertainty audit：** 另做 1,000 次 two-sample nested bootstrap；分别 resample dev images 与 confirmation images，每次只在 resampled dev 重拟合 calibrator，再应用 resampled confirmation。
- **Reader：** 不 bootstrap；三人是固定 panel，LORO 是 sensitivity，不是 reader-population CI。
- **Patient：** 当前 manifest 无 patient key，因此不声称 patient-cluster inference。若补到可靠映射，patient bootstrap supersedes image bootstrap；不能两者择优报告。
- **模型 conjunction：** Huatuo/Hulu 共享 image draw。两模型均须满足，而不是 pooled 后由一模型抵消另一模型。
- **Finding heterogeneity：** 四个 findings 为冻结 fixed effects，继续报告逐 finding 方向；不把四个 findings 当随机样本做 cluster bootstrap。

## 五、执行时序

### 正式 CE 前现在必须完成

1. 冻结本 sensitivity schema、calibrator family、threshold sweep、LORO 与 bootstrap unit；
2. 校验 4-finding closure、named-reader manifest join、source hashes 与 dev/confirmation image disjoint；
3. 拒绝 positional reader list；
4. 不读取 model scores，不创建结果目录，不启动 GPU。

### 仅在 locked CECD GO 后执行

复用已经生成的 dev/confirmation factorial rows 做 S0–S4，全部 CPU；不再运行模型。这样 CECD NO-GO 时不浪费分析，也避免 sensitivity 结果反向影响主门槛。

在 S0–S4 完成前，locked CECD GO 只能称 **primary behavioral GO**，不能写 reader-grounded product-specific mechanism，更不能据此授权 mitigation novelty。若 sensitivity 被 aliasing 吸收，应保留可信负结论：模型存在 generic product instability，但其临床方向依赖固定三人 panel 的离散化。

## 六、本轮落地与验证

- 新增 outcome-blind 正式合同：`configs/cecd_reader_threshold_alias_sensitivity_v1.json`。
- 新增合同测试：`tests/test_cecd_reader_threshold_alias_sensitivity_v1.py`。
- 修复 reader list silent permutation：`analyze_reader_threshold_aliasing_v1.py` 现在拒绝 unnamed positional votes。
- 更新旧 preflight 的 analyzer SHA/bytes；所有统计阈值保持不变。
- 定向回归：`13 passed`。

最终判定：**reader-threshold aliasing 是 CECD 临床解释的真实高风险替代解释，但不是 raw product interaction 的生成机制。最正确的时序是“CE 前冻结与接线审计，primary GO 后、任何机制/paper claim 前一次性执行”。**

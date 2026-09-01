# CECD 近邻碰撞控制可实施性审计

**日期：** 2026-08-03  
**模式：** outcome-blind；只审计论文、官方代码、当前协议与合成测试。未读取 human return、sealed model output、确认集指标或任何科学 outcome；未启动 GPU。  
**范围：** 仅限 CECD 的 controlled CE product orbit；不扩张到 OE、报告生成或 mitigation novelty；不修改既有冻结阈值。

## 结论

当前已经有两个真实而且进入最强行为门的控制：

1. **generic two-axis/full-grid instability**：slice score SD、axis entropy、axis/full-orbit probability dispersion、cell-to-orbit KL、orbit predictive entropy；
2. **behavioral synergy**：在均匀独立 render/prompt source 上计算的三态输出分布 MMI-PID-style synergy 及 local excess。

它们均可直接由现有 15-cell `Yes/No/Maybe` logits 在 CPU 上重算，不需要任何新生成。正式 dev-fit/confirmation predictor 也确实把 behavioral PID rung 放在 CECD interaction 之前，而不是只把字段写进日志。

但它们还不能支撑“CECD 超出所有 generic product instability / behavioral synergy”的 headline。现有 generic/PID 特征主要是 **reader-label-invariant 的幅度与分布摘要**，而候选特征是 reader-oriented signed interaction；二者并不构成匹配零假设。此外，候选项补齐了定义当前 cell error 的同一个 signed score，存在已单独审计的 algebraic target coupling。当前门最多是 behavioral contribution screen，不是独立机制识别。

[MedFocusLeak](https://aclanthology.org/2026.acl-long.1768/) 是危险近邻，但不是当前 CECD grid 的等价 baseline：它优化一个错误诊断 target、非诊断背景 perturbation 和 attention distraction，属于 **targeted adversarial construction**；CECD 的两个轴必须在输出出现前被临床医生独立承认为 support/proposition/speech-act preserving。最小正确决定是：**不把 MedFocusLeak 塞入 locked behavioral gate；只在 CECD-GO 后作为非等价 adversarial boundary stress test。**

## 1. 实施状态与优先级

| 优先级 | 碰撞/控制 | 当前等价覆盖 | 可用现有生成物重算 | 必须新增生成 | 判定 |
|---|---|---|---|---|---|
| P0 | generic marginal + two-axis/full-grid instability | **有，且进入 dev OOF 与 frozen confirmation baseline** | 是；完整 15-cell signed score、三态 logits、entropy、length、view 足够 | 否 | 已有控制真实有效，但不是 matched signed-product null |
| P0 | behavior-level synergy | **有，MMI-PID-style scalar + cell-local excess 已进入最强 rung** | 是 | 否 | 足以排除“仅仅是 label-free output synergy”的弱说法；不足以排除 reader-oriented 或 causal synergy |
| P0 | reader-label-destroying matched product null | **无** | 是；仅需现有 interaction matrices 与 reader votes | 否 | headline 前必须新增；不改变当前 0.25 RE/0.03 AUROC 阈值，只形成独立非授权控制 |
| P1 | full three-state generic product geometry | **部分**；现有 PID/dispersion 不等价于完整 shift-invariant logit surface | 是 | 否 | 应新增 CPU control，避免只在 Yes-minus-No 一维上定义“generic” |
| P1 | cell-identity / flexible full-grid sanity baseline | **无显式 cell/product categorical** | 是 | 否 | 用于证明当前 incremental-AUROC 是否只是补齐当前 cell score；若吸收增量，停止 predictive-residual 话术 |
| P2，仅 CECD-GO 后 | MedFocusLeak-like targeted adversarial boundary | **无等价 baseline** | 否 | 是；错误 target text、mask、CLIP/BLIP optimization、victim inference 全需新产物 | 不是 core baseline；用于划清 natural equivalence failure 与 adversarial vulnerability |
| P2，仅行为 GO 后 | representation-level PID / attention diversion | **无；代码也明确不声称已有** | 否 | 是；需要 hidden states/attention 或干预输出 | 仅机制归因需要，不能由 behavioral MMI 冒充 |

## 2. Generic grid/two-axis instability

### 2.1 已经实现的部分

当前 analyzer 在 `make_cell_table` 中生成：

- `render_main_harmful_re`、`prompt_main_harmful_re`；
- 两个 marginal RMS 与 `full_orbit_harmful_re`；
- visual/language slice score SD；
- visual/language entropy mean；
- visual/language/full-orbit probability dispersion；
- cell-to-orbit KL 与 orbit predictive entropy。

这些字段进入 `CLOSEST_WORK_FEATURES`，之后 MMI fields 进入 `BEHAVIORAL_PID_CONTROL_FEATURES`。`fit_dev_stage` 以后一组作为 baseline predictor，只有 candidate predictor 再加入 `interaction_harmful_re` 与 `interaction_abs_re`。`apply_confirmation_stage` 固定 dev 参数、在 image-disjoint confirmation apply-only。因此“generic grid 只是文档要求、没有进入 formal model”的风险已经排除。

相关源：`anchor/corrected_sgta/analyze_clinical_equivalence_composition_defect_v1.py` 的 `make_cell_table`、`CLOSEST_WORK_FEATURES`、`BEHAVIORAL_PID_CONTROL_FEATURES`、`fit_dev_stage` 与 `apply_confirmation_stage`。合成测试 `test_reader_grounded_interaction_explained_by_generic_stability_fails` 已证明：一个 RMS 很大的 interaction 若被 generic rung 解释，门会失败。

### 2.2 仍未闭合的碰撞

现有控制回答的是：

> 这个 cell 是否位于一个总体不稳定、熵高、分布离散的 grid 中？

headline 需要回答的是更强的问题：

> 在保持同一 interaction subspace、Frobenius norm、两个 marginals 与三态分布几何时，为什么 interaction 恰好沿 reader-grounded clinical-harm direction 对齐？

当前 generic features 多数不携带这一对齐。反之，candidate 使用 reader truth sign 定向。故其增量可能只是“generic baseline 被故意拿掉的 signed direction”，不能写成 product-specific information 已经超出 generic grid。

更严重但属于同一 collision family 的事实是：对清晰病例，当前 target 为 `1[s_i M_rp < 0]`；baseline 含 `h_i mu`、`h_i a_r`、`h_i b_p`，candidate 再加入 `h_i J_rp`，从而精确重建 `h_i M_rp`。这不是 train/test leakage，但会使 predictive AUROC 的增量带有代数必然性。详细推导见 `docs/CECD_PRODUCT_ESTIMAND_AUDIT_20260803.md`。

### 2.3 最小 CPU 实现

不动原阈值，增加三个 outcome-blind、non-authorizing controls：

1. **Orbit-sign matched null**  
   对每个 image-claim orbit 以随机 `+/-1` 翻转完整 centered interaction `J_i`，保持 row/column sums 为零、Frobenius norm、grand mean、marginals、cell count 与 image clustering；破坏 reader orientation。用 image-cluster randomization 比较实际 `Delta loss` 或 harmful alignment 与 null envelope。

2. **Interaction-subspace rotation null**  
   在维数 `(R-1)(P-1)` 的 centered interaction subspace 内做正交旋转，保持 interaction energy，销毁特定 cell/reader alignment。它比只翻 sign 更强，但同样只需 NumPy/CPU。

3. **Flexible grid sanity baseline**  
   在 dev-only 冻结 `render_id`、`prompt_id`、product-cell identity，以及 shift-invariant 完整 grid coordinates；confirmation 只 apply。该 baseline 若吸收 CECD delta-AUROC，结论应改为“product interaction 对当前决策的描述性 attribution”，而不是“独立预测 residual”。由于 CECD 是 raw grid 的确定函数，不能把击败一个不知道完整 grid 的低容量 baseline 写成信息论独有性。

建议把主要可解释统计从“增量预测同一 score 定义的 error”改为 additive counterfactual loss：

\[
M^A = M-J,\qquad
\Delta L=\ell(s,M)-\ell(s,M^A),
\]

并与上述 matched null 比较。现有 `0.25 RE` 和 `0.03 AUROC` 继续保留为原 screen；新的 `Delta L` 在独立 power/MCID 冻结前不得授权 headline。

## 3. Behavioral synergy

### 3.1 已经实现的部分

当前 `behavioral_pid_mmi` 把 render 与 prompt 当作均匀独立 source，把模型的三态概率当作 stochastic target：

\[
S_{MMI}=I(R,P;Y)-\max\{I(R;Y),I(P;Y)\}.
\]

它同时返回 orbit-level synergy 与 cell-local excess；三者都进入 strongest behavior-only baseline。实现的 guardrail 是正确的：这是 **output-distribution MMI-PID-style control**，不是 hidden-state PID、causal synergy 或 representation mechanism。

### 3.2 最小加强

现有 logits 已足够增加一个完整三态的 label-free product geometry，无需模型生成：

- 对每个 cell 的三态 log-probability 去除共同平移，得到二维 simplex coordinate；
- 分别对两个 coordinate 做 two-way centering；
- 报告 per-cell residual norm、orbit Frobenius norm、与 additive multinomial/log-linear surface 的 KL deviance；
- 把这些 label-free features 放在 reader-oriented CECD 之前，并用同一 dev-fit/confirmation apply-only 协议。

这样可排除“MMI scalar 过粗，CECD 只是捕获了完整三态 joint curvature”。但它仍不能排除 causal/representation synergy；后者只能在 behavioral GO 后用 hidden-state PID、joint noising/denoising 或 attention/activation intervention 检验。

### 3.3 Fatal collision 判据

以下任一成立，停止“超出 generic/behavioral synergy 的独立机制”表述：

1. reader-oriented product statistic未超过 orbit-sign/interaction-rotation matched null；
2. 完整三态 additive-vs-product geometry 或 flexible full-grid baseline 吸收所谓增量；
3. 正面结果只存在于 target-coupled current-cell error AUROC，而 additive counterfactual `Delta L` 不显示 product 增加 reader-grounded loss；
4. representation-level或 causal synergy 完全解释 joint-cell specificity，CECD intervention 不再有独立选择性。

前三项可以用当前生成物在 CPU 上裁决；第四项需要后续新 hidden-state/causal 生成。

## 4. MedFocusLeak-like coordinated attack

### 4.1 为什么不是当前 grid 的等价 baseline

MedFocusLeak 的科学对象是 targeted black-box adversarial transfer：[ACL 2026 论文](https://aclanthology.org/2026.acl-long.1768/)先生成错误但看似合理的诊断 target，再把 perturbation 限制到非诊断背景，并用 attention-distraction 把视觉焦点从病灶移走。其 image-only/text-only/joint ablation 证明定向攻击的多模态协同，而不是证明两个**独立临床等价**操作自然组合后失效。

因此二者的识别前提相反：

| CECD | MedFocusLeak |
|---|---|
| 两轴在任何 model output 前由独立临床 reviewer 判为 support/proposition/speech-act preserving | 优化目标就是诱导一个错误诊断 |
| 自然 nuisance product orbit | 定向 adversarial optimization |
| reader-vote clinical loss | attack success + imperceptibility/transferability |
| 无 attacker/target diagnosis | 显式 malicious target 与 surrogate ensemble |

如果 CECD render/prompt 需要被优化到错误 target 才出现 defect，它就不再是 CECD，而退化为已知 adversarial vulnerability。

### 4.2 官方代码可实施性

官方项目页链接的[代码仓库](https://github.com/AkashGhosh/When-Background-Matters-Breaking-Medical-Vision-Language-Models-by-Transferable-Attack)已发布；本轮冻结只读 commit 为 `32618c6dcf486623654838447d017ea1e872c58b`（2026-04-25）。代码不是可直接接入当前 common protocol 的一条命令：

- 五阶段流水线包含 GPT-4o-mini adversarial text、white-canvas target、CLIP feature attack、background mask attack、BLIP/CLIP attention-shift；
- 需要 OpenAI API、MedSAM masks、多个 CLIP/BLIP 权重和约 20–40GB VRAM；论文报告 300-step optimization，attention stage README 默认 500 steps；
- 阶段之间依赖手工复制目录，尚无 Huatuo/Hulu victim adapter、record-key 对齐、原子 shard、source/input hash binding 或 image-cluster paired evaluator；
- source/README 在“最终 victim 是否接收 adversarial prompt，还是 text 仅用于构造 adversarial image target”上不够清晰；paper formulation 主要查询 `(I_adv, x)`，而 threat appendix 又允许 image and/or prompt；
- README 的 Step 3/4 handoff 与代码对象存在需要 canary 澄清之处；旧 `generate_adversarial_samples.py` 还有 `torch.utlis` 拼写错误，`generate_adversarial_samples2.py` 才修正；`Adv_Text.py` 默认代码会写入占位 API key；
- 仓库根目录未见覆盖全项目的 LICENSE；内嵌 Modified M-Attack 子目录有自己的 LICENSE，不能自动外推到全部新代码。

所以“照论文名增加一个 MedFocusLeak arm”目前不具备 faithful-baseline 资格。必须先做 paper↔code conformance canary，并把 target-victim 输入语义锁死；否则只能标为 inspired surrogate，不能用于 novelty closure。

### 4.3 哪些必须新增生成

若 CECD 已经通过 locked behavioral GO，最小 adversarial boundary pilot 才需要：

1. 选一小批与 CECD confirmation 完全 image-disjoint 的 VinDr clear cases；
2. 生成并人工核对错误 target diagnosis；
3. 生成 MedSAM/等价 foreground mask；
4. 运行 image-only、text-representation-only、joint、joint-without-attention-shift 四个官方语义 arm；
5. 对 Huatuo/Hulu 用与 CECD 相同 one-token reader-grounded readout重新推理；
6. 记录 perturbation budget、SSIM/MedCLIP、foreground preservation、attention shift 与 reader-grounded error，不采用 MTR/MAS 或 LLM judge 定义真值。

这些都不能从已有 CECD logits 倒推出。保守估算至少是“每例数百步 × 多 surrogate”的新 GPU workload，且需要新的 mask/target/provenance 数据，不应与当前 locked confirmation 抢 GPU 或污染主 orbit。

### 4.4 Fatal collision 判据

MedFocusLeak 对 CECD 真正致命的不是“joint attack 比两个单模态 attack 强”，而是以下任一结果：

1. clinical admission 显示 CECD 的 render 或 wording 事实上改变了视觉支持、命题、certainty demand、speech act 或 answer grammar；此时所谓 equivalence product 不成立；
2. CECD 的 joint residual 只有在错误 target、background perturbation 或 attention distraction 存在时出现；自然 admitted orbit 不成立；
3. 一个与临床等价性无关的 generic/adversarial joint-sensitivity score，在 matched reader-grounded分析中完全吸收 CECD residual；
4. CECD 与 adversarial stress 的效应大小、cell specificity、跨模型 transfer pattern 无法区分，且无 admission/causal evidence证明二者边界。

反之，仅仅证明 adversarial joint attack 很强 **不会**杀死 CECD；它只说明另一种更强、不同 threat model 的 product vulnerability 已被研究。

## 5. 最小执行顺序

1. **现在、零 GPU：** 保留当前 generic + MMI strongest rung；在独立版本中实现 orbit-sign 与 interaction-subspace matched null、三态 product geometry、cell/full-grid sanity baseline及合成 falsification。不要改原阈值，也不要读取 confirmation outcome后再定新阈值。
2. **修复识别后：** 用 additive counterfactual `Delta E/Delta L` 表达 product attributable clinical loss；先做 power/MCID，再决定它是否可授权 headline。
3. **仅 locked CECD-GO 后：** 才启动 hidden-state PID/causal synergy；若简单 behavioral controls 已吸收，不做这一步。
4. **仅 reviewer boundary 需要时：** 做小规模 MedFocusLeak conformance canary 与 adversarial boundary pilot。它不是 core CECD baseline，不进入临床 admission gate，也不与自然 orbit 混合估计。

## 最终 KEEP / KILL

- **KEEP：** current generic grid rung 与 behavioral MMI rung；它们是真实实现，不需要重新生成。
- **ADD NOW（CPU）：** matched signed-product null、完整三态 product geometry、flexible full-grid sanity、additive counterfactual loss。
- **DEFER：** representation PID、attention diversion、MedFocusLeak faithful pilot；都要求新生成，只在 CECD behavioral GO 后有价值。
- **KILL：** 把现有 MMI 写成 causal synergy closure；把 MedFocusLeak 的 image/text ablation写成当前 clinician-equivalent product 的等价 baseline；把 target-coupled delta-AUROC单独包装为“独立预测 clinical error”。

这一路径不扩张论文，反而把剩余新颖性压到唯一可辩护的对象：**在自然、独立临床承认的等价 product orbit 上，reader-oriented product-attributable loss 是否超过保持同一 generic interaction energy 的匹配零假设。**

# Endogenous Observation Is Not Evidence：从 crop 假阳性到 agent 的“证据回声”

> 高层机制升级与严格碰撞，2026-08-12。本文只做文献、数学与实验设计；不占用 GPU，不改动 baseline。**当前状态：CLOSED / NO-GO。** 同像素 provenance gate 已失败，按预注册逻辑关闭 Endogenous Evidence Recycling 与 Policy-Cut Update，不运行 Replay。

## 0. 被检验的一句话假设（现已关闭）

> **Agent 自己选择“去看哪里”的动作是控制量，不是新证据；只有动作返回的内容才可能是新证据。**

我们曾假设，VLM/agent 把 `思考 → 选择 crop/检索/工具 → 看结果` 整段轨迹重新喂回模型时，会把“我选择了这个动作”也当作支持原判断的线索，从而把自己的怀疑循环计数，形成 self-confirming hallucination。**后续门控没有观察到所要求的动作归属特异性，因此这句话只能保留为被否定的研究假设，不能写成项目发现。**

这个问题比“crop prompt 会改变答案”更高一层：它问的是所有主动感知与工具调用系统是否缺少一个基本的因果类型区分：

```text
外部观察：可能携带关于世界的新信息
自己采取的动作：由既有历史生成，条件于历史后不应再次提供世界信息
动作返回的内容：可能提供条件于动作的新信息
```

该假设暂名 **Endogenous Evidence Recycling（内生证据循环）**。预注册已规定：若只出现通用文字 framing / 判定阈值变化，则不得使用这个名字。实际结果正落入这一失败情形。

## 1. Provenance gate 的最终结果

实验使用完全相同的 crop 像素，只改变 provenance 文字；Huatuo 上各包含 62 个阴性和 62 个清晰阳性 image–claim。核心结果来自 [`analysis.json`](../../corrected_runs/daylong_idea_search_v1/observation_policy_huatuo_v1/analysis.json)：

| 预注册检查 | 结果 | 判定 |
|---|---:|---|
| crop 特异的 provenance 交互 `Γ` | `−0.117`，95% CI `[−0.188, −0.042]` | 方向相反，FAIL |
| crop 阴性 FP 下降 | `61.3pp` | 单看似乎很好 |
| crop 阳性 recall 下降 | `59.7pp` | 几乎等量损伤，属于整体变保守 |
| full-image 阴性 FP 下降 | `4.84pp` | 超过允许的 `3pp`，说明不是 crop 特异作用 |
| 去除/模糊外部 context 后阴性 FP 增加 | `62.9pp` | 最大效应来自 context/render 改变 |
| 纯 scale 相对 native 的阴性 FP 增加 | `3.2pp`，CI 跨 0 | 不支持“单纯放大尺度”解释 |

### 1.1 为什么它不支持 self-action story

1. **核心交互符号错了。** 假设要求同像素条件下，“动作来源”在 crop 上产生额外正向增信；实际 `Γ=−0.117`，且置信区间完全低于 0，不是样本不足造成的模糊失败。
2. **所谓收益由统一保守化解释。** FP 降 `61.3pp` 的同时阳性 recall 降 `59.7pp`，更像一句话让模型普遍倾向回答阴性，而不是去除了错误的自我证据。
3. **效应并非局限于 action/crop。** 相同文字在完整图像上也使 FP 降 `4.84pp`，超过门槛允许的 `3pp`；因此它会移动整体 decision criterion。
4. **本门只测 provenance sensitivity，本来还没有真实 self-owned action。** 即使通过，也只能允许继续做 Replay；现在前置门失败，直接运行 Replay 会变成在无现象依据下寻找故事。
5. **已有分解指向更简单的视觉原因。** native 外部 context 被移除或模糊时 FP 增加 `62.9pp`，而纯 scale 只增加 `3.2pp`。crop 的主要危险更符合“丢失全局解剖上下文/引入 render artifact”，而不是“模型相信了自己的动作”。

因此，早先的 `8.1%→62.9%` crop FP 跳升不能再被用于支持“内生证据循环”。它只说明 crop 输入会破坏判断，不能说明破坏来自 self-action。

## 2. 理论背景：为什么公式仍正确，但不能救活该方向

### 2.1 变量及背景

- `Y`：真实临床状态，例如胸腔积液是否存在；
- `H`：动作前已经看到的历史，包括原图和先前回答；
- `A`：观察动作，例如选择一个 crop、发起某条检索或调用某个工具；
- `Z`：动作返回的内容，例如 crop 像素、检索文档或工具结果。

训练语料常来自临床工作流。医生因为怀疑 `Y` 才选择局部放大、补拍或进一步检查，因此训练分布中可能有

\[
Y\rightarrow A,\qquad (Y,A,H)\rightarrow Z.
\]

这里 `A` 的出现本身确实可能暗示疾病。医学 MNAR / informative observation 文献早已指出“是否被测量”可以携带医生怀疑的信息。

但部署时若动作由 agent 根据已有历史自己产生，

\[
A\leftarrow \pi(H),
\]

它是一个控制输入；条件于 `H` 后，agent 不应从自己刚做出的选择再次学习 `Y`：

\[
p(y\mid h,\operatorname{do}(a))=p(y\mid h).
\]

### 2.2 观测后验与干预后验

训练分布里的普通观测后验是

\[
p(y\mid h,a,z)
\propto
p(z\mid y,h,a)\,
p(a\mid y,h)\,
p(y\mid h).
\]

其中 `p(a|y,h)` 是“为什么会执行这个动作”的 selection likelihood。它在医生选择检查时可能合理；在 agent 自己执行动作时却不该进入更新。

正确的部署后验是

\[
p(y\mid h,z,\operatorname{do}(a))
\propto
p(z\mid y,h,a)p(y\mid h).
\]

由 Bayes 公式，若 positivity 成立，可得到

\[
\boxed{
p(y\mid h,z,\operatorname{do}(a))
\propto
\frac{p(y\mid h,a,z)\,p(y\mid h)}{p(y\mid h,a)}
}
\tag{1}
\]

背景解释：`p(y|h,a,z)` 同时含动作和结果；`p(y|h,a)` 只含动作；两者相除去掉 action likelihood，再乘回动作前信念 `p(y|h)`。

式 (1) 是标准 Bayes / inverse-selection transport identity，**不是新的数学定理**。其价值只可能来自：现有 agent 是否系统性违反它，以及一个 frozen model 的三个条件分布是否足以近似这个干预量。

## 3. 已关闭的候选方法：Policy-Cut Update

对同一个临床 claim，做三个冻结前向：

1. `base`：动作前，得到 `q(y|H)`；
2. `action-only`：保留 agent 的 tool call / crop 坐标，但环境结果尚未返回，得到 `q(y|H,A)`；
3. `action+result`：加入真实 crop / retrieval / tool output，得到 `q(y|H,A,Z)`。

曾据此定义 **Policy-Cut Update (PCU)**：

\[
\boxed{
q_{\mathrm{PCU}}(y)
\propto
q(y\mid H,A,Z)
\frac{q(y\mid H)}{q(y\mid H,A)}
}
\tag{2}
\]

对二元 claim，用 log-odds `m=log(q(Y=1)/q(Y=0))` 后只是一行：

\[
\boxed{m_{\mathrm{PCU}}=m_{HAZ}+m_H-m_{HA}.}
\tag{3}
\]

在真实且相容的概率分布下，这个式子能从后验中去掉动作选择项。然而，本项目没有确认模型存在需要去除的 self-action likelihood；冻结 VLM 在三种 prompt 下的分数也未证明来自同一联合分布。**所以 PCU 现在只是一个适用于假设成立时的标准因果恒等式实例，不再是待验证算法，更不能作为论文贡献。**

### 3.1 三个可证性质

**性质 1：action-only nuisance 不变性。** 若动作让 joint 与 action-only odds 同乘任意正因子 `s_A`，

\[
O_{HAZ}=O_H\,s_A\,\ell_Z,\qquad O_{HA}=O_H\,s_A,
\]

则 PCU 精确恢复

\[
O_{\mathrm{PCU}}=O_H\ell_Z.
\]

因此它保留 result likelihood `\ell_Z`，同时删除 action likelihood `s_A`。

**性质 2：无新结果，不更新。** 若 `Z` 对 claim 没有新增信息，使 `q(y|H,A,Z)=q(y|H,A)`，则式 (2) 给出

\[
q_{\mathrm{PCU}}(y)=q(y\mid H).
\]

这提供了一个不依赖标签的硬单元测试：把同一个 crop 结果重复返回，第二次以后不应继续增加信心。

**性质 3：乘法更新族中的唯一性。** 设校正后的 odds 只依赖动作前 odds `O_H` 与 action-matched innovation ratio `O_{HAZ}/O_{HA}`；同时要求当动作不产生偏置 (`O_{HA}=O_H`) 时恢复普通 joint odds。则唯一满足两项要求的形式是

\[
O^*=O_H\frac{O_{HAZ}}{O_{HA}}.
\]

这是简单的群不变性/odds 代数刻画，不足以单独作为 Oral 理论贡献，但它说明式 (3) 不是随意挑的 logit subtraction。

### 3.2 “证据回声”动力学

若没有新像素证据，但模型每次看到自己采取的同向动作都会加入 `b>0` 的 log-odds，普通更新为

\[
m_{t+1}=m_t+b\operatorname{sign}(m_t).
\]

于是

\[
|m_T|=|m_0|+Tb,
\]

即使世界没有给出任何新信息，置信度也会随步骤线性发散；若初始判断错误，错误会变得越来越确定。PCU 在 action-only 与 action+null-result 相等时每步增量为 0，因此 `m_T=m_0`。

这个动力学是 data incest / self-confirming belief 的单 agent 特例，也不是全新的概率论。但若在 multimodal tool agents 中跨动作出现，并能预测 hallucination 随 tool-loop 深度增长，它可以成为新的实证规律。

### 3.3 当前方法的两个致命接口风险

1. **三个 `q` 未必是相容概率。** 式 (1) 对真实条件分布精确，但冻结 VLM 在三种 prompt 下给出的分数未必来自同一个联合分布；若不满足概率相容性，式 (2) 只剩启发式 logit subtraction。
2. **`action-only` 必须是自然状态。** 空图、黑图或任意 null result 会重演本项目已经证伪的 OOD negative-control 问题。实验必须在真正的 agent trace 中，于 tool call 已发出、环境结果尚未 append 的边界读取状态；若模型接口做不到，再用语法原生的 `result unavailable` sentinel，并配置等长 no-op trace 控制。无法构造可信 action-only 状态时，PCU 方法应关闭。

所有分支都要用 stateless reconstruction、相同的最终 claim-verifier suffix 和相同答案 token；否则“动作效应”会与询问位置和生成历史混在一起。

## 4. 跨领域知识如何真正参与，而不是装饰

| 领域 | 已有核心认识 | 对本问题的约束 |
|---|---|---|
| Informative observation / MNAR | 是否进行检查可能反映医生怀疑 | **不能无条件删 action 信息**；外部医生动作可能是真信息，只能对 self-generated / randomized action 做 cut |
| Causal `do`-operator | 干预切断动作的自然生成机制 | 给出式 (1) 的目标 posterior，但公式本身标准 |
| Selective inference / post-ADC | 选择后重复使用数据会导致 anti-conservative inference，需除 selection likelihood 或条件化选择事件 | 说明 selected crop 的 raw score 不能直接当未选择 score；与本方法有强数学碰撞 |
| Active perception / POMDP filtering | 动作是已知 control，观测才更新 belief | 给出最简规范：agent 不应把自己的 control 当 sensor reading |
| Data incest / decentralized fusion | 信息沿反馈图回流会被误当独立证据，造成过度自信 | 给出“同一怀疑通过 action trace 回流”的结构类比与循环深度预测 |
| Choice-supportive bias | LLM 再看到“自己的”初始答案时，即使没有新信息也会提高信心 | 给出 self-ownership 的直接竞争机制与 owner-swap 对照 |
| Performative prediction | 部署策略会改变后续数据分布 | 支持“policy 不是被动 preprocessing”；但其经典对象是人群分布变化，不等同于单轨迹 evidence recycling |
| Agentic active reasoning | action selection 与 belief tracking 相互耦合，可能造成 information self-locking | T3/AREW 研究“没有获取/吸收足够信息”；本候选研究相反边界：“把自己的动作吸收成了额外信息” |

## 5. 2023–2026 直接碰撞

| 工作 | 已覆盖内容 | 与 PCU 的重合 | 剩余研究差异 |
|---|---|---|---|
| **PriDe**, ICLR 2024 | 用 label-free prior 估计除去 MCQ option-token bias | “总体后验 ÷ prior bias”代数非常接近 | 静态选项位置，不含 agent action、result 与序列回声 |
| **Active Statistical Inference**, 2024 | 对主动采样数据构造有效推断 | 同属 adaptive selection correction | 统计总体/置信区间，不研究 frozen agent 后验或 hallucination |
| **Selective Randomization Inference**, 2024 | 对 adaptive experiment 条件化 selection event | 选择后推断的原则相同 | 不分 tool action 与返回内容 |
| **Fine-Grained Visual Prompting**, NeurIPS 2023；**HALC**, ICML 2024 | crop / blur / local-global view 改变 VLM 输出并用于缓解 hallucination | 同一输入操作 | 不研究 self-owned action likelihood |
| **LLM Agents Can Be Choice-Supportive Biased Evaluators**, AAAI 2025 | agent 会支持自己先前的选择，且 perceived control 增强该偏置 | self-ownership 机制直接相邻 | 没有动作产生的新观测，也没有 do-correction |
| **Benchmarking and Mitigating MCQA Selection Bias of LVLMs**, EMNLP 2025 | contextual prior vector + logit correction | 实现层面与 logit prior subtraction 相邻 | 仍是静态 option bias |
| **Prompt-Induced Hallucination**, ACL 2026 | 文本预设能压过图像并定位相关 heads | 若只有 provenance wording 生效，将直接降级为该工作的医学复现 | 尚未研究“模型自己的动作”与动作结果的因果分解 |
| **Competing Biases underlie Overconfidence and Underconfidence in LLMs**, Nature Machine Intelligence 2026 | 显示自己的原答案使 change-of-mind odds 降低 71%，无新信息也增信；换成 other-agent attribution 后效应消失 | 对“self-owned action echo”是最强行为碰撞 | 研究答案可见性，不研究主动视觉/工具结果及 action likelihood cancellation |
| **T3**, ICLR 2026 Oral；**Information Self-Locking / AREW**, ICML 2026 | action selection 与 belief tracking 的反馈导致低信息锁定 | 同一 agent action-belief loop | 它们研究 RL 训练与 under-acquisition；本候选研究 inference-time over-counting |
| **MED: What Does Vision Tool-Use RL Really Learn?**, 2026 | crop-and-zoom RL 的增益主要来自 intrinsic learning；tool RL 主要减少 tool-induced harm | 强烈支持 tool 自身可引入 harm | 不区分 action-only 与 result evidence，不做干预 posterior |
| **BCEA**, 2026 | adaptive crop max 会破坏风险保证；把完整 acquisition policy 纳入 calibration 后恢复 finite-sample guarantee | setting 高度重合；同样反对 naive selected score | BCEA 校准整个固定 policy，不问 self-action 是否被重复计数，也不从 frozen posterior 中除 action likelihood |
| **Post-ADC Inference**, 2026 | 对主动采集与数据驱动 target 同时作 post-selection correction | selection-adjusted likelihood 是式 (1) 的统计近邻 | 不是 agent belief update / multimodal generation |
| **Causal Decoding for Hallucination-Resistant MLLMs**, 2026 | 用 `do`-calculus、detector 与 finetuned VLM 干预 previous-token confounding | “causal decoding”命名与叙事强碰撞 | 它干预 previous tokens，需要训练和 detector；不研究 observation action lineage |

### 碰撞后的诚实结论

式 (1) 的因果校正、式 (3) 的 prior subtraction、以及证据反馈导致过度自信，都分别有成熟前作。**PCU 不能以新公式或新 decoding trick 作为 Oral 主贡献。**

门控之前，唯一可能成立的 research delta 曾是一个联合新问题：

> 在主动多模态 agent 中，模型是否把 self-generated observation action 当作独立外部证据，从而让 hallucination 随闭环深度自我增强；这种失败是否能由 action/result 因子切分统一解释 crop、retrieval、tool use 和 clinical follow-up？

本次检索没有找到同时满足“self-owned action、动作产生新多模态结果、无信息重复实验、interventional quotient、跨 tool hallucination”五项的工作。但“文献空白”不等于“现象存在”；provenance gate 已否定本项目进入该问题的经验入口，因此不再据此启动后续实验。

## 6. 预注册闭门决定

### 6.1 为什么不运行 Replay

Replay–Ownership–No-Innovation 原本是**只有 provenance gate 通过才允许执行**的二阶段实验。当前 gate 不只是“没有显著”，而是核心交互显著朝相反方向，且 FP 改善被几乎等量的 recall 损失解释。因此：

- 不运行 Replay、Hulu 复现或跨工具扩展；
- 不尝试更换 provenance 句式、阈值或样本子集来重新开门；
- 不运行 PCU，因为它需要一个已经被本项目门控否定的经验前提；
- baseline 队列不受影响，可恢复原计划资源调度。

这不是“实验预算不足”的暂停，而是按预注册标准完成的方向关闭。

### 6.2 是否还有不依赖该 gate 的合法残余假设

**在 Endogenous Evidence Recycling / PCU 内部：没有。** 标准 do-identity 在数学上仍成立，但本项目没有证据表明 frozen VLM 正在重复计算 self-action likelihood。脱离现象继续测试 quotient，只会变成“拿一个因果公式寻找可用场景”。

唯一可分离出来的观察是另一个、完全不同的问题：

> **病灶外的全局解剖上下文是否提供了分布式阴性证据，而 crop 通过删除或破坏这些证据造成假阳性？**

该假设不依赖 provenance gate，因为 native 外部 context 被移除/模糊后阴性 FP 增加 `62.9pp`；纯 scale 的 FP 增加只有 `3.2pp`，且 CI 跨 0。真实 context 相对 sham 的 FP rescue 约 `9.68pp`，CI 为正，但略低于预注册 `10pp` 门槛。

不过它必须作为独立候选处理，不能写成 self-action story 的残余贡献：

1. blur、panel 边界和 context removal artifact 仍可能解释效应；
2. 它与 Fine-Grained Visual Prompting、HALC、BCEA 等 local–global/context 工作高度相邻；
3. 当前尚未证明外部 context 在 patient-specific、finding-specific 意义上提供增量信息；
4. 因而它目前既不是算法，也不具备 ICLR Oral 新颖性。

若其他研究线单独预注册这一问题，最低合法对照应固定 ROI，比较真实同患者 context、同患者扰乱 context、跨患者匹配 context 与 render-matched sham；只有真实 context 在控制 artifact 后仍提供 finding-specific 增益，才能继续。该实验不属于 PCU 的复活条件。

## 7. ICLR Oral 严格判断（门控后）

| 维度 | 最终判定 |
|---|---|
| 问题重要性 | 理论上高，但本项目未确认现象 |
| 现象证据 | **失败**：`Γ=−0.117`，CI 完全低于 0 |
| 缓解证据 | **失败**：FP `−61.3pp` 伴随 recall `−59.7pp` |
| 数学新颖性 | 低：do-transport、inverse selection 与 odds quotient 均为标准工具 |
| 方法资格 | 无：经验前提未成立，PCU 不进入实现阶段 |
| 当前 Oral-ready | **否；该主线已关闭** |

即使未来另一个真实 agent setting 观察到 self-action echo，也需要重新从现象门开始，不能引用本次 crop 结果作为先验证据。

## 8. 最终决定

1. **关闭 Endogenous Evidence Recycling / Policy-Cut Update。**
2. **不运行 Replay。** 这严格遵守“provenance gate 失败即关闭”的预注册逻辑。
3. **不把 FP 下降写成 mitigation。** 它主要来自与阳性 recall 几乎等量的整体保守化。
4. **不把 crop FP 写成 self-action 证据。** 当前最强解释是全局 context/render 改变。
5. **PCU 公式只保留为理论背景。** 它既不是本项目发现，也不是候选论文方法。
6. **全局 context 假设可独立审计，但不能救活本方向，也尚不具备 Oral 资格。**

## 9. 核实参考文献

1. Yang et al. *Fine-Grained Visual Prompting*. NeurIPS 2023. <https://proceedings.neurips.cc/paper_files/paper/2023/hash/4e9fa6e716940a7cfc60c46e6f702f52-Abstract-Conference.html>
2. Chen et al. *HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding*. ICML 2024. <https://proceedings.mlr.press/v235/chen24bi.html>
3. Zheng et al. *Large Language Models Are Not Robust Multiple Choice Selectors*. ICLR 2024. <https://proceedings.iclr.cc/paper_files/paper/2024/hash/54dd9e0cff6d9214e20d97eb2a3bae49-Abstract-Conference.html>
4. Zrnic and Candès. *Active Statistical Inference*. 2024. <https://arxiv.org/abs/2403.03208>
5. Freidling, Zhao, and Gao. *Selective Randomization Inference for Adaptive Experiments*. 2024. <https://arxiv.org/abs/2405.07026>
6. Chen et al. *Performative Prediction with Bandit Feedback*. ICML 2024. <https://proceedings.mlr.press/v235/chen24al.html>
7. Zhuang et al. *LLM Agents Can Be Choice-Supportive Biased Evaluators*. AAAI 2025. <https://ojs.aaai.org/index.php/AAAI/article/view/34843>
8. Atabuzzaman et al. *Benchmarking and Mitigating MCQA Selection Bias of Large Vision-Language Models*. EMNLP 2025. <https://aclanthology.org/2025.emnlp-main.1703/>
9. Rudman et al. *Mechanisms of Prompt-Induced Hallucination in Vision–Language Models*. ACL 2026. <https://aclanthology.org/2026.acl-long.1941/>
10. Ma et al. *What Does Vision Tool-Use Reinforcement Learning Really Learn? Disentangling Tool-Induced and Intrinsic Effects for Crop-and-Zoom*. 2026. <https://arxiv.org/abs/2602.01334>
11. Tan et al. *Causal Decoding for Hallucination-Resistant Multimodal Large Language Models*. 2026. <https://arxiv.org/abs/2602.21441>
12. Zou et al. *On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents*. ICML 2026. <https://arxiv.org/abs/2603.12109>
13. *T3: Reducing Belief Deviation in Reinforcement Learning for Active Reasoning*. ICLR 2026 Oral. <https://iclr.cc/virtual/2026/poster/10007172>
14. Xu et al. *Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable Vision-Language Models*. 2026. <https://arxiv.org/abs/2606.16667>
15. Nishino et al. *Post-ADC Inference: Valid Inference After Active Data Collection*. 2026. <https://arxiv.org/abs/2605.11511>
16. Kumaran et al. *Competing Biases underlie Overconfidence and Underconfidence in LLMs*. Nature Machine Intelligence, 2026. <https://www.nature.com/articles/s42256-026-01217-9>
17. Krishnamurthy. *Multi-agent sensing: social learning and data incest*. In *Partially Observed Markov Decision Processes*, 2016. <https://www.cambridge.org/core/services/aop-cambridge-core/content/view/319E5628A9B382732A9B520163EC85CE/9781316471104c5_p93-118_CBO.pdf/multiagent_sensing_social_learning_and_data_incest.pdf>
18. Sisk et al. *Informative presence and observation in routine health data: a review*. 2021. <https://pmc.ncbi.nlm.nih.gov/articles/PMC7810439/>

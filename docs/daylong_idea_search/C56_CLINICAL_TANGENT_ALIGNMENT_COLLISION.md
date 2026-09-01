# C56：Clinical Tangent Alignment 公式级撞车审计

**结论：严格 NO-GO；不值得进入 L0，更不应占用 GPU。**

这里审计的不是“实验可能有没有一点提升”，而是它是否还保留一个足以成为论文方法的、不可被已有工作替换的最小创新。答案是否定的：C56 是“教师 Jacobian 对齐分数”与“视觉 head 抑制”的直接组合；前者已由 Jacobian matching / gradient grounding 占据，后者已由 SPIN / CausalLens 占据。剩下的差别只是**用什么分数给 head 排序**，不是新的干预原理。

## 1. 候选方法的最简公式

### 1.1 背景：什么是 clinical tangent

令冻结的胸片专家为 $f(x)\in\mathbb R^C$，其中 $x$ 是图像，$f_c(x)$ 是疾病 $c$ 的分数。对图像求梯度得到

\[
J_f(x)=\frac{\partial f(x)}{\partial x}\in\mathbb R^{C\times d}.
\]

梯度表示“图像向哪个微小方向变化会改变专家判断”。因此，$J_f(x)$ 的行空间可被解释为专家认为具有临床意义的局部变化方向，记为

\[
\mathcal T_f(x)=\operatorname{rowspan}(J_f(x)).
\]

它的正交投影矩阵为

\[
P_f(x)=J_f(x)^\top\big(J_f(x)J_f(x)^\top\big)^\dagger J_f(x),
\]

其中 $\dagger$ 是伪逆。

### 1.2 给 VLM head 打分

设 VLM 第 $l$ 层第 $i$ 个 head 的输出为 $h_{l,i}(x)$，其图像 Jacobian 为 $J_{l,i}(x)$。候选分数是该 head 的图像敏感方向落在临床子空间中的比例：

\[
a_{l,i}(x)=
\frac{\|J_{l,i}(x)P_f(x)\|_F^2}
{\|J_{l,i}(x)\|_F^2+\varepsilon}.
\]

直观上：若一个 head 主要响应专家认可的临床方向，则 $a$ 高；若主要响应专家的零空间，则 $a$ 低。C56 再对开发集取平均 $\bar a_{l,i}$，保留高分 head、静态抑制低分 head。

这条公式本身是标准的**子空间投影能量比**，等价于平方余弦 / Rayleigh quotient；换成 principal angle 或 CCA 也不会产生新的数学对象。

## 2. 公式级碰撞矩阵

| C56 部分 | 最接近的已有工作 | 占据程度 | 审计结论 |
|---|---|---:|---|
| 用教师输出对输入的 Jacobian 定义功能方向 | [Jacobian Matching, ICML 2018](https://proceedings.mlr.press/v80/srinivas18a.html) | 完全占据基本对象 | C56 未提出新的教师函数几何 |
| 用梯度敏感性约束 VQA grounding | [HINT, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Selvaraju_Taking_a_HINT_Leveraging_Explanations_to_Make_Vision_and_Language_Models_More_Grounded_ICCV_2019_paper.html) | 高度占据 | “梯度对齐代表正确 grounding”不是新洞察 |
| 用教师梯度识别并去除无关成分 | [Learning to Focus, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/236264ac647eef86b41991d53452fd0b-Abstract-Conference.html) | 高度占据 | 已存在 teacher-gradient → prune 的逻辑 |
| 测试时选择并抑制低视觉依赖 head | [SPIN, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.631/) | 完全占据干预形式 | C56 只替换 head ranking score |
| 依据视觉敏感性选择 head、分解视觉/系统路径并干预 | [CausalLens, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.html) | 几乎完全占据系统结构 | “sensitivity-guided head intervention”已存在 |
| 按 head 的促进/干扰作用实施门控 | [Causal Head Gating, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a7f530e11fa19e9551b7a51dbd0f336f-Abstract-Conference.html) | 占据 head 功能筛选 | 独立 head 可安全裁剪的假设也不新 |
| 多头可在推理时裁剪 | [Are Sixteen Heads Really Better than One?, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html) | 占据基础操作 | 静态 head pruning 是标准操作 |

其中最致命的直接等价关系是：

\[
\text{C56}
=
\underbrace{\text{teacher-Jacobian alignment score}}_{\text{Jacobian matching/HINT/LeaF}}
+
\underbrace{\text{head suppression}}_{\text{SPIN/CausalLens}}.
\]

SPIN 已经动态保留高图像注意力 head、抑制低图像注意力 head；CausalLens 已经用视觉敏感性选 head，并在单次推理中调整视觉路径贡献。C56 的唯一变化是把内部 attention/sensitivity score 换成外部医学专家 Jacobian score。

## 3. 是否存在可辩护的最小新 delta

字面上尚未检索到完全相同的一句实现：

> “不训练 VLM，用冻结医学专家的输入 Jacobian 行空间，离线选择 VLM attention heads。”

但这不是可辩护的论文级 delta，理由有三点。

1. **它是评分器替换，而非干预原理变化。** 删除“医学专家”后，算法仍是 SPIN/CausalLens 式 head suppression；删除“head suppression”后，剩下的是标准 Jacobian alignment。
2. **组合没有产生新数学。** $\|JP\|_F^2/\|J\|_F^2$ 只是标准投影能量；CCA、主角度或低秩近似只是同一分数族。
3. **组合暴露了自相矛盾的运行假设。** 病例级 tangent 需要每个输入都运行专家并反向传播；若改成离线静态 tangent，又丢失病例和 claim 特异性。

因此，这个最小差别最多能作为“expert-Jacobian score”消融或 baseline，不能成为新方法主线。

## 4. 即使不考虑撞车，方法也有结构性缺陷

### 4.1 动态版本不再是轻量 training-free decoding

若使用 $P_f(x)$，每张测试图都必须运行医学专家并计算输入梯度。目标 VLM 虽可只前向一次，但整个系统仍多出一次专家前向和反向，不能宣称“专家只离线使用”或“单模型单次推理”。

### 4.2 静态版本的临床子空间会膨胀

若把许多病例的 tangent 合并为

\[
\mathcal T_D=\operatorname{span}_{x\in D}\mathcal T_f(x),
\]

样本增多后该空间很容易变成高秩甚至接近整个输入空间，此时 $P_D\approx I$，所有 head 的对齐分数都趋近 1，方法失去区分力。强制截断为 top-$r$ 只会引入一个开发集学习的秩和阈值，退化为普通监督式 head pruning。

### 4.3 开放生成中没有预先给定的 claim

一个 head 的 Jacobian 必须相对于某个输出 token、claim 分数或生成时间步定义。OE 生成前并不知道模型随后会声称“胸腔积液”还是“肺结节”。静态聚合会抹掉 claim 身份；逐 token 动态计算则需要昂贵反向传播，也不再简洁。

### 4.4 医学专家的零空间不等于“非临床”

教师没有响应某个方向，可能因为它没有训练该标签、存在盲点，或该方向表示正常解剖、阴性证据、位置和关系。抑制 expert-null head 会把教师盲点蒸馏进 VLM。没有任何定理能从“与教师梯度对齐”推出“减少幻觉”。

### 4.5 head 不是相互独立的临床模块

head 通过残差、MLP 和后续层共同作用。单个 head 的局部对齐分数不能保证抑制后仍保持同样的功能；现有 head gating 工作本身也需要因果验证，而不是仅凭相似度下结论。

## 5. 与现有本地证据的一致性

冻结医学专家确实能给 Huatuo 补充病例信息，但不构成跨模型的通用方法前提：

- Huatuo：macro-AUROC $0.7667\rightarrow0.8264$，增量 $+0.0598$，image-bootstrap 95% CI $[0.0397,0.0783]$。
- Hulu：$0.8606\rightarrow0.8708$，仅 $+0.0102$，低于预注册的 $0.02$ 门槛；Brier 改善 CI 跨 0。

更直接的 one-bit expert veto 也未满足“安全缓解”：

- Huatuo 移除 17.43% FP，但伤害 1.52% TP；
- Hulu 移除 17.39% FP，但伤害 2.33% TP；
- 两者都超过冻结的 1% clear-case harm budget。

证据文件：

- [`result.json`](../../corrected_runs/xrv_visual_increment_v1/result.json)
- [`one_bit_veto.json`](../../corrected_runs/xrv_visual_increment_v1/one_bit_veto.json)

这说明“专家有信息”不等于“按专家几何裁 head 就能安全纠错”，也与项目已经反复观察到的 **response is not evidence** 一致。

## 6. L0 决策

**不运行任何 L0。**

- 当前 cache 只有 pooled visual features、patch scores 和最终 margins，没有 VLM 每个 head 对图像的 Jacobian，也没有医学专家逐例梯度；无法做诚实的 CPU/cache proxy。
- 采集精确量需要 VLM per-head backward + expert backward，会占用 GPU，并可能干扰 baseline 队列。
- 更关键的是：即使实验为正，也只能证明“外部专家梯度是比视觉 attention 更好的 head 排序器”，无法修复公式级创新不足。

## 7. 最终裁决

| 问题 | 判定 |
|---|---|
| clinical tangent 数学是否新 | 否，标准 Jacobian 行空间投影 |
| head suppression 是否新 | 否，SPIN/CausalLens 已直接覆盖 |
| 是否有不可替代的最小新 delta | 否，只有外部专家评分器替换 |
| 是否满足通用、优雅、简洁 | 否；动态版昂贵，静态版丢失病例/claim 特异性 |
| 是否值得 cache/CPU L0 | 否，缺少精确缓存且 proxy 无法回答问题 |
| 是否值得 GPU probe | 否，公式碰撞已足够致死 |

**C56 从候选池关闭，保留为碰撞记录，不进入实验队列。**

# C57：Small-Specialist × Large-VLM 的 speculative / distributed 分支审计

## 结论

**严格 NO-GO，不进入 GPU。** 在冻结大 VLM、training-free、且排除融合、重排、RAG、prompt、mask、head suppression、Jacobian alignment、ontology retreat、probe 后，small-specialist 能提供的两个最自然原语都已被理论或现有工作覆盖：

1. **无偏 speculative commit** 保持大模型输出分布不变，因此数学上不可能改变幻觉风险；
2. **specialist veto + rollback + replacement** 精确等价于 verifier-guided rejection / reward tilt，创新性已被占据，而且本地固定内容量实验没有收益。

这不是说“小模型永远无用”，而是说：在当前排除条件下，不能把 verifier/gating 换成 distributed-systems 术语后当作新基本原语。

## 候选 A：Lossless Clinical Speculative Commit

设大 VLM 的下一个输出分布为 \(p\)，小 specialist 的草稿分布为 \(q\)。标准 speculative decoding 对草稿 \(c\sim q\) 的接受率为

\[
\alpha(c)=\min\left(1,\frac{p(c)}{q(c)}\right),
\]

拒绝后从归一化的 \([p-q]_+\) 中补采样。这个接受—补偿规则的目的正是使最终边际分布仍然等于 \(p\)。因此，对任意幻觉损失 \(H\)：

\[
Y_{\mathrm{spec}}\sim p
\quad\Longrightarrow\quad
\mathbb E[H(Y_{\mathrm{spec}})] = \mathbb E_{Y\sim p}[H(Y)].
\]

换成直白的话：**小模型可以让相同答案生成得更快，但只要算法仍然“无偏”，它就不能让答案更正确。** 这正是原始 speculative decoding 的保证；Leviathan 等人的 ICML 2023 论文明确强调其加速“不改变输出分布”。

- 文献：[Fast Inference from Transformers via Speculative Decoding, ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html)
- 裁决：理论 NO-GO，无需实验。

## 候选 B：Fixed-K Transactional Claim Commit

这是 distributed systems 的 two-phase commit / rollback 在医学 claim 上最自然的迁移：

1. 大 VLM 提议多个临床 claim；
2. 小 specialist 仅给出接受函数 \(A_S(c,x)\)，不生成最终报告；
3. 被 veto 的 claim 回滚，由大 VLM 的下一候选补位；
4. 每张图保持原有阳性 claim 数 \(K\)，禁止靠少说、拒答或统一阴性获益。

但其最终采样分布就是

\[
p_{\mathrm{tx}}(c\mid x)
=
\frac{p_L(c\mid x)A_S(c,x)}{\sum_{c'}p_L(c'\mid x)A_S(c',x)}.
\]

若 \(A_S\in\{0,1\}\)，它是“大模型分布在 specialist 接受集合上的条件采样”；若 \(A_S=\exp(\beta r_S)\)，它就是

\[
p_{\mathrm{tx}}(c\mid x)\propto p_L(c\mid x)\exp(\beta r_S(c,x)),
\]

即 reward-guided exponential tilt。异步执行、rollback、事务日志等系统实现不会改变这条代数。

### 公式级碰撞

- [Reward-Guided Speculative Decoding, ICML 2025](https://openreview.net/forum?id=AVeskAAETB)：small draft + reward checker + rejection/refinement；
- [Guided Speculative Inference, 2025](https://openreview.net/forum?id=cRTWN5iwiy&noteId=cRTWN5iwiy)：目标分布明确写成 \(\pi_B(y\mid x)\exp(\beta r(x,y))/Z\)；
- [Generate, but Verify / REVERSE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5eff08bd064f0cdd92182cdf6fd06b99-Abstract-Conference.html)：VLM 幻觉场景中的在线验证与 retrospective resampling。

因此 C57-B 即使有效，也更像上述方法在医学 claim 上的 hard-verifier 版本，而不是新原语。

## CPU/cache-only 致死实验

脚本：`anchor/corrected_sgta/screen_transactional_claim_commit_l0_v1.py`

结果：`corrected_runs/daylong_idea_search_v1/transactional_claim_commit_l0_v1/result.json`

实验只复用现有 VinDr confirmation claim 与 XRV specialist 分数，不占 GPU。每张图保留大 VLM 原有的阳性 claim 数 \(K\)，仅允许 specialist veto 后由更低 VLM margin 的未 veto claim 补位。预注册门槛为：两个模型都至少降低 20% 总 claim 错误，且 FN 不增加。

| 模型 | 图像 | claims | 固定总 K | 原错误 | 事务后错误 | 相对下降 | 实际替换 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Huatuo | 20 | 40 | 20 | 18 | 18 | 0% | 0 |
| Hulu | 29 | 63 | 34 | 17 | 17 | 0% | 0 |

判定：**NO-GO**。在这批满足固定 K 条件的缓存图像中，开发集拟合的 specialist 阈值没有 veto 任何一个可替换的 claim；所以不是“有效但幅度小”，而是操作根本没有触发。

这个 L0 是偏乐观的：specialist 概率与阈值用开发标签拟合，已经不符合最终希望的 calibration-free 条件；样本也不足以给出推断性置信区间。即便如此仍为 0 收益。

## 更大规模的现有反证

L0 的稀疏触发不是孤立现象：

- one-bit veto confirmation 中，Huatuo 移除 38/218 FP（17.43%），但伤害 5/330 TP（1.52%）；Hulu 移除 12/69 FP（17.39%），伤害 7/301 TP（2.33%）。两者均未达到“FP 至少降低 20%、TP 伤害不超过 1%”的门槛；
- 480 份 confirmation 报告的 fixed-K one-swap 中，Huatuo 总错误从 1613 增至 1667（相对 **恶化 3.35%**，患者 bootstrap 95% CI 为 [-6.25%, -0.71%]）；Hulu 从 1235 增至 1475（**恶化 19.43%**，CI [-24.36%, -14.98%]）。

这说明 specialist 确实含有部分信号，但把信号变成 veto / replacement 时没有跨模型的安全工作点。

## 本分支的结构性二分

在“冻结大 VLM 分布 \(p_L\)，small specialist 只参与协调”的边界内：

1. 若协调器保持 \(p_L\)，它只改变速度，不改变幻觉风险；
2. 若协调器根据 specialist 改变 \(p_L\)，它必然通过接受、拒绝、重排、限制支持集或 reward tilt 改变概率质量，落回 fusion / gating / verifier-guided sampling 家族。

所以一个真正新的 small-specialist 方法若要继续，必须改变当前二分的前提，例如引入一个**新的可观测量**或一种**不是通过输出选择来生效的模型计算操作**。在找到这样的对象之前，不应继续为本分支申请 GPU。


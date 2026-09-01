# Maximal-Coupling Grounded Decoding：公式级碰撞审计

日期：2026-08-13  
范围：仅做数学、2024--2026 文献与本地可实现性审计；未占用 GPU，未中断 baseline。

## 裁决

**严格 NO-GO，不进入实验。** 候选的归一化共同质量

\[
r(v)=\frac{\min\{p(v),q(v)\}}{\alpha},\qquad
\alpha=\sum_v\min\{p(v),q(v)\}=1-\operatorname{TV}(p,q)
\]

同时是：

1. 经典 classifier-combination 的 normalized min rule；
2. maximal coupling 在“两次采样恰好相等”条件下的公共值分布；
3. 2026 `f`-ensemble 语言模型框架中的 min ensemble。

所以它不是新的 speculative primitive。把冲突质量保留为第三状态也只能得到 disagreement abstention / conflict-to-ignorance，而不能纠正错误 claim。

## 1. 数学还原

对两个离散分布 `p,q`，定义共同子概率测度

\[
\mu(v)=\min\{p(v),q(v)\}.
\]

其总质量为

\[
\mu(\mathcal V)=\sum_v\min\{p(v),q(v)\}
=1-\operatorname{TV}(p,q)=\alpha.
\]

maximal coupling 能使两个随机变量相等的最大概率恰为 `alpha`；在相等事件上，共同 token 的条件分布正是 `mu/alpha`。因此候选不是“利用 maximal coupling 产生新的 grounded 分布”，而是**永久丢弃 maximal coupling 的 disagreement branch，并条件化在 agreement event 上**。

标准 lossless speculative decoding 恰恰不能丢弃该分支：草稿被拒绝后必须从 `[p-q]_+` 的归一化残差中补采样，才能恢复目标边际 `p`。保留补偿分支则输出分布不变，不能降低幻觉；删除补偿分支则变成有偏 expert ensemble。

## 2. 原拟定安全性质不成立

未归一化时确有

\[
\mu(v)\le p(v),\qquad \mu(v)\le q(v).
\]

但 `mu` 的总质量通常小于 1，不能直接作为 next-token 分布。一旦归一化，原性质消失。例：

\[
p=(0.9,0.1),\qquad q=(0.1,0.9).
\]

则

\[
\mu=(0.1,0.1),\quad \alpha=0.2,\quad r=(0.5,0.5).
\]

第一个 token 在 `q` 中只有 `0.1`，归一化后却被提高到 `0.5`；第二个 token 对 `p` 同理。因此“最终 token 概率不超过任何一个专家的原始支持”不能作为定理或安全保证。

若不归一化，而定义

\[
\Pr(V=v)=\mu(v),\qquad
\Pr(V=\bot)=1-\alpha,
\]

则 `bot` 仅表示两个专家不一致。它是 disagreement-based abstention；不会指出哪一方正确，也不会把 FP 修成正确 finding。在医学开放回答中，它仍通过少说或不确定化降低错误，落回 selective prediction / conflict-to-ignorance，而不是 fixed-content mitigation。

## 3. 公式级碰撞

- Kittler et al., *On Combining Classifiers*, TPAMI 1998，已把逐类最小后归一化作为标准 `min rule`：<https://doi.org/10.1109/34.667881>。
- SpecTr 把 speculative decoding 写成 token-level maximal coupling / optimal transport，并保留残差分支以严格维持大模型分布：<https://arxiv.org/abs/2310.15141>。
- Chan et al., *Ensembling Language Models with Sequential Monte Carlo* (2026)，给出通用 `f`-ensemble；明确包含 min/product 等 consensus-seeking operators，并用 byte-level SMC 处理不同 tokenizer：<https://arxiv.org/abs/2603.05432>。
- Reward-Guided Speculative Decoding 与 Guided Speculative Inference 已覆盖把小模型/奖励检查器用于有偏接受和重采样：<https://openreview.net/forum?id=AVeskAAETB>、<https://openreview.net/forum?id=cRTWN5iwiy>。
- REVERSE 已覆盖 VLM 幻觉中的在线验证、回滚与 retrospective resampling：<https://proceedings.neurips.cc/paper_files/paper/2025/hash/5eff08bd064f0cdd92182cdf6fd06b99-Abstract-Conference.html>。

因此无论解释为 min fusion、maximal coupling、跨 tokenizer ensemble，还是 verifier-guided rejection，基本操作都已有直接前件。

## 4. 与 Product-of-Experts 的关系

`min` 和 PoE 确实不是同一个数值分布：

\[
r_{\min}(v)\propto\min(p(v),q(v)),\qquad
r_{\rm PoE}(v)\propto p(v)q(v).
\]

但这种不同早已属于 generalized-mean / `f`-ensemble 中不同 consensus operator 的选择，而不是新的基本原语。两者都没有无分布假设的 FP 单调改进保证；归一化还可能放大一个专家低支持的 token。故“相对 PoE 数值不同”不足以产生医学幻觉理论。

## 5. 无同 tokenizer specialist 的实现边界

有三种实现路径，均已闭合：

1. **另一个生成模型，tokenizer 相同**：直接就是 local min ensemble。
2. **另一个生成模型，tokenizer 不同**：2026 byte-level SMC `f`-ensemble 已明确解决；其报告开销约数十到上百秒/例，也不满足低时延目标。
3. **非生成式影像专家（如 XRV）**：它只输出 finding posterior，不能定义整个 vocabulary 上的 `q(v)`。把 finding 分数映射到词或完整句后，方法变成 claim-level reward tilt、hard veto、replacement 或 ontology constraint，分别落回 C44/C47 与 C57 的已失败分支。

本地证据也不支持把它作为通用方法：XRV 对 Huatuo 有条件增量，但 Hulu 仅 `+0.0102` AUROC 且 Brier CI 跨零；one-bit veto 两模型都未同时满足“移除至少 20% FP、误伤不超过 1% TP”；fixed-K replacement 还分别恶化 `3.35%` 和 `19.43%`。

## 6. 最终边界

该方向形成一个完整二分：

- 保留 maximal-coupling 的补偿残差：边际仍为大 VLM 的 `p`，幻觉风险不变；
- 删除补偿残差并归一化共同质量：标准 min ensemble；
- 把缺失质量送入 `unknown`：标准 disagreement abstention；
- 用 non-generative specialist 给缺失分支补位：标准 verifier/reward-guided decoding。

不存在尚待实验辨别的公式级新分支。因此不应运行 32 例 canary；效果即使为正，也只能证明经典 min-rule ensemble 在某一模型上有效，不能支持 ICLR 方法创新。

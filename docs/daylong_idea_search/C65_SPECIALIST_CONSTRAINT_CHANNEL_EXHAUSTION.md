# C65 — Specialist Constraint Channel Exhaustion

> 日期：2026-08-13  
> 裁决：**在当前排除项下 NO-GO，不进入 GPU。** 冻结主 VLM 时，一个小医学模型若真的改变生成结果，只能经由输入、表示、输出分布、搜索或后处理五条通道；它们分别回到 prompt/visual prompt、feature steering、guidance/fusion、rerank/veto/constrained decoding 和 editing。控制屏障函数是其中最优雅的写法，但已有直接同构工作，且本地专家约束不够可靠。

## 1. 冻结问题

目标是假设主 VLM 参数完全冻结，小医学模型只给一个低时延、病例相关的“可验证约束”，主 VLM 仍负责完整自然语言生成。候选更新律不能是：

- feature fusion / hidden-state steering；
- logit fusion、层融合、energy/reward guidance；
- candidate reranking、veto、claim exchange；
- prompt/RAG、双中心化或生成后编辑。

问题是：排除这些类别后，是否仍存在一个会改变输出错误率的 training-free 更新通道？

## 2. 约束通道穷尽定理

设冻结模型在前缀 `s_t=(x,y_{<t})` 上的 next-token 分布为

\[
p_t(v\mid s_t).
\]

小模型输出约束 `z=C(x)`，任一 stochastic wrapper 最终都定义另一个 next-token 分布

\[
q_t(v\mid s_t,z).
\]

标准 softmax 对每个词都有正概率。因此，只要 wrapper 不新增模型词表外的输出，就总能写成

\[
q_t(v\mid s_t,z)
=
\frac{p_t(v\mid s_t)\exp r_t(v;s_t,z)}
{\sum_u p_t(u\mid s_t)\exp r_t(u;s_t,z)},
\]

其中

\[
r_t(v;s_t,z)=\log q_t(v\mid s_t,z)-\log p_t(v\mid s_t)+\text{constant}.
\]

这不是一种特定算法，而是有限离散分布的恒等式。它说明：

1. 任何只改变 token 概率的软约束方法，代数上都是对 base logits 加一个 reward/energy，即 guidance 或 probability fusion；
2. 若某些词被设为零概率，相当于 `r=-infinity`，即 support mask、veto 或 constrained decoding；
3. 若先产生若干 continuation 再选一个，最终仍是 search/reranking；随机选择也只是在边际上定义另一个 `q`；
4. 若修改 hidden state `h_t` 后再交给 LM head，则属于 representation intervention / feature steering；
5. 若修改 `x` 或已生成文字，分别属于 prompt/visual prompt 与 post-hoc editing。

反之，如果 wrapper 既不改输入、表示、输出分布、搜索，也不改最终文字，那么每一步仍有 `q_t=p_t`。由概率链式法则，完整序列分布完全相同：

\[
q(y\mid x)=\prod_tq_t(y_t\mid x,y_{<t})
=\prod_tp_t(y_t\mid x,y_{<t})=p(y\mid x).
\]

所以任何期望 hallucination risk 也完全不变。

**结论：**在冻结生成器下，排除 prompt、feature update、guidance、rerank/veto 和 editing 后，没有第六条能改变输出分布的算法通道。这是计算图与概率分解的穷尽结果，不是经验猜测。

## 3. 最优雅候选为何仍然碰撞：Clinical Control Barrier Filter

控制论中最自然的候选是 control barrier function（CBF）。设 `b_z(h)>=0` 表示当前生成状态位于专家定义的临床安全集，冻结 VLM 的一步动力学为 `F(h)`，则求最小控制：

\[
u_t^*=\arg\min_u\|u\|_G^2
\quad\text{s.t.}\quad
b_z(F(h_t)+Bu)\ge(1-\alpha)b_z(h_t).
\]

直观上，它不让小模型接管报告，只在下一步即将越出“临床安全区”时做最小修正。这一写法简洁、通用、具有安全不变集解释。

但它不能成为本项目的新原语：

- [CBF-LLM](https://arxiv.org/abs/2408.15625) 已经使用小 RoBERTa evaluator 与 CBF safety filter 干预冻结 Llama 的 token generation；
- [Control Barrier Function for Aligning Large Language Models](https://arxiv.org/abs/2511.03121) 明确提出 add-on、无需微调主模型，并允许已有 evaluation model 直接进入 filter；
- [BarrierSteer](https://arxiv.org/abs/2602.20102) 已把非线性 barrier 写进 latent space 并在线 steering；
- [Provably Safe Generative Sampling with Constricting Barrier Functions](https://arxiv.org/abs/2602.21429) 已通过逐步收紧的 safety tube 与每步凸 QP 最小化 KL/distribution shift。

若 `b_z` 定义在词或 claim 概率上，QP 退化为 constrained decoding / posterior projection；[NeuroLogic Decoding](https://aclanthology.org/2021.naacl-main.339/) 已覆盖逻辑约束下的 training-free generation，[Controlled Decoding](https://openreview.net/forum?id=jo57H1CpD8) 的最优策略也是 `p exp(value)` 的指数倾斜。若 `b_z` 定义在 hidden state 上，则必须学习 barrier/readout 和作用矩阵，落回 feature steering，并与 BarrierSteer 直接相邻。

因此 CBF 是**唯一最佳控制论候选**，但也是明确的公式级碰撞，不能换成“Clinical CBF”后当新方法。

## 4. “小模型预测”不等于“可验证约束”

令专家给出的允许输出集合为 `Gamma_S(x)`，真实临床允许集合为 `Gamma_Y(x)`。一个 hard wrapper 保证

\[
\hat y\in\Gamma_S(x).
\]

要从这个保证推出临床正确，至少需要专家约束的 soundness：

\[
\Gamma_S(x)\subseteq\Gamma_Y(x).
\]

若事件

\[
U=\{\Gamma_S(x)\cap\Gamma_Y(x)=\varnothing\}
\]

发生，则任何严格满足专家约束的 wrapper 都必错。特别地，在 base 原本正确的样本集合 `B` 上，新增伤害至少为

\[
P(B\cap U).
\]

所以要求 clear-case harm 不超过 `1pp`，实际上要求该专家约束在 base-correct 条件下的致命不可靠率低于 `1%`。一个普通分类器 posterior 没有这种保证；它是第二份有噪意见，不是形式约束。

本地结果正好验证了这一风险：

- XRV 相对 VLM final margin 的 conditional AUROC 增量在 Huatuo 为 `+.0598`，但 Hulu 只有 `+.0102`，不具跨模型稳定性；
- one-bit expert veto 在 Huatuo/Hulu 只去除约 `17.4%` FP，却分别误伤 `1.52%/2.33%` TP；
- fixed-K one-swap 在 480 份 confirmation 报告上使总错误分别恶化 `3.35%` 和 `19.43%`；
- 小缓存上的 transactional replacement 没有触发可用交换。

这不是 CBF 参数未调好，而是 barrier 本身没有达到硬约束所需的 soundness。

## 5. 与信息论、统计决策的对应

### 信息论

若只把小模型输出 `z` 作为 side information，任何真正使用 `z` 的 decoder 都在构造 `q(y|x,z)`。不改变 `q` 就不改变风险；改变 `q` 则必然落入上一节的 density-ratio `q/p`。因此“信息进入系统”和“通过什么计算通道影响输出”不能分开讨论。

### 统计决策

若小模型只给一个 binary constraint，最优行动仍需知道 false-veto 与 missed-error 的代价和条件概率。没有可靠度模型时，hard constraint 只是 AND classifier；有可靠度后，Bayes rule 变成阈值/score fusion。不存在既使用 noisy expert、又完全免于校准与决策权衡的更新律。

### 控制论

CBF 能保证的是相对于所定义 barrier 的 forward invariance，而不是 barrier 与真实临床正确性的等价。错误的安全集可以被非常稳定地维持；控制理论不能替代医学约束的 soundness。

## 6. 当前唯一未被边界否掉的特殊情形

如果把“小医学模型约束”放宽为**可由物理、元数据或形式规则验证的约束**，则现有 C58 Frame-Covariant Decoding 是当前唯一幸存者：

- DICOM/图像坐标变换给出精确 group action，不需要相信另一个 classifier 的诊断；
- 约束只作用于 laterality/location attribute，不接管 finding identity 或报告语言；
- Huatuo 的 13 个 parseable named laterality 样本上，直接问 patient-frame 只有 `1/13` 正确，而先在 screen-frame 回答再按已知变换编译为 patient-frame 得到 `12/13`，改善 `+84.62pp`，bootstrap CI `[+53.85,+100]`，非方位内容保持 `100%`；
- 已有 cross-model + natural OE fatal runner，但尚未占 GPU。

它仍然只解决“坐标系混淆导致的侧别幻觉”，不是通用医学 hallucination；但它满足“至少解决一个明确子问题”，且与 noisy specialist constraint 有本质差别：约束是真正可验证的。

## 7. 严格裁决与下一动作

| 候选 | 数学身份 | 新颖性/可行性裁决 |
|---|---|---|
| Clinical CBF | CBF-QP safety filter | 与 CBF-LLM / BarrierSteer 直接碰撞 |
| Posterior projection | KL/Bregman projection 到约束集 | posterior regularization / constrained decoding |
| Small-model token reward | `q proportional p exp(r)` | guidance / product-of-experts |
| Hard clinical constraint | support restriction | veto / constrained decoding；受专家误差下限限制 |
| Latent minimal control | hidden-state projection | feature steering；需 learned barrier |
| 不改变上述任一对象 | `q=p` | joint law 不变，无法改善风险 |
| Exact frame law | group-covariant attribute compilation | **唯一当前幸存的明确子问题；需完成跨模型 natural OE 致死实验** |

最终决定：

1. 不为 noisy small-model constraint 启动 GPU；在当前排除项下不存在未占据的第六种更新律。
2. 不把 CBF 的控制论语言包装成医学新方法。
3. 下一份可执行实验只能选择 C58：等待 baseline GPU 空档后，在不终止 baseline 的前提下运行 cross-model named + natural OE canary；通过才扩展到更一般的 reference-frame attributes。
4. 并行继续生成**数据源不同**的新候选：只允许 exact metadata、known acquisition geometry、formal clinical consistency law 或新可观测信号，不再对同一 noisy XRV posterior 反复更换优化器。


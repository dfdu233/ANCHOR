# Bayes Reciprocal Score：致死实验预注册

## 1. 问题与直觉

一个医学 VLM 说“这张片有胸腔积液”，不等于这个结论真由图像产生。我们用一个独立的医学图像生成模型提出更强的问题：

> VLM 认为“哪种图像变化会更支持该 claim”，是否与真实医学图像分布中“有该病与无该病的变化”一致？

若一致，claim 至少具有生成–判别互易的视觉依据；若不一致，高置信可能来自语言先验、伪影或离开真实图像流形的敏感方向。

## 2. 数学背景

设 `x` 是图像，`c` 是一个临床 claim。Bayes 定理给出：

\[
\log p(c\mid x)=\log p(x\mid c)-\log p(x)+\log p(c).
\]

对图像 `x` 求梯度后，与具体图像无关的先验项 `log p(c)` 消失：

\[
\underbrace{\nabla_x\log p(c\mid x)}_{\text{判别模型的 claim 方向}}
=
\underbrace{\nabla_x\log p(x\mid c)-\nabla_x\log p(x)}_{\text{生成模型的条件密度比方向}}.
\]

右边正是 conditional diffusion score 与 unconditional diffusion score 之差。这个恒等式是 classifier guidance 的经典基础，**不宣称为理论新意**。待验证的新问题是：独立训练的医学 VLM 和医学 diffusion 是否在幻觉 claim 上系统性违反该互易性，以及这个违反能否用于固定覆盖率的纠错。

## 3. 可计算量

在 diffusion 潜空间的同一噪声时刻 `t`，定义：

\[
g_t=\nabla_{z_t}\log q_{\mathrm{VLM}}(c\mid D(z_t)),\qquad
s_t=s_{\mathrm{diff}}(z_t,c)-s_{\mathrm{diff}}(z_t,\varnothing).
\]

其中 `D` 是 diffusion VAE decoder。互易分数使用正向投影比例：

\[
R_t(c,x)=
\frac{\langle g_t,s_t\rangle_+^2}
{(\lVert g_t\rVert^2+\epsilon)(\lVert s_t\rVert^2+\epsilon)}.
\]

`R_t` 介于 0 和 1：1 表示 VLM 的局部 claim 敏感方向几乎可全部由医学生成分布解释；0 表示两者正交或反向。实验同时使用对称有限差分

\[
\Delta_t(c,x)=m_c(z_t+\eta s_t)-m_c(z_t-\eta s_t)
\]

作为无需对 VLM 反传的黑盒近似。一阶 Taylor 展开给出 `Δ_t ≈ 2η⟨g_t,s_t⟩`。

## 4. 与最近工作的边界

| 工作 | 其操作 | 本候选必须保持的差异 |
|---|---|---|
| Diffusion Classifier | 用生成似然单独分类 | 检验 VLM posterior 与 diffusion density-ratio 的病例级兼容性 |
| DeGF | 生成图像反馈后再解码 | 不生成完整替代图；比较原图邻域的对称 score 方向 |
| CIPHER | 离线 diffusion 反事实构造低秩幻觉子空间 | 不学全局子空间；测量每个病例、每个 claim 的 Bayes 互易性 |
| CoEV | 遮挡临床证据区做双向验证 | 不依赖定位区遮挡；使用真实图像流形的局部密度比方向 |
| GenRep | 训练时联合生成与感知并对齐参数梯度 | 冻结现成模型；检验输入梯度的概率兼容性，不联合训练 |

若实现最终变成“将 diffusion/CLIP 分数与 VLM 分数加权”，则退化为普通 external verifier，必须关闭。

## 5. 分阶段致死门

### L0：廉价独立视觉增量

- 数据：VinDr 7 findings，只用 0/3 与 3/3 reader vote，image-disjoint dev/confirmation。
- 外部模型：冻结 BiomedCLIP，只作 diffusion 之前的廉价 proxy。
- 对照：finding identity + VLM final margin。
- 增强：对照 + BiomedCLIP positive-minus-negative score。
- PASS：Huatuo 和 Hulu 的 confirmation macro-AUROC 均提高至少 .02，image-bootstrap 95% CI 下界 > 0，NLL 改善 CI 下界 > 0。
- FAIL：不下载 diffusion checkpoint，关闭本轮 reciprocal-score 路线。

### L1：互易分数是否识别真伪 claim

- 模型：Huatuo + Hulu；每模型 128–256 个平衡 claims。
- 生成模型：冻结开源 CXR text-conditioned diffusion；先验证其 positive/negative prompt 在 VinDr 上具有正确方向。
- 主检验：`R` 或 `Δ` 在 final margin + finding identity 之上 AUROC 增量至少 .02，且 95% CI 排除 0。
- 特异性：正确 claim prompt 必须超过 off-claim、打乱 prompt、同范数随机方向与只用 diffusion likelihood 的基线。
- 若只有 diffusion likelihood 有效而 reciprocity 无增量，则关闭方法新意，将结果降级为 external-model ensemble。

### L2：固定覆盖率纠错

- CE：仅在 L1 通过时，用 dev-only 参数将互易分数并入三态 claim 分布。
- OE：草稿 claims + 固定 ontology，保持每例原始阳性 claim 数 `K`，只做弱互易 claim 与强互易遗漏 claim 交换。
- GO：fixed-K hallucination 相对降低至少 20%，omission 不增，clear-case 下降不超过 1pp，至少两模型。
- 任何靠删 claim、缩短报告、统一阴性或拒答的改善均判 FAIL。

## 6. 结果与关闭原因

L0 在 1,003 张 VinDr 影像上完成，未占用 4090，baseline 队列未中断。

| VLM | final-margin macro-AUROC | +BiomedCLIP | 增量 | image-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Huatuo | .7667 | .8084 | +.0417 | [.0239, .0596] |
| Hulu | .8606 | .8634 | +.0027 | [-.0028, .0084] |

Hulu 失败使预注册的双模型门判为 `NO_GO`。这个结果说明：外部视觉模型可以补充某些较弱 VLM，但不是稳定的跨模型增量来源。

其次，数学自审发现一个更根本的不可识别性：`∇x log p(c|x)` 只指出“往哪改图会让 c 更可能”。对当前真阳性图像和真阴性图像，这个方向都可以与 diffusion 条件方向同向；方向一致不能判断当前图像是否已支持 claim。梯度只把 log-density ratio 确定到一个加法常数；要恢复当前支持度，必须做路径积分或直接估计 `log p(x|c)-log p(x)`，这会退化为 Diffusion Classifier/生成似然验证。

因此本路线在 L0 关闭：不下载 diffusion checkpoint，不启动 L1，不将 Bayes 恒等式包装成新理论或幻觉缓解方法。

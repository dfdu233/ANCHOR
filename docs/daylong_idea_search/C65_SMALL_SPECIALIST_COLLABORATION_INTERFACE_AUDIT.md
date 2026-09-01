# C65 — Small Medical Specialist × Frozen VLM：协作接口穷举与下一合法种子

> 日期：2026-08-13  
> 目标：寻找一种 training-free、低时延、改变生成计算本身、直接降低 fabricated positives，且不是 posterior constraint / phrase guidance / rerank / veto 的大小模型协作原语。  
> 裁决：**在这些约束同时成立时，没有诚实的未占据接口；不应继续给 fusion/verification 换名。** 若允许最小幅度的接口训练，`write-protected clinical memory` 是一个可研究种子，但尚非 ICLR-ready。

## 1. 为什么严格约束下接口几乎为空

令大 VLM 为 `L`，小医学专家为 `S`。输入图像为 `x`，输出序列为 `y`。任何让 `S(x)` 改变 `L` 结果的算法，都必须在下面四个位置之一介入：

1. **观察前**：改变 `x`；
2. **表征/计算中**：改变 visual tokens、attention、hidden states、KV 或计算路径；
3. **决策时**：改变 next-token / claim 分布或候选集合；
4. **决策后**：删、改、重排或验证生成内容。

这不是分类术语，而是按因果图中的介入位置穷举。对应到已有方法：

| 介入位置 | 小模型最自然的输出 | 算法实际身份 | 本项目证据/碰撞 |
|---|---|---|---|
| 观察前 | ROI、heatmap、warp、normal reconstruction | visual prompt / crop / foveation / counterfactual input | C33/C50/C54/C55；Learning-to-Zoom、LZU、医学 anomaly residual |
| 计算中 | feature、metric、graph、memory key | adapter / feature fusion / attention bias / token mixing / steering | C56/C59/C60/C61；Chimera、VILA-M3、DINO-HEAL、SPIN |
| 决策时 | posterior、score、acceptance | PoE/fusion / contrastive decoding / constrained decoding / reward tilt | C44/C46/C47/C52/C57；CCD、CoS、guided speculative inference |
| 决策后 | flag、veto、replacement | verifier / rerank / report rewriting | C47/C53/C57；CoEV、CXR-Agent、REVERSE |

如果再要求算法**保持大模型原分布**，标准 speculative correction 可加速但不能改变任何 hallucination loss：

\[
Y_A\sim p_L(\cdot\mid x)
\quad\Rightarrow\quad
\mathbb E[H(Y_A)]=\mathbb E_{Y\sim p_L}[H(Y)].
\]

所以严格边界形成一个二分：

- 不改变 `p_L`：不能提高正确率；
- 改变 `p_L`：必然通过观察、隐藏计算、决策或后处理中的至少一个接口，而这些接口已被当前排除条件逐项删除。

这说明本轮反复失败不是“还没找到足够 fancy 的数学”，而是约束组合本身把可实现空间清空了。

## 2. 对已尝试大小模型角色的完整审计

### 2.1 小模型给病种概率

TorchXRayVision 对弱一些的 Huatuo 有真实增量：macro AUROC `.7667→.8264`，但 Hulu 只有 `.8606→.8708`。把概率用于 logits、固定 K 交换或 one-bit falsification 后，分别落入 fusion、replacement 和 veto；真实 confirmation 没有跨模型安全工作点。

### 2.2 小模型给空间地址

- 用地址裁剪/放大：foveation；
- 用地址改变 attention：mask/bias；
- 用地址给视觉 token 加语言 header：region token / visual prompt；
- 用地址做 coarse-to-fine refinement：TokenPacker/LLaVA-HR 或经典 multigrid lifting。

CRR 的新审计进一步证明，`fine = coarse + zero-mean detail` 是标准 wavelet/multigrid；一个 coarse token 替换为多个 children 还会改变 softmax 分母、causal order 与 RoPE，因而 feature mean 守恒不等于生成函数守恒。

### 2.3 小模型给 feature geometry

Mahalanobis metric 可以吸收到 Q/K adapter；pairwise metric 等于 attention bias；specialist kernel 要么是 bias，要么是 tensor-product feature fusion。跨模型空间还需要 translator，无法凭 training-free 几何自然对齐。

### 2.4 小模型给 claim 或报告草稿

若小模型生成内容、大模型润色，它是 specialist-observes/generalist-realizes 的 structured realizer；若大模型检查小模型，是 verifier/speculative decoding。两者都实用，但不是当前要求的新计算原语。

## 3. 唯一值得放宽的假设：允许一个很小的共享接口训练

严格 training-free 不是免费的优点：两个冻结模型没有共享 codebook，迫使所有协作退回概率融合、文字提示或 OOD feature 注入。最小且科学上合理的放宽是：

> 两个主干仍冻结，但允许学习一个很小、显式受约束的 specialist-to-VLM interface。

### 候选：Write-Protected Clinical Memory（WPCM）

小医学模型不输出诊断答案，只产生 patch-level anatomy key 与局部医学 feature。一个小 translator 将它们写入独立的视觉 fast-weight memory：

\[
M_i=M_{i-1}+\beta_i\big(v_i-M_{i-1}k_i\big)k_i^\top.
\]

背景说明：`k_i` 是第 `i` 个视觉区域的地址，`v_i` 是该区域要保存的病例视觉值；括号里的量是“当前 memory 还没写对的残差”。这就是 delta-rule memory 的核心更新。生成文字只能读取

\[
r_t=M q_t,
\]

但不能写回 `M`。因此视觉病例状态和自回归语言状态被物理分开：前者 write-once/read-many，后者照常生成。

### 可证明但不能夸大的性质

若地址 keys 单位正交，且 `\beta_i=1`，顺序写入后有

\[
Mk_i=v_i,
\]

且写入新 key 不改变旧 key 的读取结果。换言之，语言 token 无法污染视觉 memory；不同地址也不互相覆盖。这是 delta rule / associative memory 的标准性质，不能当作新数学定理。

真正可能新的医学命题只能是：

> fabricated positive 是否来自“临床语言在共享 residual stream 中形成了无视觉写入的 association”，而把 visual state 变成只读、可寻址 memory 后，这类错误是否下降。

### 为什么它还不是当前答案

1. 需要训练 translator/read interface，不符合严格 training-free；
2. fast-weight / delta-rule memory、cross-attention memory 与 domain-expert feature integration 都已有强邻近工作；
3. 本地 C34 没发现 claim-level FP 自激，视觉 memory contamination 尚无自然正现象；
4. 若读 memory 的结果直接加 logits，它又退化为 expert guidance；必须在架构训练中证明 read-only state 的作用，而不是推理时硬插。

因此 WPCM 只是一条**允许轻量训练后可继续调查的架构种子**，不是现在可运行、可命名的 ICLR 方法。

## 4. 另一个严格 training-free 的近似种子为何也不够

可用医学小生成模型合成 patient-specific healthy twin `x_0`，再把 `x-x_0` 作为异常 innovation 交给 VLM。它满足“一次 specialist + 一次 VLM”，也直接面向 fabricated positive；但：

- healthy reconstruction residual 是医学 anomaly detection 的成熟对象；
- MIDL 2022 已系统说明 reconstruction error 会被模型重建误差淹没；
- 胸片 normal synthesis / disease residue 已有直接模型；
- 接入冻结 VLM 只能成为 pseudo-RGB、visual prompt、feature subtraction 或 contrastive input；
- 没有共享 codebook 时，冻结 VLM 未必把 residual 解释成病灶。

所以它可作为 future baseline，不能作为 Oral 主线。

## 5. 对下一步的严格建议

### 若坚持全部原约束

停止生成新方法名。没有候选同时满足：training-free、冻结异构模型、非 prompt/fusion/attention/constraint/verifier、低时延、改变结果、且有 correctness property。

### 若真正想要创新方法

只放宽一条：允许 `<1%` 参数的共享接口训练。然后先做一个 32 例致死门，而不是先造完整系统：

1. 冻结 Huatuo/Hulu 与一个医学 patch encoder；
2. 训练同预算的三种接口：普通 feature concat、普通 cross-attention、write-protected memory；
3. 保持视觉 FLOPs、token 数与训练数据完全相同；
4. 只取 native greedy 的 fabricated positives 与 matched true positives；
5. 主门：WPCM 相对最强普通接口额外移除至少 20% FP，TP 伤害 `<=1pp`，patient bootstrap CI 排除 0；
6. 因果控制：允许 language tokens 写 memory 后收益应消失；打乱 anatomy keys、交换病人 memory、随机等范数 memory 均应失败；
7. 若 read-only 与普通 cross-attention无差异，立即关闭，说明“memory contamination”只是架构叙事。

这个实验需要训练和新 GPU 预算，当前不应抢占 baseline；只有先在现成小 panel 上发现“write protection > same-capacity fusion”的正差异，才值得放量。

## 6. 最终结论

**严格 training-free 小模型协作不是当前缺少的灵感，而是一个已被接口穷举封闭的空间。** 小专家当然能提升弱 VLM，但只要不学习共享表示，它只能通过提示、概率、热图、feature 注入或验证生效；这些不是新的基本原语。

真正可能产生创新结果的转移，不是再换一个数学名词，而是研究一个具体、可证伪的架构现象：

> 将病例视觉状态与语言生成状态分成 write-protected memory 和 mutable language state，是否能阻断“无视觉写入的阳性 association”。

这条线需要最小接口训练；在严格 training-free 条件下，不应虚假承诺存在一个未被占据的大小模型方法。

## 参考近邻

- Fu et al., *Fast Large Language Model Collaborative Decoding via Speculation*, ICML 2025.
- Nath et al., *VILA-M3: Enhancing Vision-Language Models with Medical Expert Knowledge*, CVPR 2025.
- Peng et al., *Chimera: Improving Generalist Model with Domain-Specific Experts*, 2024/2025.
- Li et al., *DINO-HEAL / VidHalluc*, CVPR 2025.
- Yang et al., *Gated Delta Networks*, ICLR 2025.
- Meissen et al., *On the Pitfalls of Using the Residual Error as Anomaly Score*, MIDL 2022.
- Tang et al., *A Disentangled Generative Model for Disease Decomposition in Chest X-rays via Normal Image Synthesis*, Medical Image Analysis 2021.

# C54 Specialist-as-Encoder：严格机制与碰撞审计

审计日期：2026-08-13  
资源边界：公式级审计与本地 artifact 检查；未运行 GPU，未修改 baseline 队列。

## 裁决

> **作为 ICLR 级新算法原语严格 NO-GO；不进入 CPU/GPU 效果筛选。**

“专家不投票，而把 VLM 无法预测的病例级创新图编码进 RGB nullspace”有一个好的工程动机，但它没有形成新的证据代数。它精确分解为三个已有模块：

1. `specialist residual = specialist - prediction from VLM`：conditional residual stacking / partial regression；
2. `specialist map -> modified input image`：auxiliary-model heatmap visual prompting；
3. `payload -> luma-null RGB`：固定线性 side-channel / reversible color mapping。

把三个已有模块串联不产生新的可识别性质。更致命的是，**training-free 与可解释通信不能同时满足**：可见热图能利用 VLM 已学会的颜色/标记语义，但直接撞上 visual prompting；不可见或 luma-null 数值码没有与冻结 VLM 共享的语义 codebook，第一层“可见”不意味着模型知道正负、病种身份或证据强度。

## 1. 冻结候选并还原代数

令小医学专家对 finding `c` 产生空间证据图

```text
H_c(x) in R^(h×w),
s_c(x) = b_c + mean(H_c(x)).
```

本地 TorchXRayVision DenseNet 的 CAM 正好满足后一恒等式，而不是近似 Grad-CAM。令 `M(x)` 为 VLM 可获得的图像/空间表征。候选的“创新”定义为

```text
U_c(x) = H_c(x) - E[H_c(x) | M(x), c].
```

再将它编码为

```text
X_c = q·1 + alpha U_c v,     w^T v = 0,
```

送入冻结 VLM。

### 1.1 空间求和后就是 residual stacking

若条件预测器保持均值，则

```text
mean(U_c)
= s_c - b_c - E[s_c-b_c | M,c].
```

右侧就是专家分数对 VLM 信息做回归后的 residual。在线性版本中，`U=(I-P_M)H` 是 Frisch--Waugh--Lovell partialling-out；把 `U` 再给下游模型就是 residual stacking。把 residual 保留为空间图不会改变其统计身份，只是保留了更多坐标。

而且 `U` 的“不可预测”只有正交性含义：

```text
E[U | M,c] = 0.
```

它不推出 `U` 是正确临床证据。专家噪声、设备 shortcut 和病种标签错配也都可能位于该正交补中。

### 1.2 在 VLM 第一层它就是固定 side-input adapter

若 RGB patch 投影为线性算子 `A`，则

```text
A(X_c) = A(1)q + alpha A(v)U_c.
```

所以 nullspace 只把专家图乘以一组固定有效权重 `A(v)` 后注入视觉 token。这和把 `U_c` 作为第二输入、经过一个固定线性 adapter 相同。若后续输出对小扰动一阶展开，

```text
delta z_c ~= alpha <grad_X z_c, U_c v>,
```

仍是一个隐式的 expert-score adjustment；没有出现不同于 fusion / prompting 的决策对象。

## 2. 直接文献碰撞

### 2.1 Auxiliary heatmap visual prompting 已有精确系统近邻

- **Attention Prompting on Image (ECCV 2024)**：辅助 CLIP/LVLM 先生成 query-dependent attribution heatmap，再把 heatmap overlay 到原图并输入目标 LVLM；论文明确同时讨论 `g=f` 的 self-reflection 与 `g!=f` 的 model ensemble。C54 将 overlay 改成 luma-null blending，没有改变“辅助模型证据图通过像素提示目标 VLM”的系统结构。  
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04374.pdf
- **Visual Evidence Prompting (ACL 2025)**：小目标检测/scene-graph 专家先提取视觉证据，再提示冻结 LVLM 以减少 hallucination。它使用自然语言载荷，C54 使用像素载荷；“specialist 为 generalist 提供证据”本身已经是其核心贡献。  
  https://aclanthology.org/2025.acl-long.205/
- **Black-Box Visual Prompt Engineering (2025)**：直接将 bbox/circle 等专家或对象 cue 叠到图像以降低 hallucination，并训练 router 选择提示。C54 不训练 router，但也失去 learned selection。  
  https://arxiv.org/abs/2504.21559
- **CoEV (2026)**：医学 VLM 中用专家空间证据核验、纠正生成 claim；C54 即使不显式投票，最终若依据专家图修改 claim，功能目标仍处于同一邻域。  
  https://arxiv.org/abs/2606.18609

因此，“specialist-as-encoder instead of voter”是接口实现差异，不是机制差异。

### 2.2 隐藏像素通信也不能提供新语义保证

隐写与 steganographic prompt-injection 已证明 VLM 会对人眼近似不可见的像素载荷响应；但这只说明存在操纵通道，不说明冻结 VLM 能正确解释任意新的医学码字。Invisible Injections 报告的正是隐式载荷导致行为操纵，而非可靠的、无需训练的语义通信。  
https://arxiv.org/abs/2507.22304

更直接地，2025 年 PET/CT 工作已经把高位灰度/HU/SUV 可逆地分配到 RGB color bytes，使 PNG 保留原始定量值。它削弱了“把科学数值编码进 RGB 冗余通道”的表示新颖性；C54 剩下的差异只是让 frozen VLM 充当一个未经训练的 decoder。  
https://doi.org/10.3390/info16050352

## 3. Training-free 可解释协议为何不存在

候选存在一个不可消除的二分：

### A. 使用 VLM 已理解的 visual vocabulary

例如红色热图、框、箭头、轮廓或透明 mask。这样 VLM 可能理解“这里值得看”，但：

- 语义来自训练中已有的颜色/标注约定；
- 直接属于 API / BBVPE / SoM / visual evidence prompting；
- 会改变普通图像语义，不再是 legacy-preserving nullspace channel。

### B. 使用数值型 luma-null 隐藏码

这样可保持标准灰度，但冻结 VLM 从未学习协议

```text
chroma sign -> supports/refutes,
chroma magnitude -> evidence strength,
channel/band -> finding identity.
```

因此它最多对颜色纹理产生 OOD 响应。若通过训练 adapter、codebook、prompt demonstrations 或标签选择 carrier 来教会解码，就不再 training-free，并退化为 learned visual prompt / model communication adapter。

所谓 `E[H|M,c]` 也必须从数据估计。无标签回归虽然不使用真值，仍然是在训练跨模型 translator；若完全不拟合，只做解析投影，则不能称为“VLM 无法预测的 innovation”。

## 4. 一次前向约束下的 claim-identity 矛盾

开放报告同时有 `C` 个候选 findings，每个有不同 `U_c(x)`。RGB luma-null 每个像素只有二维载荷空间：

```text
dim ker(w^T) = 2.
```

单张编码图不能无损、可解释地承载任意多个 claim-specific maps。只能选择：

1. 为每个 claim 生成一张图：需要 `C` 次 VLM 前向；
2. 先由 VLM 草拟 claim 再编码：至少两次 VLM 前向；
3. 将所有 maps 求和/取最大：丢失 claim identity，VLM 无法知道哪个区域对应哪个病种；
4. 将病种分频/分通道：需要目标 VLM 学习 codebook，破坏 training-free。

因此“多 claim 开放生成 + 一次 VLM 前向 + training-free + specialist innovation map”四项不能由当前协议同时满足。这是结构性矛盾，不是超参数问题。

## 5. 本地证据也不支持继续筛效果

- C46 已测 XRV 分数相对 VLM final margin 的条件增量：Huatuo `+.0598`，但 Hulu 仅 `+.0102` 且 Brier CI 跨 0；专家创新不跨模型成立。
- C47 将同一专家用于阳性 claim veto：两模型都只移除约 `17.4%` FP，同时误伤 `1.52%/2.33%` TP，未过 `20% FP / 1% TP` 门。
- C53 已确认多个医学专家 CAM 的交集直接碰撞胸片 Ensemble-CAM；空间形式没有自动解决专家错误或开放式 replacement。
- C50 的第一层 channel audit 只证明 luma-null payload 对 Huatuo/Hulu patch projector 可见，绝不证明其具有共享语义 codebook。

也就是说，C54 没有一个新的、尚未测试的病例信息源；它只是把已经不够通用的 XRV 信息换一种载体送给 VLM。

## 6. 是否值得做 30 分钟 CPU 实验？

**不值得作为候选晋级实验。** 即使实验成功，也只能证明 API 式 heatmap prompt 的另一种 blend 有用，无法修复公式级碰撞和 claim-identity 矛盾。

若未来只把它作为 baseline，可做如下小测试，但结果不得命名为新方法：

```text
n=64 VinDr clear images
expert = frozen XRV exact CAM
arms = base / visible heatmap overlay / luma-null CAM / spatially shuffled CAM
tower = frozen Huatuo CLIP only, CPU
metric = claim AUROC and true-CAM minus shuffled-CAM paired bootstrap
```

该测试只能回答“冻结视觉塔是否响应专家空间图”，不能证明 hallucination 缓解、开放生成、training-free semantic communication 或创新性。鉴于 API 已完成更完整的辅助热图实验，当前不运行。

## 7. 最终 fail-closed 判断

| 问题 | 判断 |
|---|---|
| 是否只是 conditional residual stacking | **是**：`U=H-E[H|M]` 正是 partialled-out expert residual |
| 是否只是 visual prompt / heatmap overlay | **是**：像素载体与 API 系统结构相同 |
| nullspace 是否创造新证据代数 | **否**：第一层等价于固定 side-input adapter |
| 是否有 training-free 可解释协议 | **否**：可解释可见码撞车；隐藏码无共享 codebook |
| 是否适配一次前向开放多 claim | **否**：二维 chroma 载荷无法保留任意 claim maps 的身份 |
| 是否值得运行致死实验 | **否**：数学与系统碰撞已足够致死 |

最终结论：

```text
NO-GO AS A NOVEL METHOD.
Do not spend GPU; do not promote a CPU win.
At most retain as an API/VEP-style baseline ablation.
```

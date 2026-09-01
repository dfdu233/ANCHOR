# C69 — Luma-Null CAM Payload 严格审计

审计日期：2026-08-13  
资源边界：只做公式、文献与既有 artifact 审计；未运行 GPU，未暂停 baseline。

## 裁决

> **严格 NO-GO：不能作为新的 ICLR 级方法进入实验队列；最多作为 auxiliary-heatmap visual prompting 的一个 blend ablation。**

候选为：对灰度医学图像 `Y`、claim `c` 的小模型 CAM `M_c`，构造

\[
X'_c=Y\mathbf 1+\alpha M_c v,\qquad w^\top v=0,
\]

再对冻结医学 VLM 做一次前向。`w` 是人为指定的线性亮度权重，`v` 是不改变该线性亮度的 RGB 色度方向。

这个候选与仓库已经完成的 [C54 Specialist-as-Encoder](./C54_SPECIALIST_AS_ENCODER_COLLISION.md) 是同一公式的简化版；C54 已逐式审计相同的 `specialist CAM -> luma-null RGB -> frozen VLM`。重新命名不能形成新候选。

## 1. Nullspace 只证明可编码，不证明 VLM 会正确解释

若 `w^T 1=1`、`w^T v=0` 且没有 clipping/rounding，则 `Y=w^T X'`，并且 `M_c` 可从 `X'` 与 `Y` 线性恢复。因此该公式确实是一个两信号到 RGB 的可逆编码。

但令冻结视觉塔首层 patch 投影为 `A`，载荷造成的 embedding 变化只是

\[
\Delta e=\alpha A(M_c\otimes v).
\]

亮度约束 `w^T v=0` 对 `A` 没有任何约束。存在两个立即成立的反例：

1. `A(I\otimes v)=0`：视觉塔完全忽略载荷，输出与原图相同，收益严格为零；
2. `A(I\otimes v)` 始终投向 `present` logit，且 `M_c\ge 0`：正负病例都被推向阳性，假阳性增加。

两者都满足亮度守恒。因此不存在从 `w^T v=0` 推出 hallucination risk 下降的定理。若额外假设

\[
\langle \nabla_X m_c, M_c v\rangle
\]

在阳性和阴性病例上都恰好具有正确符号，这个假设本身已经等于待验证的效果结论，而不是由编码结构导出的性质。

## 2. “亮度保持”不能穿过实际 VLM 预处理

无裁剪时的线性恒等式不等于真实管线中的影像不变性。要保持每个 RGB 通道在 `[0,1]`，`alpha` 必须满足所有像素和通道的联合边界；在 `Y=0` 或 `Y=1` 的饱和像素上，可用单侧容量可以直接降为零。

一旦进行 clipping，通常有

\[
w^\top\operatorname{clip}(X')\ne Y.
\]

8-bit rounding 会丢弃弱 CAM；逐通道标准化后载荷方向变为 `D^{-1}v`，也不再是视觉塔坐标中的亮度零方向。该方法最多保持一个预先选定的线性 luma functional，不能保证人眼等亮、医学信息不变或 encoder 表征不变。

## 3. 这是已有 heatmap visual prompt 的 blend 变体

系统结构已经被直接覆盖：

- **Attention Prompting on Image, ECCV 2024**：辅助模型根据文本 query 产生 attention heatmap，并直接写回原图后输入冻结 LVLM。候选只把乘法/可见 overlay 换成 luma-null blend。  
  https://arxiv.org/abs/2409.17143
- **Visual Evidence Prompting, ACL 2025**：小视觉专家先提取对象/关系证据，再把证据交给冻结 LVLM 以缓解 hallucination；“small specialist helps frozen generalist”已经是其核心命题。  
  https://aclanthology.org/2025.acl-long.205/
- **Black-Box Visual Prompt Engineering, NAACL 2025**：在图像上叠加 bbox/circle 等对象提示以减少 hallucination。  
  https://aclanthology.org/2025.naacl-short.45/
- **Visual Prompt Engineering for VLMs in Radiology, MIDL 2026**：将箭头、框、圆和轮廓直接嵌入放射影像，引导模型关注小病灶。  
  https://proceedings.mlr.press/v301/denner26a.html

若色度变化足够可见、且 VLM 能利用预训练中学到的“红色/彩色区域值得看”语义，它就是 visual prompt；若色度变化是隐蔽数值码，冻结 VLM 没有学过 `颜色方向 -> 病种/极性/置信度` 的共享 codebook。前者直接碰撞，后者没有 training-free 可解释通信协议。

## 4. CE 到 OE 的一次前向存在结构性矛盾

在 CE 中，问题已经给定 claim `c`，所以可以为一个 claim 生成一张 `M_c`。开放报告有 `C` 个潜在 findings，而固定一个亮度轴后，每个像素的 RGB 色度空间只有

\[
\dim\ker(w^\top)=2
\]

个自由度。任意线性编码 `C>2` 个重叠 claim maps 时 rank 至多为 2，必有至少 `C-2` 维信息丢失；当前单一 `v` 实际只传输一个加权和。

因此只能四选一：

1. 每个 claim 一张图、做 `C` 次 VLM 前向；
2. VLM 先草拟 claim，再生成 CAM、再做第二次前向；
3. 聚合所有 CAM，丢失病种身份，只保留普通 saliency；
4. 学习颜色 codebook/adapter，让 VLM 解码病种身份，但不再 training-free。

所以“一次 VLM 前向 + 开放多 claim + claim identity + training-free”不能由当前构造同时满足。它不具备所要求的通用性；天然彩色的内镜、皮肤、病理、眼底图像也不存在空闲 RGB 色度带宽。

## 5. 本地证据不支持载体替换能创造增量信息

已有正式结果已测同一个 XRV 小专家的信息上限：

- Huatuo：加入专家分数后条件 AUROC `+0.0598`；
- Hulu：仅 `+0.0102`，未过 `+0.02` 门，且 Brier CI 跨 0；
- one-bit veto：两模型都只移除约 `17.4%` FP，同时误伤 `1.52%/2.33%` TP，未满足 `20% FP / 1% TP` 门。

CAM 是同一专家分数的空间归因，而不是新的临床观测。改变通信载体不能凭空制造跨模型增量证据，也不能保证 CAM 指向病灶而非专家 shortcut。

## 6. L0/L1 决策

### 论文候选

**L0 公式/碰撞门已失败，因此不启动 L1 GPU。** 即使得到正数，也只能说明 API/VEP 风格的 auxiliary heatmap prompt 换一种颜色混合后有用，无法挽救方法新颖性、OE identity 或普适保证。

### 未来仅作为 baseline ablation 时

若论文需要补视觉提示 baseline，可在固定 64 例上比较：

```text
original
ordinary visible heatmap overlay
luma-null true CAM
energy-matched spatially shuffled CAM
wrong-claim CAM
uniform chroma shift
```

至少要求 true CAM 同时显著优于 visible overlay、shuffled、wrong-claim 与 uniform shift，并且无 Yes-rate/长度漂移。这个实验只能评价一种 visual-prompt 实现，不能重新命名为新方法。

## 最终结论

```text
NO-GO AS A NOVEL MITIGATION METHOD.
Exact duplicate of the already audited C54 family.
Do not spend GPU; retain only as an optional API/VEP baseline ablation.
```

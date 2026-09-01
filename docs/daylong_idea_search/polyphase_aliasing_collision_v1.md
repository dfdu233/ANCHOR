# Polyphase / Patch-Grid Aliasing 候选：碰撞审计与致死实验 v1

日期：2026-08-12  
范围：文献、代码配置和已有结果审计；**未运行 GPU，未修改或中断 baseline**。

## 结论先行

这个候选包含两个强度完全不同的命题：

1. **方法命题：对图像做若干个小于一个 patch 的平移，再平均 VLM 输出。** 该命题与
   [Phase Marginalization for Patch-Grid Instability in Vision Transformers](https://arxiv.org/abs/2606.08132)
   的 training-free 方法发生直接碰撞；在没有空间输出需要反向对齐的 VLM claim 分类中，
   它在数学上就是 structured shift TTA。将其用于医学 VLM 只是任务迁移，**不能作为 ICLR
   新方法**。
2. **机制命题：医学小病灶错误是否由 patch-grid phase 决定，并呈现“半 patch 位移比整
   patch 位移更破坏证据”的周期性。** 这个更窄的命题尚未被本地结果检验。它与普通风格
   不稳定、局部 score 不可读是不同假设，值得一次 Huatuo 真实模型致死实验。

最终判定是：**作为算法，当前版本直接碰撞并应关闭；作为机制致死实验，保留一次低成本
条件机会。** 即使机制通过，论文也应研究“病灶尺度 / tokenization phase 的错误边界”，
不能把 phase averaging 重新命名成贡献。

## 1. 为什么会从本地结果想到它

### 1.1 支持继续问“视觉前端是否丢失小病灶”的事实

- VinDr 的两个独立 panel 都显示：在同一 finding 内，医生框越小，模型正确 claim margin
  越低。fresh panel 的 Spearman 为 Huatuo `0.323 [0.158, 0.462]`、Hulu
  `0.415 [0.222, 0.559]`。
- 两模型最小病灶四分位的联合 miss rate 为 `26.7%`，最大病灶四分位仅 `4.2%`；这支持
  “小病灶是共享困难”，但尚未解释困难发生在哪一层。
- 两个模型的视觉 tokenizer 都以 `14×14` 像素为 patch 单位：Huatuo 的本地配置明确为
  CLIP ViT-L/14、输入 `336×336`；Hulu 的本地 vision encoder 也记录 `patch_size=14`。

这些事实允许提出一个具体问题：同一个小病灶落在 patch 中央和落在 patch 边界时，视觉
token 是否不同到足以改变临床判断？

### 1.2 已有负结果为什么没有直接否定它

- 普通 style 变换的 flip rate 只有 `3.13% / 2.34% / 1.56%`，全部低于冻结的 5% 门；
  style drift 预测错误 AUROC `0.425–0.446`，显著不如原始 margin `0.798`。这关闭的是
  photometric/style 主线，不等价于改变 patch 边界的整数像素相位。
- Huatuo 的 supervised multiscale patch scan 在 fresh 266 claims 上仅比强基线增加
  `0.00396` macro AUROC，95% CI `[-0.0194, 0.0273]`，未过门；evidence-conserving
  mixture 也未超过 final margin。它们测试“固定 token 网格内能否读出局部证据”，没有改变
  token 网格本身。
- layer convex mixture 失败关闭的是 decoder 层间恢复，不是 vision tokenizer 的输入相位。

因此该候选没有与本地负结果逻辑冲突。但它的先验概率也不能被高估：style 现象弱、局部
scan 无增量，都意味着模型可能只是对这些样本整体缺乏视觉证据，而不是恰好选错 patch 相位。

## 2. 数学背景：什么叫 patch-grid phase

Huatuo 把预处理后的 `336×336` 图像切成 `24×24` 个非重叠 `14×14` patch。设 patch 大小
为 `P=14`，网格起点偏移为

\[
\phi=(d_x,d_y),\qquad d_x,d_y\in\{0,\ldots,P-1\}.
\]

同一个病灶不移动临床语义，但改变 \(\phi\) 会改变哪些像素被分到同一个 token。若病灶小于
一个 patch，它可能在某个相位完整落入一个 token，在另一个相位被切到两个或四个 token。
这就是本候选所说的 **subpatch phase**；它不是亮度、颜色或 DICOM window 风格。

对冻结模型的 claim margin 记为 \(m_\phi(x,c)\)。最简单的 phase averaging 是

\[
\bar m(x,c)=\frac{1}{K}\sum_{\phi\in\Phi_K}m_\phi(x,c).
\]

必要的背景是：这其实是经典的 group averaging / Reynolds operator。若 \(\Phi\) 是一个
封闭的平移群，则对其中任意平移 \(h\)，完整平均满足

\[
\bar m(hx,c)=\bar m(x,c).
\]

证明只是把求和索引从 \(\phi\) 换成 \(\phi h\)。所以“平均多个相位得到平移不变量”不是
新定理。若各相位误差方差为 \(\sigma^2\)、两两相关为 \(\rho\)，平均后的方差为

\[
\operatorname{Var}(\bar\epsilon)
=\frac{\sigma^2}{K}\bigl[1+(K-1)\rho\bigr],
\]

这也只是经典 ensemble 方差公式；当不同相位高度相关时，增加 forward 几乎没有收益。

另一个重要限制是：对 VLM 的全局 claim margin，没有 segmentation map 需要 inverse alignment。
因此

\[
\text{phase marginalization} = \text{对结构化平移视图做 TTA 平均}.
\]

若改为取最佳相位 \(\max_\phi m_\phi\)，又会直接回到本项目已经发现的 visual search tax / 
winner's curse：相位越多，正常图像上最大的随机阳性响应也越高。因此致死实验禁止把 oracle
max 当作可部署方法。

## 3. 文献碰撞

| 工作 | 已覆盖内容 | 对本候选的裁决 |
|---|---|---|
| [Blending Anti-Aliasing into Vision Transformer, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/2b3bf3eee2475e03885a110e9acaab61-Abstract.html) | 将 ViT 不连续 patch tokenization 解释为 aliasing，并用架构/训练修改减轻 | “ViT patchification 有 aliasing”不是新观察 |
| [Vision Transformer for Small-Size Datasets / Shifted Patch Tokenization](https://arxiv.org/abs/2112.13492) | 将半 patch 多方向平移后的图像拼接后再 tokenization，增强 locality | “用半 patch shifts 丰富 token”已被覆盖，但需要训练/改结构 |
| [Reviving Shift Equivariance in Vision Transformers](https://arxiv.org/abs/2306.07470) | adaptive polyphase anchoring；修复 patch embedding、window/global subsampled attention 的 shift equivariance | polyphase 机制与最大范数相位选择已有完整公式和理论 |
| [Making Vision Transformers Truly Shift-Equivariant, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Rojas-Gomez_Making_Vision_Transformers_Truly_Shift-Equivariant_CVPR_2024_paper.pdf) | 对 tokenization、attention、patch merging、position encoding 做 adaptive 设计，四类 ViT 达到 circular-shift 100% consistency | “使 ViT shift-equivariant”已被顶会直接覆盖，且方法更系统 |
| [Phase Marginalization for Patch-Grid Instability, 2026 preprint](https://arxiv.org/abs/2606.08132) | frozen ViT 上取 `P/2` 的四个结构化 phase，inverse-align 后平均；对照 compute-matched random/integer shift TTA | **与 proposed training-free 多相位平均直接碰撞**；该文尚是 preprint，但足以否定新颖性 |
| [SECOND, ICML 2025](https://proceedings.mlr.press/v267/park25c.html) | 熵驱动的局部多尺度 token 选择与 contrastive decoding | 不能把相位方法包装成“更精细的局部搜索” |
| [Seeing the Trees for the Forest, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Huy_Seeing_the_Trees_for_the_Forest_Rethinking_Weakly-Supervised_Medical_Visual_ICCV_2025_paper.html) | 医学局部病灶被背景/global token 稀释 | “医学小病灶需要局部信息”也已有直接邻近工作 |

检索中未发现同行评审工作同时研究“医学 VLM 临床 claim 错误、病灶相对 patch 尺度、以及
modulo-patch 周期性”。这个剩余缝隙是**机制边界**，不是 phase-averaging 算法。

## 4. 为什么不能把普通 TTA 成功误判成 phase aliasing

仅发现 \(m(T_7x,c)\ne m(x,c)\) 不够，因为任何缺乏平移不变性的模型都会如此。真正区分
phase aliasing 的预测是 **modulo-patch recurrence**：

- 平移半个 patch（7 pixels）改变 patch membership；
- 平移一个完整 patch（14 pixels）恢复相同网格相位，只把内容移动一个 token；
- 若 patch 边界是主要原因，则更小的 7-pixel shift 反而应比 14-pixel shift 更破坏 margin；
- 若只是普通位置敏感、边界填充或风格不稳，扰动通常随位移幅度平滑变化，没有理由在
  14 pixels 处恢复。

对每张图定义相位反转量

\[
A_i=\frac14\sum_{a\in\{x,y\}}\sum_{s\in\{-1,+1\}}
\left(
|m(T_{sP/2,a}x_i,c_i)-m(x_i,c_i)|
-|m(T_{sP,a}x_i,c_i)-m(x_i,c_i)|
\right).
\]

其中 `T` 在**模型原生 resize/crop 之后**用 reflect padding 和整数 crop 实现，不做第二次
插值，也不改变亮度。`A_i>0` 表示半 patch 位移比幅度更大的整 patch 位移更破坏输出；这才
是相位机制的独特预测。该统计量和公式是本实验的鉴别设计，不应冒充通用新理论。

## 5. 最小真实模型致死实验

### 5.1 目的与预算

先只跑 Huatuo，不碰当前 baseline 队列。使用现有 VinDr DICOM、reader vote、bbox 和原生
trinary claim scorer；预计 `192 images × 9 phases = 1,728` 次单步 scoring。该规模只用于
判死，不产生正式论文结论。Huatuo 失败则不跑 Hulu。

### 5.2 冻结样本

- `48` 个 3/3 reader-positive、bbox 面积最低四分位的 claims；
- `48` 个 3/3 reader-positive、bbox 面积最高四分位的 claims；
- `96` 个同 finding 匹配的 0/3 reader-negative claims；
- patient/image-disjoint，七个已合格 findings 尽量均衡；一个图只保留一个 primary claim，
  防止同图伪重复；
- 不在这些样本上选择 offset、阈值或 aggregation rule。

### 5.3 冻结图像干预

原生 processor 先产生模型实际使用的 pixel tensor，再施加九个无插值平移：

```text
(0,0), (±7,0), (±14,0), (0,±7), (0,±14)
```

padding 固定为 reflection；另保存边缘变化像素比例和 pixel L1，确认 `7` 的低层改动小于
`14`。prompt、tokenizer、answer verbalizer 和所有生成设置完全固定。

### 5.4 三个分层 gate

#### Gate M1：确有 modulo-patch recurrence

- `A` 在 small-positive 组均值的 image-bootstrap 95% CI 下界 `>0`；
- `A_small - A_large` 的 95% CI 下界 `>0`；
- 同 finding 回归中，`A` 随 resized lesion area / `P²` 减小而增大，系数 CI 排除 0；
- negative 组不得出现同量级的周期结构，否则更像全图位置/边缘 artifact。

任一条件失败：**NO-GO，关闭 phase-aliasing 机制**。只出现若干 label flips 不算通过。

#### Gate M2：不是已知的普通低-margin 不稳定

用 development-only 拟合两种错误预测器，在冻结 confirmation 比较：

```text
base: finding + original final margin
enhanced: base + A + half-phase variance
```

enhanced 相对 base 必须增加至少 `0.02` macro AUROC，paired image-bootstrap 95% CI 下界
`>0`，NLL improvement CI 下界也 `>0`。否则即使 M1 存在，它也只是架构微扰，不解释临床
错误。

#### Gate H：existing phase averaging 是否真的缓解错误

这里只评估现有技术的可用性，不主张新方法：

- structured `K=5`：原图加四个 `P/2` 轴向相位，平均 claim logits；
- compute-matched generic shift TTA：五个预先固定的非结构化 integer offsets；
- K=1 原始模型；
- oracle best phase 只报告 headroom，绝不列为方法。

structured K=5 必须同时满足：small-positive recall `+5pp`、总体 BAcc `+2pp` 且 paired CI
下界 `>0`、negative FP 增幅不超过 `1pp`、并比 compute-matched generic TTA 至少高 `1pp`
且 CI 下界 `>0`。否则结论只能是“相位敏感”，不能声称 hallucination mitigation。

## 6. 结果分支与最高可写结论

| 结果 | 科学解释 | 后续动作 |
|---|---|---|
| M1 FAIL | 小病灶面积效应不是 patch-grid phase 周期造成 | 永久关闭；不跑 Hulu，不开发算法 |
| M1 PASS、M2 FAIL | 存在 tokenizer phase artifact，但不比 final margin 更能解释临床错误 | 只作架构诊断附录；不做 mitigation |
| M1/M2 PASS、H FAIL | 找到新的医学错误机制，但现成 phase averaging 不能安全纠错 | 可研究“lesion-to-patch ratio 的错误边界”；方法仍未完成 |
| M1/M2/H 均 PASS | 机制和实用价值成立 | 在 Hulu、LLaVA-Med、通用 VLM 及非胸片小病灶任务复现；phase averaging 只能作 baseline |

即使全部通过，单独结果仍**不足以完成 ICLR oral 方法论文**，因为方法与 2026 Phase
Marginalization 直接碰撞。较有价值的潜在论文方向只能是一个跨模型的无量纲边界：

\[
\lambda=\frac{\text{resized lesion extent}}{P^2},
\qquad
\text{phase sensitivity}=\operatorname{Var}_{\phi}[m_\phi(x,c)],
\]

并证明错误率、phase sensitivity 和 mitigation headroom 在不同模型/分辨率下由 `λ` 统一，而
不是由病种名或原始像素面积决定。这会像 scaling-law / phase-boundary 论文，而不是“医学上
用了 TTA”。当前尚无数据支持这条 collapse，不能预先声称。

## 7. 工程接口与可复现边界

- Huatuo vision config：`/home/dbw/models/HuatuoGPT-Vision-7B/vit/clip_vit_large_patch14_336/config.json`
  (`image_size=336`, `patch_size=14`)。
- Hulu config：`/home/dbw/models/Hulu-Med-4B/config.json` (`patch_size=14`)；因其动态视觉
  token 压缩，只有 Huatuo M1/M2 通过后才适配，且必须在 patchification 前、原生 resize 后
  注入 shift。
- 使用现有 claim margin scorer，不生成长文本；真实方法放量前才进入 OE/report fixed-K。
- 所有 shift tensor、hash、offset、pixel difference、margin 和 bootstrap cluster ID 必须落盘。
- 不复用 test 选择 phase；不使用最大相位输出；不把 flip rate、oracle 或短回答当改善。

## 最终 verdict

- **现象依据：** 中等；小病灶面积—margin 关系已跨模型复现。
- **机制新颖性：** 中等偏低；医学 VLM 的 modulo-patch 临床错误边界尚有缝隙，但 ViT
  shift/aliasing 理论高度成熟。
- **方法新颖性：** 低；structured phase averaging 与现有 Phase Marginalization / TTA 直接碰撞。
- **执行性：** 高；单模型约 1,728 次单步 forward，无需训练或新数据。
- **决策：** 只批准一次 Huatuo 致死实验；**不批准注册新算法名或将其升为 ICLR 主线**。


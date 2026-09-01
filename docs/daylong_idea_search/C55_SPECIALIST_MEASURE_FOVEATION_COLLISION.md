# C55 Specialist-Measure Foveation：公式级碰撞审计

审计日期：2026-08-13  
范围：specialist 只产生非负空间测度；以 OT / retinal warp 在固定输出像素预算内重分配采样密度；保持全图一一映射；一次冻结 VLM 前向。未运行 GPU，未修改 baseline。

## 裁决

> **严格 NO-GO：这是已知 saliency-guided nonuniform sampling / foveated warping 的医学专家实例，不存在不可约的新算法 delta。**

它与 SECOND/ARCD 的执行位置不同，但与更早、更直接的视觉采样文献精确同构；仓库 C33 也已对同一候选做过正向 CPU substrate 和公式级关闭。把密度图来源从普通 saliency network 换成 frozen medical specialist、把 warp 求解器换成 optimal transport，都只是 source/solver substitution。

## 1. 冻结候选与数学还原

令 specialist 对 claim `c` 输出非负测度

```text
mu_c(u) >= 0,       integral_Omega mu_c(u) du = 1.
```

为了防止背景完全坍缩，目标采样密度通常写成

```text
rho_c(u) = (1-beta)/|Omega| + beta mu_c(u),     0 <= beta < 1.
```

构造一个可逆映射 `T_c : Omega -> Omega`，使均匀输出网格的 push-forward 等于 `rho_c`。若使用二次代价 OT，它是

```text
T_c = argmin_T integral ||T(u)-u||^2 du
      subject to  T_c#Uniform = rho_c.
```

最终输入是

```text
X_c(u) = x(T_c(u)),
```

再统一采样到 VLM 的固定 `H×W` 网格。`mu` 高的区域占用更多输出像素，背景被压缩但未删除。

这正是 saliency-guided sampling layer / foveated rendering 的标准连续形式。是否用 Brenier map、CDF、核密度重心、piecewise bilinear grid 或 `grid_sample` 只改变数值实现，不改变算法对象。

## 2. 精确碰撞

### 2.1 Learning to Zoom 已完整覆盖核心操作

ECCV 2018 **Learning to Zoom: a Saliency-Based Sampling Layer for Neural Networks** 已提出：根据 saliency map 非均匀采样高分辨率输入，在固定低分辨率预算下放大 salient regions、压缩其余区域，以提高任务网络性能；其输出就是 distorted/caricature-like whole image，并在 saliency 不确定时退化为 uniform sampling。  
https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Adria_Recasens_Learning_to_Zoom_ECCV_2018_paper.php

C55 的全部主要性质均已出现：

| C55 | Learning to Zoom |
|---|---|
| 非负空间测度 `mu` | saliency map |
| 固定像素预算 | task network fixed input size |
| salient 区域占用更多像素 | learned saliency sampler |
| 保留整幅图而非 crop | distorted whole-image sample |
| 不确定时接近 uniform | graceful uniform fallback |

specialist 提供 `mu` 而非学习 saliency network，不产生新的 warp 原语。

### 2.2 LZU 又直接覆盖可逆 warp/unwarp

CVPR 2023 **Learning to Zoom and Unzoom (LZU)** 明确使用 invertible、piecewise bilinear mapping：先 zoom salient region，再 unzoom spatial outputs，目标是 model/task-agnostic spatial attention 与固定成本的高分辨率处理。  
https://tchittesh.github.io/lzu/

所以“一一映射/保持全图/可逆”也不是 C55 新性质；它是 LZU 为修复检测/分割坐标变形而引入的标准约束。

### 2.3 OT solver 已被 saliency compression 直接采用

2025 Pattern Recognition **Image compression using optimal transport mapping based on ranking visual saliency** 使用 OT 将重要区域放大、低重要区域缩小，并用 inverse OT 恢复图像。  
https://doi.org/10.1016/j.patcog.2025.112201

因此从 kernel warp 改成 OT 不能作为数学创新：OT 在这里仍只负责实现给定采样密度的质量搬运。

### 2.4 DG 与医学换域也已有系统近邻

- WACV 2025 **Instance-Warp**：用 instance saliency 在图内 oversample objects、undersample background，以降低背景依赖并改善 UDA；这直接覆盖“把 DG/specialist prior 变成 warp”的跨域故事。  
  https://openaccess.thecvf.com/content/WACV2025/papers/Zheng_Instance-Warp_Saliency_Guided_Image_Warping_for_Unsupervised_Domain_Adaptation_WACV_2025_paper.pdf
- 医学图像已有 lesion-emphasis/foveation：例如 fundus ALES 根据 CNN lesion saliency 做空间变化的局部增强；乳腺 FAP 使用 gradient-guided fixation 与 foveated preprocessing。医学 specialist measure 不是一个新 setting。  
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5443420/

## 3. 与 SECOND、ARCD 和现有 VLM mitigation 的关系

### 不与 SECOND 完全同式

SECOND 根据跨注意力熵选择/遮蔽 patches，并用 coarse-to-fine 的多分支 contrastive decoding；C55 在输入前改变连续采样密度，只做一次 VLM 前向。因此它不是 SECOND 的 exact formula duplicate。

但该差异不足以构成新颖性，因为 C55 的精确前身不是 SECOND，而是 Learning to Zoom / LZU。它只是把一个成熟视觉采样模块用于 hallucination。

### 不与 ARCD 完全同式

ARCD 用 anatomical mask 在 token、attention、logit 三层做区域引导 contrastive decoding；C55 不改 decoder，也不使用负分支。因此它不是 ARCD exact duplicate。

但两者都依赖外部区域先验；C55 的“单前向”效率只是 foveated preprocessing 的既有优势，不是新的机制。

### 与 HALC / visual prompting 的系统重叠

HALC 由 grounding detector 为当前 token 选择多个 field-of-view，再做 focal-contrast decoding。C55 把多 FOV 合成一张连续 warp，可减少前向数，但“根据 grounding 分配更多像素给局部”仍在 focal visual context 邻域。Attention Prompting on Image / BBVPE 则说明 auxiliary region cue 修改输入是成熟的 visual prompting 路径。

## 4. 仓库已有直接 NO-GO 证据

本地候选注册表 C33 **Detail-Budget Warping** 已经定义了同一问题：

> 固定视觉 token 预算下，按局部细节密度可逆重采样全图。

其 CPU L0 在 `800` 图、`4345` bbox 上证明 10 类病灶均被 density map 富集，是正 substrate；但公式审计因 ECCV 2018 Learning to Zoom 与 CVPR 2023 LZU 精确碰撞而关闭，没有启动 GPU。

C55 只是把 C33 的 label-blind detail density 替换为 specialist `mu_c`：

- 若 `mu_c` 是 CAM/heatmap，进一步与 C54 auxiliary heatmap prompting 相邻；
- 若 `mu_c` 不带 claim identity，无法覆盖开放式多 finding；
- 若每个 claim 各有 `mu_c`，需要先知道 claim 或运行多个 warp/前向；
- 若将所有 `mu_c` 聚合，专家常见病/高幅值 finding 会抢占固定像素预算。

所以 specialist 并未修复 C33 的创新性问题，反而带来开放式 claim-identity 问题。

## 5. 非平凡数学性质也不成立

“一一映射”不等于信息无损。连续 `T` 可逆，但最终仍在固定 `H×W` 网格取样：

```text
x -> x o T -> Sample_HxW(x o T).
```

最后一步必然丢信息；高 `mu` 区域的 aliasing 降低，以背景更严重的 aliasing 为代价。不存在对所有 findings 同时提高 Bayes 信息的保证。

对多个 claim 的测度 `mu_1,...,mu_C`，任何单一融合密度

```text
rho = F(mu_1,...,mu_C)
```

都必须在固定总质量 `integral rho=1` 下竞争。提高一个区域的像素预算必然降低别处预算。这是资源分配约束，不是 hallucination 单边改善定理；在正常片上专家的伪 saliency 还会主动放大 shortcut。

OT 的 Brenier 唯一性只说明在给定源/目标测度下存在最小二次位移 map，不说明该 map 最大化 VLM 临床信息或最小化 hallucination。因此不能把 OT 唯一性包装成方法理论贡献。

## 6. 是否运行 CPU / GPU？

不运行。理由不是预期效果差，而是：

1. C33 的 bbox enrichment CPU 门已经通过，重复证明 specialist saliency 会把病灶放大没有信息价值；
2. 即使 32 例 GPU 提升，也只能得到“known foveation 对某医学 VLM 有效”的应用结果；
3. 无法越过公式级 collision gate，违背当前 goal 的创新优先约束；
4. claim-specific warp 对开放生成还要求额外 proposal/多前向，破坏“一次前向、通用 OE”卖点。

若未来作为 baseline，可比较 uniform resize、crop、Learning-to-Zoom warp、LZU warp、SECOND 和 ARCD；但不得命名为新算法。

## 7. Fail-closed 总结

| 门 | 判断 |
|---|---|
| 固定预算、saliency 引导非均匀采样 | **已被 Learning to Zoom 覆盖** |
| 可逆全图 warp/unwarp | **已被 LZU 覆盖** |
| OT 实现显著性 warp | **已有直接 compression 先例** |
| DG 结合 | **Instance-Warp 已覆盖** |
| 与 SECOND/ARCD 是否 exact collision | 不是 exact，但更早的视觉采样工作已 exact collision |
| 本地 substrate | C33 已正向验证，无需重复 |
| 开放多 claim 一次前向 | **未解决 claim identity 与预算竞争** |
| 是否值得 GPU | **否** |

最终裁决：

```text
NO-GO AS A NOVEL METHOD.
Known foveation with a medical specialist as the saliency source.
Do not run CPU/GPU; retain only as a future baseline.
```

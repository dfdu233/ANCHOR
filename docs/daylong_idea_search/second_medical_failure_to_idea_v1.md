# 从 SECOND 类方法在医学场景的边界，到一个仍然合法的研究缝隙

日期：2026-08-12  
范围：只做文献、官方代码和已有输出审计；未占用 GPU，未修改或中断 baseline。

## 结论

不能把当前结果写成“SECOND 在医学 VLM 上失败”。更准确的结论是：

1. **SECOND 本身尚未得到可比较的医学效果。** 官方递归路径在
   LLaVA-Med-v1.5-Mistral 上依赖不存在的 `CLIPVisionTower.image_attentions`，而关闭
   SECOND 时其 fork 与 canonical backend 也只达到 `29/32=0.90625` normalized exact 和
   `0.96734` token-F1，未过事先冻结的 `0.95/0.98` 一致性门。它当前是
   `N/A: architecture/backend incompatibility`，不是负效果。
2. **真正被本地实验反复否定的是一个更一般的假设：高响应区域等于正确证据。**
   医生框删除没有降低 Nodule/Mass 分数，反而平均提高 `0.084`；把病灶纹理搬到对侧后，
   分数比原图平均高 `0.291`。因此“选到一个让模型更兴奋的 patch”不能证明选到了病灶。
3. **局部证据稀释却是真现象。** 在两组独立 VinDr 3/3 reader-positive 样本中，病灶框
   越小，模型正确 claim margin 越低；fresh 组的 within-finding Spearman 为 Huatuo
   `0.323 [0.158,0.462]`、Hulu `0.415 [0.222,0.559]`。问题存在，但现有“选择后放大”
   解法会把噪声也一起放大。
4. 在核对 2025--2026 的最近工作后，**唯一仍值得一次致死实验的缝隙，不是更好的 mask，
   而是校正“从很多区域里挑最大响应”本身引入的视觉搜索税**。

建议把唯一保留候选暂称为 **Search-Calibrated Visual Decoding (SCVD)**，但在致死实验
通过前不注册方法名、不立论文主线。

## 1. SECOND 和 SPIN 实际做了什么

### 1.1 SECOND

[SECOND（ICML 2025）](https://proceedings.mlr.press/v267/park25c.html)先以当前生成 token
对图像 patch 的注意力选出高响应区域，再逐步提高分辨率；不同尺度的 logits 被逐级做
contrastive decoding。其项目页给出的核心设计是：注意力熵越高，下一尺度保留的 patch
比例越大，随后用细尺度减粗尺度来强化新增视觉信息。官方代码仓库为
[AIDASLab/SECOND](https://github.com/AIDASLab/SECOND)，本地审计 commit 为
`4ad65872d9c03ea7b60ea68c2b663d22a373ec33`。

这个设计隐含两个假设：

- 注意力排序能找到真正相关的区域；
- 被选区域在更细尺度产生的额外 logit 是证据，而不是由选择、裁剪或编辑产生的响应。

本地目前只能否定第二个假设的朴素版本，不能评价 SECOND 完整算法的医学效果。

### 1.2 SPIN

[SPIN（EMNLP 2025）](https://aclanthology.org/2025.emnlp-main.631/)在每个生成 token 上，
保留对图像 token 响应最高的 top-k attention heads，压低其余 heads。官方
[代码](https://github.com/YUECHE77/SPIN) commit
`a6ddf4a3f583af9192175da40d99990324341955` 中，`attentionSPIN.py` 对每个 head 汇总
image-range 的 pre-softmax attention score，标准化后 top-k 路由，再把未入选 head 输出乘
`small_num_mask`。所以它选择的是“看图更多的 head”，并没有独立检查“这个 head 看到了什么”。

这一点已经被 [Same Attention, Different Truths（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Same_Attention_Different_Truths_Put_Logit-Lens_over_Visual_Attention_to_CVPR_2026_paper.html)
直接挑战：真实对象和幻觉对象可以得到相近的视觉注意力，关键是高注意区域是否真的能解码出
对应语义。该工作已经进一步使用 logit-lens 检查高注意区域，并按 visual uncertainty 与
contextual prior 选择不同干预。因此“attention amount 不是 evidence”不能再作为我们的新贡献。

## 2. 医学场景为何暴露出结构性问题

### 2.1 小而弱的病灶确实会被全局表征稀释

[Seeing the Trees for the Forest（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Huy_Seeing_the_Trees_for_the_Forest_Rethinking_Weakly-Supervised_Medical_Visual_ICCV_2025_paper.html)
发现胸片背景 token 范数高、局部病灶 token 与全局图像 token 对齐不足，并以 disease-aware
feature prompting 强化病灶区域。本地 VinDr 结果也支持“小框对应较低 claim margin”，所以
SECOND 的动机在医学场景并不荒谬。

但“病灶小”只说明平均池化会稀释信号，不说明最大的局部响应就是真病灶。

### 2.2 医学图像更像弱信号搜索，而不是普通物体定位

COCO 中的 `dog` 往往有清晰边界和高对比；胸片上的小结节、轻度积液、纤维化或肺纹理改变
可能是低对比、弥漫、与正常结构重叠的 `stuff`。因此局部方法面对的是：

> 在很多高度相似的区域里搜索一个很弱的异常，而不是从背景中找一个显眼物体。

搜索区域越多、尺度越细，即使图像完全正常，也越容易偶然出现一个很高的响应。SECOND、
VGA、ARCD 等方法主要研究如何选得更准或如何放大所选区域，没有显式支付“找过多少地方”
带来的统计代价。

### 2.3 阴性结论没有单个可指认的病灶

“有一个结节”只需找到一个可靠 witness；“没有气胸”则需要足够覆盖整个胸膜空间。
top-k patch 天然适合前一种存在性搜索，却无法单靠一个高响应 patch 证明后一种全局阴性。
这解释了局部增强方法为何容易改变 Yes/No 工作点，而不一定同时降低 FP 与 FN。不过把这个
现象做成 noisy-OR/quantifier-aware aggregator 与经典 multiple-instance learning 太接近，
单独不足以成为 ICLR 方法。

### 2.4 选择与证明使用同一个噪声信号

如果一个方法先用响应选出最大 patch，再用同一响应证明该 patch 重要，就会出现
**winner's curse**：被选中的区域一部分之所以最大，只是噪声恰好最大。本地 relocation
实验给出了直接警告——把真实病灶纹理搬到不真实的位置仍能令分数显著高于原图；如果继续
枚举更多位置并只报告最大值，伪证据只会更多。

## 3. 完整输出给出的实际证据

LLaVA-Med-v1.5-Mistral-7B 在 MedHEval CXR-VisHal 的 3,669 个二元问题上，不同干预主要
改变输出工作点，而不是形成稳定一致的临床修正：

| 方法 | Yes / all | No / all | Invalid / all | strict BAcc |
|---|---:|---:|---:|---:|
| VISTA | 98.09% | 0.76% | 1.14% | 0.5000 |
| DoLa | 97.49% | 0.74% | 1.77% | 0.4980 |
| OPERA | 87.52% | 3.19% | 9.29% | 0.5078 |
| VCD | 72.66% | 6.43% | 20.90% | 0.4471 |
| AvisC | 79.80% | 10.55% | 9.65% | 0.5314 |

这不是 SECOND 结果，但它展示了同族 training-free 干预在医学 CE 上的共同风险：减少
affirmative bias、增加局部/视觉影响和生成可解析性纠缠在一起。

真实样本 `cxr-vishal-0` 的真值为 No，问题是“大量胸腔积液是否存在”：VCD 输出
`No, ... does not show any signs of a large pleural effusion.`，VISTA 则输出
`Yes, ... shows signs of a large pleural effusion.`。反过来，在真值为 Yes 的
`cxr-vishal-4`，VCD 给出不以 Yes/No 开头的解释而被严格判 invalid，VISTA 输出 Yes。
因此只看官方 `else -> Yes` proxy 会把方法比较和严格临床决策混在一起。

在 VCD 与 VISTA 的 972 个不同预测中，VCD 正确而 VISTA 错误有 185 个，VISTA 正确而
VCD 错误有 353 个；后者有 298 个只是 VCD `invalid` 变 VISTA `yes`。这再次说明“输出变了”
不能直接解释为“视觉证据被恢复”。

## 4. 与最近工作的碰撞矩阵

| 工作 | 已经覆盖 | 对新方向的约束 |
|---|---|---|
| [SECOND, ICML 2025](https://proceedings.mlr.press/v267/park25c.html) | 熵驱动 patch 选择、多尺度递归、逐尺度 contrast | 不能再主张“精细 mask / 多尺度对比” |
| [SPIN, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.631/) | token-wise image-attention head routing | 不能再主张“只保留看图 head” |
| [Trees for the Forest, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Huy_Seeing_the_Trees_for_the_Forest_Rethinking_Weakly-Supervised_Medical_Visual_ICCV_2025_paper.html) | 医学小病灶被背景和 global token 稀释；disease-aware feature prompting | “医学病灶小，所以增强局部”已被直接覆盖 |
| [ARCD, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37620) | 医学 anatomy mask；token/attention/logit 三层 region-guided contrast | “小模型/MedSAM 给 mask 再解码”已被直接覆盖 |
| [VGA, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Tell_Model_Where_to_Look_Mitigating_Hallucinations_in_MLLMs_by_CVPR_2026_paper.html) | 用 visual-token semantics 构造 grounding，再导引 attention | “由视觉 token 找区域再增强”已被直接覆盖 |
| [Same Attention, Different Truths, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Same_Attention_Different_Truths_Put_Logit-Lens_over_Visual_Attention_to_CVPR_2026_paper.html) | 检查高注意区域的语义，不只看 attention amount；分类干预 | “attention 不是 evidence / attention+语义验证”也已覆盖 |
| [BCEA, 2026 preprint](https://arxiv.org/abs/2606.16667) | answer/abstain/acquire，按 claim crop/zoom，并对 acquisition 后 score 重做 conformal calibration | 不能再主张“低置信时主动 crop”；新工作必须区别于 risk-coverage guarantee |

**剩余 delta：** 上述工作没有把“在 claim × region × scale 候选集中取最大值”本身视为
multiple testing，并研究其 search-size scaling law。这里只能说在本次检索下未发现机制等价
工作，不能声称绝对首创。

## 5. 唯一保留候选：Search-Calibrated Visual Decoding

### 一句话

> 一个局部区域只有在其响应超过“搜索这么多区域本来就会出现的最大假信号”之后，才有资格
> 被用来增强 claim。

### 必要的数学背景

假设正常图像的每个候选区域都有一个标准化噪声分数 `z_j`，共搜索 `M` 个区域。即使每个
`z_j` 的均值都是 0，最大值也会随 `M` 增长；对近似独立的标准高斯噪声，典型量级约为

\[
\max_{j\le M} z_j \approx \sqrt{2\log M}.
\]

所以提高分辨率、增加尺度或枚举更多 crops，会自动制造更大的“最佳局部证据”。这是搜索
过程的统计结果，不是模型变得更会看图。

SCVD 不直接使用最大响应，而使用超过 null-search distribution 的剩余量：

\[
T_c=\max_{(r,s)\in\mathcal R} z_c(r,s)-q_{1-\alpha}^{\text{null}}(|\mathcal R|),
\]

其中 `r,s` 表示区域与尺度，`q` 是相同搜索规模下、无该 finding 图像或保结构 permutation
得到的最大值分位数。最终只在 `T_c>0` 时把它加入原始 claim margin。OE/report 还要把
claim 数加入搜索族，避免“扫描更多疾病名”自动增加 fabricated claim。

标准的 sub-Gaussian union bound 可以给出

\[
P\!\left(\max_j z_j>\sqrt{2\log(M/\alpha)}\right)\le\alpha.
\]

这条不等式是现成数学工具，不能冒充理论贡献。真正可能形成 ICLR 贡献的是：证明视觉搜索
存在跨模型的 phase law，并证明校正搜索税后可以同时恢复小病灶、抑制 fabricated local
evidence，而不是把 FP 换成 FN。

### 为什么它与现有工作不同

- SECOND/VGA/ARCD/SPIN 问“该选哪个区域或 head”；SCVD 问“在选过这么多候选后，这个赢家
  还有多少可信度”。
- SADT 用另一种语义读出验证所选区域；SCVD 校正任何读出在 adaptive search 后都会有的
  选择偏差，两者可正交。
- BCEA 保证 acquisition policy 之后的最终 risk-coverage；SCVD 的对象是区域搜索统计量，
  目标是在不默认 abstain 的情况下得到 search-size-comparable evidence。正式论文必须直接
  对照 BCEA，不能只靠措辞区分。

### 小于 30 分钟的致死实验

利用已经排队生成、但不额外占 GPU 的 Huatuo/Hulu patch-score cache：

1. 固定 7 个 VinDr findings、清晰 `0/3` 与 `3/3`、image-disjoint dev/test。
2. 依次扩大搜索族 `M`：区域数、窗口尺度数、claim 数；记录每张图的 raw maximum。
3. 只在 dev 的 `0/3` 图上拟合每个 `M` 的 empirical null quantile；冻结后应用 test。
4. 比较 final margin、mean、max、top-k、SECOND-like uncorrected scan 与 SCVD-corrected scan。
5. CPU 分析门：
   - `0/3` 中 raw max 的 95% 分位数随 `log M` 显著上升；
   - 校正后 false-exceedance 在各 `M` 下保持在目标 `alpha±2pp`；
   - fresh test 中，`final + corrected scan` 相对 `final + {mean,max,top-k}` 的最佳强基线
     macro AUROC 至少 `+0.02`，image-bootstrap 95% CI 排除 0；
   - Huatuo 与 Hulu、多数 findings 同方向；否则立即关闭。

这个实验在 patch scores 落盘后只做数组扫描和 bootstrap，预计 10--25 分钟；不会影响当前
baseline。现有 Nodule/Mass relocation `n=64` 只能作为“局部最大响应可能是假证据”的动机，
不能替代上述 null scaling test。

### 可能的论文终点

只有以下全部成立时才可能接近 ICLR oral：

1. 得到跨模型、跨数据、跨 medical/general image 的 search-size phase law；
2. 该 law 能预测 SECOND/VGA/ARCD 一类方法何时降低 FN、何时制造 FP；
3. SCVD 在 matched answer rate、matched claim count 和 matched parse rate 下同时改善 FP/FN；
4. OE/report 固定 coverage 后仍成立；
5. 相对 BCEA 与 SADT 的机制差异通过直接实验，而不是 related-work 文字成立。

若只是“减去 `sqrt(log M)` 后 VinDr 提升一点”，它是一个 calibration trick，不是 oral。

## 6. 两个诱人但应关闭的候选

### Cross-Fitted Gaze：关闭为独立主线

思路是用一组 heads/一个尺度选区域，另一组独立通道验证，再交换平均，以避免同一噪声既选择
又证明。数学上若 selection noise 与 verification noise 条件独立，held-out verifier 不受
winner's curse。这个想法很干净，但 SECOND 已经 coarse-to-fine，SADT 已经 attention-select
再 logit-lens verify；cross-fitting 更像统计加固，而不是新的 VLM 机制。只可作为 SCVD 消融。

### Quantifier-Dual Decoding：关闭为独立主线

思路是存在性 claim 用局部 witness，阴性 claim 用全域 coverage。它能解释医学 FP/FN
不对称，但最终会落到 noisy-OR/product-of-experts 和 multiple-instance learning；ARCD/BCEA
也已经做 claim/region-specific intervention。除非先发现跨任务的全新 quantifier phase law，
否则创新性不足。

## 7. 证据与可复现性

本地关键输入：

- SECOND conformance：
  `corrected_runs/unified_eval/sanity/second_identity_conformance_v1/conformance.json`，
  SHA256 `4ed0be7051fc22bf5fb212a8bb54efab76b6602820e7f8a2b0be5c097fc2b003`。
- SECOND recursion log：
  `corrected_runs/unified_eval/sanity/second_recursion_canary_v1.supervisor.log`，
  SHA256 `d0bd8d14377518299e8738c005dc6ab3c41737e1232d9eedc4476a540b96ebe7`。
- LLaVA CXR-VisHal strict evaluations：VISTA/VCD/AvisC/OPERA SHA256 分别为
  `d8204237...`, `2dbdbfe9...`, `a62af519...`, `e884dae7...`。
- Nodule relocation：`corrected_runs/c3_guard/vindr_nodule_relocation_n64_v2/analysis.json`，
  SHA256 `934a0e879e26f45023d1f0ce58a766f3d1fad2cf0f365caf8ada1b78a7ba566d`。
- sparse-lesion fresh audit：
  `corrected_runs/daylong_idea_search_v1/sparse_lesion_boundary_panel_v3.json`，
  SHA256 `c50d7087ec04e7a5a9834385a5aca2ac43b29571e644d1f6cb8f7fe09003f49f`；
  artifact 内含 dataset/model/seed/command/source fingerprint。

最终判断：**医学局部增强方向没有全部关闭，但“更聪明地选 patch”已经高度拥挤；唯一仍有
高层价值的切口是研究 adaptive visual search 自己制造了多少伪证据，并在解码前支付这笔
search tax。它当前是一个有自然现象与数学对象支撑的高风险候选，不是已完成的 ICLR idea。**

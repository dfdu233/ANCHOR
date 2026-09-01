# Rare/Weak Clinical Evidence 碰撞审计

> 检索日期：2026-08-12；范围：医学 VLM、小目标 VLM、医学 MIL、patch
> aggregation、Higher Criticism、rare/weak detection、multiscale scan。本文只做
> 无 GPU 文献与新颖性审计，不把尚未运行的算法写成结果。

## 1. 研究问题与直接判断

本轮固定三个问题：

1. VinDr 上稳定出现的“病灶越小，医学 VLM 的正确 claim margin 越低”是否是新现象？
2. `Higher Criticism (HC)` 能否把熟悉的“小目标困难”升级成一个有检测边界、可证伪的机制问题？
3. 是否存在一个足够简洁的算法，同时减少 false-positive hallucination，而不只是补回遗漏？

严格结论：

- **现象可信，但现象本身不新。** 仓库结果在两个模型、开发/确认两套图像上重复；然而 ICCV 2025 已在胸片 VLM 中明确指出小病灶 token 与 global token 失配，IJCAI 2025、NeurIPS 2025 又系统研究了小物体 VLM 失败与 training-free crop。
- **“Evidence Dilution + Top-K/局部加权”应直接淘汰。** 2026 年已有工作逐字使用 `critical evidence dilution`，方法也是 query-guided sparse Top-K visual residual；医学 MIL 中 mean/max/top-K patch pooling 更早已是标准操作。
- **可保留的窄方向是 `Sparse Evidence Boundary`，不是 `Evidence Dilution`。** 它问的不是“如何多看病灶”，而是：不同面积与强度的局部临床证据，何时在统计上根本不可检测，何时全局/最大池化失败但集体弱证据仍可恢复？在本次记录的检索式下，没有找到把 rare/weak phase diagram、reference-rank HC 与医学 VLM claim hallucination 联合起来的等价工作。
- **它目前不是 ICLR Oral idea 完成态。** `HC`、permutation-HC、检测边界均是成熟统计工具；若没有跨模型 phase collapse、增量纠错和新的有限样本/相关 patch 理论，它会被概括为“把经典 HC 用到医疗 patch score”。

## 2. 本地现象：强于偶然相关，但尚未证明“稀释”因果

本地完整结果见 [L1 Sparse Lesion Boundary](./l1_sparse_lesion_boundary.md)：所有样本均为
VinDr 三位读者一致阳性，使用病种内 Spearman，避免把“大心脏”和“小结节”直接混为一谈。

| Split | Model | n | 病种内 macro Spearman | 95% CI | 正方向 findings | FN rate |
|---|---|---:|---:|---:|---:|---:|
| Development | Huatuo | 480 | 0.232 | [0.143, 0.314] | 6/8 | 19.2% |
| Development | Hulu | 480 | 0.475 | [0.390, 0.544] | 8/8 | 34.6% |
| Fresh confirmation | Huatuo | 133 | 0.323 | [0.158, 0.462] | 5/7 | 27.8% |
| Fresh confirmation | Hulu | 133 | 0.415 | [0.222, 0.559] | 7/7 | 38.3% |

四个 permutation test 均 `p<=0.0006`。确认集里，Huatuo 的 FN 病灶中位面积为
`0.75%`，TP 为 `3.37%`；Hulu 分别为 `0.73%` 和 `3.37%`。因此“更小的、读者一致可见的病灶更容易被模型漏掉”是重复现象。

但它还不能被称为因果的 evidence dilution：

- bbox 面积是“标注范围”，不是有效视觉 token 数；
- 小病灶同时更低对比、更偏外周。在确认集，radial position 与 margin 也为负相关，Hulu 的 local contrast 与 margin 为正相关；
- 当前全是阳性，直接解释的是 omission/FN，而不是 fabricated-positive hallucination/FP；
- 所谓 fresh confirmation artifact 此前已经因其他问题打开过，不能包装成全新 prospective confirmation；
- 先前 bbox erase 的病灶特异因果效应未过门，因此“面积相关”不能自动推出“干预这些 patch 会纠错”。

## 3. 文献地图：哪些表面版本已经被占据

### 3.1 医学 VLM 与小病灶：现象已有直接先例

- **Seeing the Trees for the Forest**（ICCV 2025）发现胸片 VLM 的背景 token norm 偏高，且 global image token 不能代表小的 disease tokens；其 Disease-Aware Prompting 抑制背景、放大病灶特征。这与“局部临床证据被全局表征淹没”几乎是同一现象。[论文](https://arxiv.org/abs/2505.15123) / [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Huy_Seeing_the_Trees_for_the_Forest_Rethinking_Weakly-Supervised_Medical_Visual_ICCV_2025_paper.html)
- **Anatomy-VLM**（WACV 2026）明确以“主流 VLM 把医学图像当整体、忽略诊断所需细节”为动机，并使用 anatomy localization 与 multi-scale 信息。[IEEE DOI](https://doi.org/10.1109/WACV61042.2026.00278)
- **Attention Without Grounding**（arXiv 2026）提供最重要的反例：在 MedGemma、LLaVA-RAD、Qwen3-VL、CheXagent 上，没有一个 VLM 同时满足“使用图像”和“attention 与因果 occlusion/医生区域一致”；任何基于原生 attention 找病灶的算法都必须先过它的 shifted/random/occlusion controls。[论文](https://arxiv.org/abs/2607.18577) / [官方代码](https://github.com/thedatasense/medicalvlm_attention_without_grounding)

因此，医学 setting 不是空白；真正空白只能是**可检测边界与 hallucination 纠错之间的联合规律**。

### 3.2 一般 VLM：小物体、裁剪和局部引导已经拥挤

- **Understanding Visual Detail Hallucinations**（IJCAI 2025）评测 11 个 LVLM，确认小物体显著更差，并发现 image resizing 等 training-free 方法可改善。[论文](https://www.ijcai.org/proceedings/2025/212)
- **FOCUS**（NeurIPS 2025）使用 VLM KV cache 构造 relevance map，training-free 定位并裁剪相关区域；项目页当前仍标记 code coming soon。[论文](https://arxiv.org/abs/2506.21710) / [项目页](https://focus-mllm-vqa.github.io/)
- **AGLA**（CVPR 2025）直接融合 original-image global features 与 augmented-image local features 来缓解 object hallucination。[论文](https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html) / [代码](https://github.com/Lackel/AGLA)
- **VGA / Tell Model Where to Look**（CVPR 2026）从 visual token semantics 构造 grounding，再 training-free 引导 attention，直接面向 hallucination。[论文](https://arxiv.org/abs/2511.20032) / [代码](https://github.com/beta-nlp/VGA)
- **Beyond Scene Priors**（arXiv 2026）把同一问题命名为 `critical evidence dilution`，并在 decoder 边界加入 query-conditioned sparse Top-K visual residual。[论文](https://arxiv.org/abs/2607.04149)
- **Sparse Attention for Dense Open-Vocabulary Prediction in CLIP**（arXiv 2026）把 softmax 改为 `alpha-entmax`，用数据依赖的精确零阈值去除 low-salience tail；它使“稀疏化 attention”本身也不再新。[论文](https://arxiv.org/abs/2607.07135)

所以 crop、局部 attention、固定 Top-K、entmax、global-local assembly 都不能作为我们的核心新意。

### 3.3 医学 MIL 与 patch pooling：mean/max/top-K 是成熟基线

- **MI-Zero**（CVPR 2023）已经把 visual-language patch similarity 通过 mean/top-K 聚合到零样本病理诊断。[论文](https://arxiv.org/abs/2306.07831) / [官方代码](https://github.com/mahmoodlab/MI-Zero)
- **Key Patches Are All You Need**（CVPRW 2024）在皮肤镜和乳腺摄影中显式比较 max、mean、top-K patch pooling，并以小 patch 子集提升 demographic shift robustness。论文给出的代码链接截至本次检查返回 404。[论文](https://arxiv.org/abs/2405.01654)
- **TransMIL**（NeurIPS 2021，作为 canonical correlated-MIL baseline）已建模 patch 相关性，避免把所有 patch 当 iid。[论文](https://proceedings.neurips.cc/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html) / [官方代码](https://github.com/szc19990412/TransMIL)
- 2022 年的医学 pooling survey 已直接总结：小病灶只占图像很小区域时，average pooling 会让背景主导，而 max pooling 又易抓噪声。[综述](https://pmc.ncbi.nlm.nih.gov/articles/PMC8804673/)

因此，“医学 patch MIL + Top-K”是直接碰撞，不是跨领域创新。

### 3.4 Rare/Weak statistics：能提供边界，但经典结论不能据为己有

- Donoho–Jin 的 Higher Criticism 在 sparse normal mixture 中自适应未知 sparsity，达到 Ingster–Donoho–Jin detection boundary。[Annals of Statistics 2004](https://arxiv.org/abs/math/0410072)
- Hall–Jin 的 Innovated Higher Criticism 处理 correlated noise，说明 patch dependence 不能被随意忽略。[DOI](https://doi.org/10.1214/09-AOS764)
- Stoepker 等人的 permutation/rank HC 给出未知 null 下的 finite-sample exact test，并在一类模型中保持接近 oracle 的 power；这正适合“normal reference image”校准，但也是现成统计工具。[JASA paper](https://arxiv.org/abs/2009.03117) / [Annals 2025 rank extension](https://doi.org/10.1214/24-AOS2477)
- Sharpnack 的 multiscale scan 在位置、尺度、pattern 未知时扫描并校正尺度复杂度；因此“多尺度 scan 找病灶”也不是新数学。[COLT 2018](https://proceedings.mlr.press/v75/sharpnack18a.html)
- 更致命的是，**High-Dimensional Analysis of Single-Layer Attention for Sparse-Token Classification**（ICLR 2026）已经直接研究“weak、rare、randomly located informative tokens”，并证明训练后的 attention 相对线性分类器具有更好的 signal scaling。[论文](https://arxiv.org/abs/2509.25153) / [OpenReview](https://openreview.net/forum?id=Ae7VWAEIAW)

因此，rare/weak 数学是一个高质量**问题坐标系**，不是可直接宣称的理论贡献。

## 4. 机制级碰撞矩阵

| Work | 同现象 | 同机制 | 同干预 | 同 claim | 裁决 |
|---|---|---|---|---|---|
| Trees for Forest, ICCV 2025 | 是：小 disease tokens 被背景/global token 压弱 | 高度相同 | disease-aware feature weighting | visual grounding | `Evidence Dilution` 现象直接碰撞 |
| Visual Detail Hallucinations, IJCAI 2025 | 是：small object hallucination | size/resolution | resize/多 encoder | hallucination | “小目标导致幻觉”直接碰撞 |
| FOCUS, NeurIPS 2025 | 是 | internal relevance localization | training-free crop | fine-grained VQA | crop 路线直接碰撞 |
| AGLA, CVPR 2025 | 是 | global/local complement | local-global logit assembly | hallucination | global-local 融合直接碰撞 |
| VGA, CVPR 2026 | 是 | visual semantic localization | patch-guided attention | hallucination | patch steering 直接碰撞 |
| Beyond Scene Priors, arXiv 2026 | 是，且同名 dilution | 小目标被背景淹没 | sparse Top-K residual | fine-grained VQA | `dilution + Top-K` 直接碰撞 |
| MI-Zero, CVPR 2023 | 部分 | rare discriminative patches | mean/top-K V-L pooling | medical diagnosis | patch aggregation 直接碰撞 |
| Barnfield et al., ICLR 2026 | 抽象层相同 | sparse weak informative tokens | learned attention | classification boundary | 普通 sparse-token theory 直接碰撞 |
| Donoho–Jin / Stoepker | 抽象层相同 | rare/weak mixture | HC / rank-permutation HC | global detection | 数学工具完全已有 |
| **可保留 delta** | 医生一致病灶面积与 claim margin 的跨模型连续关系 | **claim-conditioned clinical patch field 的 detection phase** | **normal-reference rank HC + fixed-content claim exchange** | **FP 与 FN 联合纠错边界** | 未检索到机制等价工作；高风险 boundary extension |

## 5. 最简可测算法：Normal-Referenced Rank-HC Verifier

这不是“再做一个 Top-K”。输入是一张图像 `x` 和草稿中的原子 claim `c`，例如
“左侧小量胸腔积液”。用一个冻结的小型医学 image-text encoder（或目标 VLM 的
visual token，但后者风险更高）给每个 patch 一个局部 signed score：

\[
s_i(c)=\cos(E_v(P_i),E_t(c))-\cos(E_v(P_i),E_t(\neg c)).
\]

这里 `P_i` 是第 `i` 个图像 patch；第二项不是语言否定回答，而是同一 patch 对相反
临床命题的匹配度。随后从一个无该 finding 的 normal-reference bank 中，取相同解剖
位置的 patch scores `s_{j,i}(c)`，构造无需高斯假设的经验尾概率：

\[
p_i(c)=\frac{1+\sum_{j=1}^{M}\mathbf 1[s_{j,i}(c)\ge s_i(c)]}{M+1}.
\]

直观上，一个真正的小病灶不一定让任何单 patch “特别强”，但会让一小群 patch 的
`p_i` 一起偏小。将 `p_i` 从小到大排序，计算：

\[
HC(c)=\max_{1\le k\le \alpha N}
\frac{\sqrt N\,[k/N-p_{(k)}]}{\sqrt{p_{(k)}(1-p_{(k)})}}.
\]

它不是预先固定 `K`，而是自动寻找“多少个略异常 patch 合起来已经不可能由正常图像
解释”。为兼容 cardiomegaly 等 diffuse finding，用一次 multiplicity-calibrated omnibus：

\[
T(c)=\max\{Z_{mean}(c), HC(c)\},
\]

其中 `Z_mean` 负责密集证据，`HC` 负责稀疏弱证据；整个 `T` 的阈值用 reference-image
permutation 校准，而不是在 test labels 上调参。

解码只做两件事：

1. 草稿 positive claim 若 final margin 很高但 `T(c)` 不显著，则降为 uncertain，而不是把整份报告变短；
2. OE 中使用固定 positive claim 数 `K`，只允许一个低 `T(c)` 草稿 claim 与一个高 `T(c)` ontology omission 一换一，避免靠少说降低 hallucination。

这个版本与已失败的文本 RAG 不同：reference image 不作为 prompt 喂给 VLM，而只定义局部
visual evidence 的经验 null；它也与 NCD 不同，不对 claim×intervention 矩阵做双中心化。

## 6. 数学上真正能说什么

先给背景：设一张图像有 `N` 个 patch。无病灶时，校准后的 patch score 可近似看作
`Z_i~N(0,1)`；有病灶时，仅 `K=N^(1-beta)` 个 patch 的均值增加
`mu_N=sqrt(2r log N)`。`beta` 越大表示病灶越稀疏，`r` 越大表示每个 patch 越清晰。

### 6.1 一个有解释力、但属于经典结果的边界

全局均值的标准化信号为：

\[
\sqrt N\,\bar Z\;\text{的均值}
=\frac{K\mu_N}{\sqrt N}
=N^{1/2-\beta}\sqrt{2r\log N}.
\]

当 `beta>1/2`，它趋近于 0：病灶 patch 少于 `sqrt(N)` 量级时，即使每个 patch 有弱信号，
全局平均也会渐近失效。只看最强 patch 的 max test 则要求：

\[
r>\rho_{max}(\beta)=(1-\sqrt{1-\beta})^2.
\]

HC 的经典最优边界是：

\[
\rho^*(\beta)=
\begin{cases}
\beta-\frac12,&\frac12<\beta\le\frac34,\\
(1-\sqrt{1-\beta})^2,&\frac34<\beta<1.
\end{cases}
\]

所以在 `1/2<beta<3/4` 且 `rho*(beta)<r<=rho_max(beta)` 的区域里，**mean
被稀释、max 又嫌每个 patch 不够强，但 HC 可以利用一群弱 patch 的集体偏移**。这是该
方向最容易让非专业读者理解的数学 insight。

### 6.2 为什么这还不能算我们的理论贡献

上述边界来自 Donoho–Jin，不是新定理；permutation/rank calibration 也已有 Stoepker 等人。
ICLR 2026 又已经证明 attention 在抽象 sparse-token classification 上的优势。因此论文若
只重述这些式子，理论新颖性为零。

真正可能成为新结论的是一个**医学 VLM 有限 token、空间相关版本**：

> 在保留同图解剖相关性与 claim-specific language bias 的 reference-rank patch field 中，
> 哪个 `(evidence fraction, local strength, spatial dependence)` 边界决定 final decoder、
> max/Top-K、HC/scan 谁能纠错；不同模型和 finding 是否落在同一归一化 phase diagram。

这个结论只有在真实数据出现跨模型 phase collapse 后才值得证明。否则不要为了“有数学”
先写一个脱离数据的 toy theorem。

## 7. 致死实验：最快判断它是新机制还是漂亮类比

### Gate A：先验证 patch statistic 有增量，不先做完整解码

数据用 image-disjoint VinDr dev/test，必须同时包含 `3/3 positive` 与 `0/3 negative`；当前
133 个 bbox positive 只够测 FN，不够测 hallucination。每个 finding 比较：

- target VLM final margin；
- frozen encoder global score；
- max、固定 Top-K（1%、5%、10%）；
- rank-HC；
- penalized multiscale scan；
- `final margin + 每一种 patch statistic` 的 dev-only calibration。

唯一推进门：在 fresh test 上，HC/scan 相对 `final + best(mean,max,top-K)` 的 macro AUROC
增量至少 `0.02`，image-bootstrap 95% CI 下界大于 0；NLL/Brier 同向；Huatuo 与 Hulu
均成立，且至少 5/7 findings 正向。未过即关闭，不用换 K 修补。

### Gate B：定位必须是因果的，不接受漂亮热图

对 HC/scan 选中的 patch 同时做：

- radiologist bbox enrichment；
- shifted、same-size random、far-region controls；
- patch occlusion margin drop；
- 同数量 mirror/background patch intervention。

必须同时超过 bbox controls 与 causal occlusion；只 overlap 不改变 margin，按
`Attention Without Grounding` 的标准判失败。

### Gate C：把 area correlation 变成因果 scaling law

在 bbox 仅用于机制实验时，对 lesion visual tokens 做等数量、norm-matched duplication/
reweighting，并用 mirrored background tokens 做 placebo。若是 dilution，增加 lesion effective
mass 应单调提高正确 claim margin，而 background placebo 不应同向；效应还应随原始 area
变小而增强。若两者都统一推高 Yes，则只是旧的 operating-point shift，立即淘汰。

### Gate D：最终必须改善 hallucination，而非只救 FN

CE 要同时报告 FP/FN；OE 使用 fixed-K exchange。方法进入主线需满足：

- positive-content hallucination 相对下降至少 20%；
- omission 不增加；
- matched claim count、回答长度、拒答率后仍成立；
- 至少两个模型、CE 与 OE 两类任务成立。

## 8. 新颖性风险与最终裁决

| 维度 | 当前评分 | 原因 |
|---|---:|---|
| Importance | 3/3 | 小病灶遗漏与无证据阳性均是医学安全核心，且本地跨模型重复 |
| Mechanistic value | 2/3 | rare/weak boundary 能预测 mean/max/HC 的 work/non-work；但因果 patch evidence 未证 |
| Novelty space | 1/3 | dilution、Top-K、crop、sparse attention、sparse-token theory 均有强碰撞 |
| Executability | 2/3 | VinDr bbox、两个模型和多个官方 patch/occlusion 实现可用；需要新增 patch score |

**裁决：保留为高风险致死实验，不得以 `Evidence Dilution` 直接立项。** 如果 Gate A
不过，这只是又一次“小目标很难”；如果 Gate A/B 过而 Gate D 不过，它可成为检测边界/
评测论文；只有 A–D 通过、且真实模型结果按 `(beta,r)` 出现跨架构 phase collapse，才有
资格升级为 ICLR 级 `Sparse Evidence Boundary` 主线。

## 9. 可直接借鉴的开源实现

以下仓库在 2026-08-12 通过远端 HEAD 或论文项目页核验：

| Repository | 可借鉴内容 | 核验状态 |
|---|---|---|
| [thedatasense/medicalvlm_attention_without_grounding](https://github.com/thedatasense/medicalvlm_attention_without_grounding) | 16x16 patch occlusion、bbox/shift/random controls | HEAD `e79bf0a` |
| [beta-nlp/VGA](https://github.com/beta-nlp/VGA) | visual-token semantic confidence 与 training-free guidance | HEAD `e800fa7` |
| [yunhang8658/MedOPD](https://github.com/yunhang8658/MedOPD) | original vs evidence-degraded counterfactual score | HEAD `942e77b`；training-based |
| [AIDASLab/SECOND](https://github.com/AIDASLab/SECOND) | entropy selective mask 与 contrastive decoding baseline | 官方 ICML 2025 repo |
| [mahmoodlab/MI-Zero](https://github.com/mahmoodlab/MI-Zero) | patch-text similarity、mean/top-K pooling | HEAD `047922f` |
| [szc19990412/TransMIL](https://github.com/szc19990412/TransMIL) | correlated MIL baseline | HEAD `9d6aee5` |
| [Lackel/AGLA](https://github.com/Lackel/AGLA) | global/local hallucination mitigation baseline | 论文官方链接；需另审 license |
| FOCUS | KV-cache crop baseline | 项目页仍为 `code coming soon` |
| MedicalMultiple-Instance-Learning | max/mean/top-K medical baseline | 论文给出的 GitHub 链接本次返回 404，不可依赖 |

### 检索完整性说明

检索覆盖精确机制、同义词、邻域任务和数学邻域，包括：`small lesion medical VLM`、
`critical/evidence dilution`、`fine-grained VQA crop`、`visual token grounding hallucination`、
`medical MIL top-k pooling`、`Higher Criticism vision-language`、`rare weak patch`、
`softmax sparse-token classification`、`multiscale scan image`。结论只能表述为：**在这些
检索下，没有找到“normal-reference rank-HC + 医学 VLM claim hallucination phase boundary”
的机制等价工作**；不能声称证明首创。

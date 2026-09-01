# Spatial Claim Binding：机制碰撞与 VinDr CPU 淘汰审计

**审计日期：** 2026-08-03  
**范围：** finding identity 已保留、但 anatomy/location/extent 绑定在医学
VLM 的 OE/报告生成中丢失；截至审计日的顶会、顶刊、arXiv、官方代码与
本地 VinDr-CXR 数据。  
**执行约束：** 只做检索和 CPU/标签审计；未运行 GPU，未修改共享评测。  
**最终决定：** **NO-GO。** broad spatial-claim binding 在机制和医学任务两侧
均已发生实质碰撞；`/workspace/vinbigdata` 虽能提供可靠的**图像坐标定位**，
但不能提供本问题所需的反平衡 finding↔location 置换、患者解剖侧、
extent 语义或报告级绑定真值。

## 1. 精确定义：什么才算 binding failure

下列三个错误必须严格分开：

1. **identity/perception failure：** 模型没有识别 finding；
2. **localisation failure：** 模型识别 finding，但没有找对其像素区域；
3. **binding failure：** 模型分别保留了 findings 集合和 locations 集合，却把
   location/extent 配给了错误 finding，例如把 A 的位置赋给 B。

只有第三种才支持本方向。单个 finding 的框预测错误、attention 没落在 lesion
上、报告出现错误 laterality，均不足以证明 binding；它们也可由感知不足、
OCR/朝向错误、finding-specific anatomy prior、报告解析错误或 box 风格造成。

假设一个合格样本含两个可见 finding `A,B`，其独立空间真值为
`g_A,g_B`。候选机制要求至少存在某层 `l*`：

\[
I_{l^*}(A),I_{l^*}(B),G_{l^*}(g_A),G_{l^*}(g_B)
\quad\text{均可恢复，且}\quad
B_{l^*}(A\!\leftrightarrow\!g_A,B\!\leftrightarrow\!g_B)
\]

显著优于最终生成状态；错误必须表现为配对交换，而非 identity 或 location
边际本身消失。最佳层允许随架构变化，本审计**不**假设统一 early-layer
erasure，也不复活先前失败的 universal early-erasure 主张。

## 2. 真正需要的自然变量

决定性自然变量不是 `left/right` 单词，也不是 box 中心，而是同一 acquisition
与输出语义下的**反平衡空间角色置换**：

```text
同一 finding identity set {A, B}
arm 1: A@u, B@v
arm 2: A@v, B@u
```

其中 `u,v` 必须有独立 reader truth，两个 arm 的图像质量、finding prevalence、
框面积、claim 数和语言模板可匹配，并且不能从 `A/B` 身份本身猜出位置。
若目标语言是患者 laterality/anatomy，还必须有 DICOM orientation/side-marker 到
patient reference frame 的可靠映射；若目标是 extent，还必须有 radiologist
明确标注的 unilateral/bilateral、zone/lobe 或 extent 语义。box 数量和覆盖面积
不能替代这些变量。

这一定义也区分四种真值：

| 真值层 | VinDr box 能否提供 | 可支持的结论 |
|---|---:|---|
| finding identity/polarity | 是，多读者独立标签 | finding support |
| pixel geometry | 是，多读者矩形框 | 图像坐标定位/reader box agreement |
| visibility | 部分；local finding 的框表示该 reader 愿意定位 | 不能推出未框区域不可见或信息充分 |
| report semantics | 否；VinDr 没有逐 claim 报告短语 | 不能定义 anatomy/laterality/extent 语言是否正确 |

## 3. VinDr CPU substrate audit

### 3.1 输入与可复现来源

- boxes：`/workspace/vinbigdata/train.csv`；
- images：`/workspace/vinbigdata/train/*.dicom`；
- independent labels：
  `/home/dbw/datasets/physionet/vindr-cxr/1.0.0/annotations/image_labels_train.csv`；
- 固定 panel：`R8/R9/R10`；
- 已有审计 artifact：
  `/home/dbw/datasets/physionet/vindr-cxr/1.0.0/spatial_reader_v1/summary_v1.json`，
  fingerprint `534c97b8ece525a854029244e21221f0486d21456c24ecaf5b4b434e96d1fa5c`。

所有统计均条件化于 R8/R9/R10 三人独立标注；模型输出、LLM judge、RadGraph、
报告 parser 均未定义真值。

### 3.2 独立像素定位确实存在

| CPU 统计 | 数量/结果 |
|---|---:|
| R8/R9/R10 完整 panel images | 5,501 |
| 所有 `3/3` positive claims（含 global diagnoses） | 8,873 |
| 三位 reader 均提供 local bbox 的 claims | 5,112 |
| 对应 unique images | 3,537 |
| 三位 reader 均恰有一个 box | 3,988 |
| 三位 reader box 数完全一致 | 4,237 / 5,112 |
| exact-one cases：median max pairwise center distance（x/W, y/H 平面） | 0.0218 |
| exact-one cases：90th percentile max center distance | 0.0581 |
| exact-one cases：median mean pairwise IoU | 0.704 |
| exact-one cases：mean pairwise IoU ≥ 0.5 | 3,529 / 3,988 (88.5%) |

因此“VinDr 没有空间信息”是错误结论。它对大量 local findings 提供相当一致的
像素坐标。相反，875/5,112 claims 的 reader box 数不一致，且 diffuse/multiple
lesion 类别尤其明显，说明 box 数和 union area 同时混入病灶多发性、矩形粒度和
reader drawing style；它们不能直接当作临床 extent 或 uncertainty。

### 3.3 binding 所需的反平衡置换不存在

5,112 个三读者定位 claims 中：

- 2,177 images 只有一个 localized unanimous finding；
- 1,168 images 有两个，169 有三个，23 有四个；
- 因而只有 1,360 images 可形成至少一个 within-image binding pair，192 images
  含至少三个 localized claims。

然而 co-occurrence 并不等于 binding substrate：

| 门槛 | 结果 |
|---|---:|
| co-occurring finding pairs 总数 ≥100 | 3 pairs |
| 三位 reader 对两个 findings 均恰有一个 box，数量 ≥100 | 1 pair |
| 上述 clean pair | aortic enlargement + cardiomegaly，739 images |
| 两 finding 的 reader-consensus 中心距离 ≥0.15（x/W, y/H 平面） | 739/739 |

唯一大样本 clean pair 是 anatomy-fixed 的主动脉增大与心影增大；模型仅凭 finding
identity prior 就能猜到哪一个在上/下区域，无法区分 conjunctive binding。

更致命的是，在 5% midline guard 下枚举两个 findings 位于相反图像半侧的 clean
pairs：观察到的合格 pair identity 全部只出现一个方向，例如
`aortic enlargement@R / pulmonary fibrosis@L`，**没有一个 pair 同时出现
`A@L,B@R` 与 `A@R,B@L` 各至少 10 例**。因此：

\[
\#\{(A,B): n_{LR}\ge 10 \land n_{RL}\ge10\}=0.
\]

这在 CPU 阶段就淘汰了“边际 identity/location 都对但 pairing 错”的自然因果
测试；任何 GPU layer probe 都会把 binding 与 disease-specific anatomy prior
混在一起。

### 3.4 anatomy、laterality、extent 与报告语义仍不成立

- 已有完整 DICOM header audit 显示 `PatientID`、`StudyInstanceUID`、
  `ViewPosition`、`PatientOrientation`、`Laterality`、`ImageLaterality` 和
  `ImageOrientationPatient` 在该释放路径不可用；split 只能 image-disjoint，
  不能验证 patient-disjoint。
- box x/y 是 stored-image coordinate，不是患者左右侧或 lung zone。
- 多个 boxes 可表示双侧病变、多个病灶或 reader 的绘框粒度；大 box 也不等于
  radiologist 在报告中承诺“diffuse/extensive”。
- VinDr 没有与每个 box 对齐的报告短语。MS-CXR 的官方说明恰好解释了为何这很
  重要：它额外由 radiologists 审阅 phrase↔box，并排除无法从单张图像 grounding、
  高 uncertainty、多 finding 混句、位置不匹配和纵向信息等样本。

所以 VinDr boxes 可以定义 `where in stored pixels did readers draw?`，不能定义
`what anatomy/location/extent should a generated clinical sentence assert?`。

## 4. 截至 2026-08-03 的机制级碰撞

### 4.1 通用 VLM 机制已直接覆盖 identity-location binding

| 工作 | 已覆盖的核心 | 对本方向的影响 |
|---|---|---|
| [Visual Symbolic Mechanisms, ICLR 2026 Oral](https://openreview.net/forum?id=3RQ863cRbx) | content-independent spatial indices 绑定 object features，并有 causal mediation | “保留 identity/location、丢失 conjunction”不是新的通用机制问题 |
| [Linear Mechanisms for Spatiotemporal Reasoning, ICLR 2026](https://arxiv.org/abs/2601.12626), [code](https://github.com/Raphoo/linear-mech-vlms) | spatial IDs 线性绑定到 text activations；中间层 causal belief intervention | 线性 location subspace、probe→patch→belief switch 直接碰撞 |
| [The Dual Mechanisms of Spatial Reasoning, arXiv 2026](https://arxiv.org/abs/2603.22278), [code](https://github.com/Nix07/spatial-variable-binding) | vision encoder 的全局 spatial layout 是主路径，LM relation 是次路径；增强 vision-derived spatial representation 可改善 reasoning | 进一步否定“统一 late decoder erasure”；机制随路径/架构变化 |
| [Mechanisms of Object Localization in VLMs, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Schaumloffel_Mechanisms_of_Object_Localization_in_Vision-Language_Models_CVPR_2026_paper.html), [code](https://github.com/t9s9/vlm-loc-mechanisms) | classification/localisation 共用少量早期处理但主要依赖不同 sparse heads；token ablation、attention knockout、causal mediation | identity 与 localisation pathway 分化及因果定位本身已被占据 |

把通用场景的 object/color/location 换成 CXR finding/anatomy，而不发现一个由医学
成像假设导致的新 boundary，会被合理评价为 mechanism transfer。

### 4.2 医学 grounding、location error 与 correction 也已覆盖

| 工作 | 已覆盖的医学问题/方法 | 剩余 delta |
|---|---|---|
| [HEAL-MedVQA / Localize-before-Answer, IJCAI 2025](https://www.ijcai.org/proceedings/2025/0853.pdf) | 基于 VinDr/MIMIC 的 disease-at-anatomy QA、location/disease perturbation、doctor masks、先定位后回答 | “医学 hallucination 来自未定位”及 region-first mitigation 已占据 |
| [MAIRA-2 + RadFact](https://arxiv.org/abs/2406.04449), [RadFact code](https://github.com/microsoft/radfact) | finding sentence 与 boxes 联合生成；sentence correctness/completeness 和 grounding quality | OE/report 的 finding↔box 输出与评价已建立 |
| [PadChest-GR](https://arxiv.org/abs/2411.05085) | 4,555 studies、7,037 positive sentences、逐 finding categorical location、最多两套独立 reader boxes | linked finding-location-report gold 数据本身不是贡献；论文还明确记录 box-style variability |
| [Grounding on PadChest-GR, MIDL 2026](https://proceedings.mlr.press/v315/aas-alas26a.html) | full image + grounding masks、region-to-text，检验 grounding 对报告质量/临床准确性的作用 | “加入 grounding 减少 hallucination”已是现成方法线 |
| [Phrase-grounded Fact-checking, MICCAI 2025](https://papers.miccai.org/miccai-2025/0693-Paper3526.html) | 合成 finding/location perturbation，检测 finding veracity 与 indicated-location error | finding 对、location 错的 detection 已直接覆盖 |
| [Phrase-grounded APO, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Mahmood_Phrase-grounded_APO_for_Improving_Chest_X-ray_Report_Generation_CVPR_2026_paper.html) | structured finding/anatomy/laterality、location-aware fact checking、inference-time report correction | 医学 location-aware correction 已被占据；fixed coverage 只是更严格控制 |
| [RadSCR, ICLR 2026](https://openreview.net/forum?id=6sOSwgCmpH) | region-guided abnormality proposals 与多路 self-critique，兼顾 false positives/negatives | region-grounded OE/report self-correction 已形成强 baseline |
| [BoxMed-RL, MedIA 2026](https://arxiv.org/abs/2504.18453) | spatially verifiable report generation，finding-box RL | 空间可验证训练亦非空白 |

没有检索到一篇工作同时使用**独立 reader 分布 + counterbalanced binding permutation
+ fixed-identity causal patch**。但这不是当前可投稿 delta：前两部分分别已由通用
机制线和医学 grounding 线占据，而 VinDr 又缺少第三部分所需的有效自然变量。

## 5. 理论上最小的因果实验，以及为什么现在不该运行

若未来获得通过门槛的数据，最小实验应为：

1. 同一图像有 A/B 两个 finding，identity、polarity、patient-side/anatomy、
   phrase↔box 均由独立专家给出；另有同 identity set、相反 A/B 空间排列的 donor。
2. teacher-force 完全相同的输出骨架：
   `(A, location_A); (B, location_B)`，只评分两个 location tokens/boxes；因此 claim
   identity、polarity、数量、顺序、长度和 coverage 固定。
3. 在每个模型的 dev-only 候选层估计 pairing state；不得假设相同层。先投影掉
   identity、polarity 和 claim-count directions：

   \[
   d_{bind}=P_{\perp\{I,P,K\}}
   (h_l^{opposite\ arrangement}-h_l^{same\ arrangement}).
   \]

4. 只替换/旋转 `d_bind` 分量，并恢复原 activation norm；位置 marginal、非目标
   claim state 和 visual-token 数保持不变。
5. 唯一支持性预测：A/B 的 location assignment 交换，而 A/B identity logits、
   polarity、其他 claims、claim count 与 clear-case localisation 基本不变。

必须加入 same-identity image swap、random direction、norm-matched random subspace、
position-token-only patch、generic spatial-ID steering、prompt paraphrase、box-area/
finding-frequency controls。若 patch 同时改变 finding identity、删 claim 或只改善
单 finding localisation，便不是 binding mechanism。

这个实验在 concept 上可证伪，但当前 VinDr 的 counterbalanced pairs 为零，运行它
只能得到不可解释的相关性，因此不授权 GPU。

## 6. One-day kill gate（未来数据到位时）

在任何模型 forward 前，CPU manifest 必须同时满足：

1. 至少 **3 个** distinct `(A,B)` finding pairs；
2. 每个 pair 的 `A@u,B@v` 与 `A@v,B@u` 均至少 **100 个 patient-disjoint** studies；
3. A/B identity 均为 `3/3` positive，且每个 location/box 至少 `2/3` reader agreement；
4. identity-only classifier 对 assignment 的 balanced accuracy ≤0.60；
5. box count、area、center、image quality、acquisition/view 和 prevalence 在 arms 间
   预先平衡；
6. 若输出 anatomy/laterality/extent 语言，必须有 expert-linked phrase semantics、
   orientation/reference frame 与临床 attribute gold；否则 claim ceiling 只能是
   stored-pixel coordinate grounding；
7. dev/test 按 patient 分离，且同 patient、study、near-duplicate 不跨 split。

**当前结果：** gates 1、2、6、7 明确失败，gate 4 对唯一大 pair 也显然不可满足；
因此 one-day decision 已是 **KILL**，无需 GPU 行为筛选。

## 7. 唯一可能改变决定的医学 boundary

只有两种新变量可能把这个方向从“通用 binding 医学化”提升为新的机制问题：

- **patient-reference-frame mediation：** 像素位置正确，但 acquisition orientation/
  side marker 到患者解剖侧的转换在模型内部失败；
- **semantic extent admission：** reader 对同一 finding 的临床 extent/laterality/
  zone 有独立语义分布，而模型把该分布错误绑定或过度确定化。

前者要求 orientation、marker、patient-side gold 和可独立干预的 reference-frame
变量；后者要求显式 clinical attribute labels，而不是从 box 个数/面积推断。
当前 VinDr 路径两者都不具备。PadChest-GR 或 Chest ImaGenome gold 可作未来
schema/count audit，但在真实反平衡 cells 和 reference-frame 变量通过前仍是
**data candidate，不是已批准方向**。

## 8. Reviewer-style verdict

**Broad finding–anatomy/location/extent binding：NO-GO。** 通用 VLM 文献已经给出
spatial IDs、vision/LM 双路径、classification/localisation 分工和 causal patching；
医学文献已经给出 grounded VQA、grounded RRG、finding-location fact checking、
location-aware correction 与 spatially verifiable training。剩余的 fixed identity、
coverage 和 norm 控制是优秀实验规范，却不足以构成 ICLR oral 机制贡献。

**VinDr reader boxes：对 pixel localisation 有价值，对 report binding 不合格。**
它们显示三位 reader 在 3,988 个 single-box claims 上具有很强几何一致性，但没有
反平衡多 finding 空间置换、患者/朝向映射、临床 extent 语义或 phrase-level
report truth。继续运行 layerwise GPU probe 会把 anatomy prior、localisation、
reader box style 和真正 binding 混为一谈。

因此本分支应保留为将来更好数据上的严格 control protocol，但**不进入当前论文
主线、不占用 GPU、不包装成医学 VLM hallucination 的通用解决方案**。

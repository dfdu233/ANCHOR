# Observation Policy Is a Prompt：crop/zoom 是否把“为什么给你看这一区域”误当成病变证据？

> 独立机制碰撞与致死实验设计，2026-08-12。本文不占用 GPU、不修改 baseline，也不把尚未验证的解释写成结论。

## 1. 冻结问题与当前证据

研究问题不是“crop 是否有用”，而是：

1. 当前 crop 造成的巨大假阳性，主要来自**全局上下文被删掉**、**局部图像被放大后的分布外输入**，还是模型把“特意展示一个局部区域”理解成了**这里大概有异常**？
2. 如果观察操作本身携带了这种隐含预设，能否在像素完全不变时，仅改变 crop 的来源说明就改变诊断？
3. 这个现象相对 visual prompting、prompt-induced hallucination、adaptive acquisition calibration 和医学 informative observation 还有多少新颖空间？

真实确认集是 VinDr 中七个 finding 均为三位 reader 一致阴性的 62 张图。Huatuo 的结果为：

| 输入 | FP rate | mean margin |
|---|---:|---:|
| 完整 X-ray，固定单 claim | 8.1% | -0.679 |
| 同图随机局部 crop，固定单 claim | 62.9% | +0.097 |
| 搜索 7 claims × 361 regions 后的 selected crop | 79.0% | +0.617 |
| 同配置 random crop | 61.3% | +0.341 |

最大搜索配置下 selected 比 random 多 17.7pp FP，95% CI `[3.2, 33.9]`；但固定 claim 时把 region 数从 16 增至 361，selected–random gap 的增长 CI 为 `[-0.052, 0.206]`，未通过预注册门。因此：

- **搜索噪声会随 region 数稳定传到最终错误**这一强命题已失败；
- 更大的已确认现象其实是：**不论是不是 selected，只要把一块局部胸片裁出来并铺满输入，FP 就从 8.1% 跳到约 63%**；
- 这组数据只证明 crop render 与完整图的模型行为不同，尚不能区分 context loss、resize OOD、局部纹理歧义和 observation-policy prior。

## 2. 为什么“观察策略也是 prompt”是自然假设

人类看到一张完整胸片和看到一张被医生特意放大的局部图，不会只比较像素。后者还暗示：**有人为什么选择给我看这里？** 这与语言中的“请检查这个可疑结节”类似，问题本身带有预设。

在医学数据中，检查是否被开具、哪项化验被测量、哪张局部图被保存，本来就常常依赖医生的怀疑。EHR 文献称之为 informative observation / informative missingness：观察行为本身与疾病相关。真正危险的是，把训练或交流中“因可疑而观察”的关联，错误带到我们在测试时随机制造的 crop 上。

### 2.1 必要的数学背景：像素证据与观察策略先验

令：

- `Y=1` 表示 finding 存在，`Y=0` 表示不存在；
- `A` 表示观察策略，例如完整图、随机 crop、医生因怀疑而选择的 crop；
- `Z` 表示策略 `A` 最后展示出来的像素。

Bayes 公式把诊断后验的 log-odds 精确分成三项：

\[
\log\frac{p(Y=1\mid Z,A)}{p(Y=0\mid Z,A)}
=
\underbrace{\log\frac{p(Z\mid A,Y=1)}{p(Z\mid A,Y=0)}}_{\text{展示像素的临床证据}}
+
\underbrace{\log\frac{p(A\mid Y=1)}{p(A\mid Y=0)}}_{\text{为何产生这个观察的策略证据}}
+
\underbrace{\log\frac{p(Y=1)}{p(Y=0)}}_{\text{基础患病率}}.
\]

背景解释：第二项并不总是错误。若医生只在怀疑病变时拍局部放大图，那么“出现了放大图”确实包含信息；但我们在 benchmark 中用程序随机裁图时，`A` 与真实疾病无关，正确的第二项应为 0。若 VLM 仍保留了“局部图通常因异常而出现”的正偏置，便会在没有病灶的 crop 上过度报阳性。

这只是标准 Bayes 分解，不是原创定理。可能的新贡献必须来自一个真实规律：**冻结 VLM 会不会把推理时的 observation policy 当成跨模态语用信息，并且该项能否在像素不变的反事实实验中被单独识别。**

若负例上的可见像素分数为 `E`，crop 策略额外加入偏置 `b_A>0`，阈值为 0，则

\[
\mathrm{FPR}_A=P(E+b_A>0\mid Y=0)=1-F_0(-b_A).
\]

这说明一个固定的策略偏置可以大幅推高 FP；但 context removal 同样会改变 `E` 的均值和方差，所以仅凭 8.1%→62.9% 不能反推出 `b_A`。

## 3. 四个竞争机制及其独有预测

| 机制 | 因果变量 | 独有预测 | 哪个结果将其否定 |
|---|---|---|---|
| Context removal | crop 删除完整心影、双肺对称性和解剖位置 | 在 crop 像素不变时，补回一个真实全图 thumbnail 或保留模糊全图结构会显著降低 FP | 补回上下文完全不改变 margin/FP |
| Resize OOD | 1/4 边长区域被放大成完整输入，纹理尺度异常 | 在上下文和 canvas 匹配后，FP 随放大倍数增加 | native-scale 与 enlarged crop 无差异 |
| Implicit observation-policy prior | 紧裁图这种展示形式暗示“这里值得看” | 相同 crop 像素下，明确说明“均匀随机选择、选择本身不是证据”会特异性降低 crop FP，但对 full image 影响很小 | 说明文字对 full 与 crop 造成同样的全局 Yes-rate 平移 |
| Textual presupposition | “可疑区域/检查某病”直接把答案写进文字上下文 | 相同图像下，suspicious framing 比 random framing 抬高阳性；完整图上也可出现 | suspicious/random 的文字差异不改变输出 |

“局部图更难”是四者都能解释的现象。只有同像素 provenance 反事实和 context/scale 匹配才能把它们分开。

## 4. 2023–2026 邻近工作与碰撞判断

### 4.1 Crop 已经被明确视为 visual prompt

NeurIPS 2023 的 Fine-Grained Visual Prompting 已把 crop、box、circle、mask 和 blur 作为不同 visual prompt 系统比较，并发现 Blur Reverse Mask 在突出目标的同时保留空间上下文，比直接粗糙裁剪更好。这是“crop 不只是预处理，而会改变任务输入语义”的直接前作，也给 context-removal 对照提供了现成实现。[Yang et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4e9fa6e716940a7cfc60c46e6f702f52-Abstract-Conference.html)

NAACL 2025 的 BBVPE 进一步显示 box/circle 等 visual prompt 会显著改变 LVLM object hallucination，并训练 router 为每张图选择 prompt。因此，“视觉标记能改变幻觉”已经被占据；本候选只能研究**为什么一种观察格式会在无病变时提高 FP**，不能把 visual prompting 本身当创新。[Woo et al., NAACL 2025](https://aclanthology.org/2025.naacl-short.45/)

### 4.2 Crop/zoom 提升细节与自适应搜索空间已拥挤

HALC 已使用 auto-focal grounding 与 local/global focal contrast 降低 object hallucination；AGLA、SECOND、Perception Magnifier、CropVLM、AdaptVision、ZoomEye 等继续覆盖局部选择、放大、重编码和多尺度解码。尤其 BCEA 直接指出 crop max 会同时抬高真 claim 和假 claim，并要求把完整 acquisition policy 纳入 post-acquisition calibration。[HALC, ICML 2024](https://proceedings.mlr.press/v235/chen24bi.html)，[BCEA, 2026](https://arxiv.org/html/2606.16667v4)

BCEA 与本候选的差别很窄但真实：BCEA 研究 adaptive acquisition 破坏 conformal exchangeability 以及如何重校准；它没有把**同一 crop 的产生理由作为语用变量**，也没有拆分 crop 引起的 context、scale 与 provenance 效应。不过，“把 acquisition policy 纳入推断”这一高层原则已经高度碰撞。

### 4.3 文本预设导致 VLM 幻觉已有直接机制论文

ACL 2026 的 Prompt-Induced Hallucination 在受控计数任务中让问题高报物体数，发现 VLM 随对象增多更容易顺从 prompt，并定位出能降低该行为的少量 attention heads。因此，如果实验只有“suspicious wording 让 FP 上升”，它只是医学复现，不构成新方向。[Rudman et al., ACL 2026](https://aclanthology.org/2026.acl-long.1941/)

CogSci 2025 的 Multimodal Pragmatic Inference 显示 LLaVA、InstructBLIP 和 GPT-4o 会联合视觉和语言上下文作语用推断，而且这种推断对 in-context visual statistics 敏感。这支持“图像展示方式可参与语用推断”的可行性，但它研究 referring expressions，不研究 acquisition provenance 或医学 hallucination。[McGee et al., CogSci 2025](https://escholarship.org/uc/item/5pf870ff)

### 4.4 医学领域早已承认“观察行为有信息”

医疗风险预测文献系统讨论 informative presence / observation：是否测量某项指标会反映临床工作流和医生怀疑；缺失模式甚至可单独预测结局。[review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7810439/) 临床 label leakage 框架也强调诊断是“怀疑→采集证据→再评估”的动态过程，后续采集行为可能泄漏诊断阶段。[framework](https://pmc.ncbi.nlm.nih.gov/articles/PMC10746313/)

医学影像 shortcut 工作则显示 acquisition/site/position/center crop 会成为捷径，且随机 cropping 可能删除关键外周病灶。[Radiology AI review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530765/)，[MICCAI 2024 shortcut study](https://papers.miccai.org/miccai-2024/695-Paper0423.html)

### 4.5 碰撞矩阵

| 工作 | 同现象 | 同机制 | 同干预 | 同主张 | 剩余差异 |
|---|---:|---:|---:|---:|---|
| Fine-Grained Visual Prompting | 是 | 部分：crop/blur 改变视觉输入 | crop、blur、mask | 否 | 不研究无病灶 FP 与 provenance |
| BBVPE | 是：visual prompt 改变 hallucination | 否：router 选效果最好的 prompt | box/circle visual prompts | 部分 | 不拆 observation-policy prior |
| BCEA | 是：crop max 抬高 false claim | 部分：acquisition policy 进入 score | crop/zoom 后校准 | 高度相邻 | 不做同像素 policy semantics 反事实 |
| Prompt-Induced Hallucination | 部分 | 是：文本预设压过视觉 | 改 prompt、消融 heads | 部分 | 不研究视觉观察操作的隐含预设 |
| Multimodal Pragmatic Inference | 部分 | 是：视觉×语言语用推断 | 视觉集合与 referring expression | 否 | 不研究 crop provenance/医学错误 |
| Informative observation in health | 否：非 VLM | 是：观察机制与疾病相关 | missingness/acquisition modeling | 高层同义 | 不研究推理时人工 crop 错配 |

检索到的工作已经覆盖每个单独成分；没有检索到同时用**相同医疗像素、反事实 observation provenance、context/scale 匹配**证明 crop 操作本身触发 VLM 临床假阳性的工作。这里的剩余空间是联合机制测量，不是新数学或现成新算法。

## 5. 最小同像素 GPU 致死实验

### 5.1 数据与预算

- development：32 张七 finding 全阴性图 + 32 张 reader 3/3 clear-positive、且 bbox 确保 crop 内含病灶；只做接线和冻结。
- confirmation：当前 62 张全阴性图 + 62 张按 finding/病灶面积匹配的 clear-positive。
- Huatuo 先跑；只有主交互门通过才在 Hulu 原样确认。
- 单步 claim scoring，不生成长回答。开发阶段最多 `64 images × 5 renders × 3 prompts = 960` 次；确认同量级。不会占用 baseline 长任务超过一个短窗口。

### 5.2 五种 render；核心 ROI 来源完全相同

每张图先冻结一个 ROI，后续所有条件复用同一 ROI 像素和位置，不因输出重新选择。

1. **Full**：原始完整 X-ray。
2. **Native-context-removed**：ROI 保持原始像素、大小与位置；ROI 外做强 Gaussian blur，而不是裁掉。它主要测试 context removal，同时避免 resize。
3. **Native+sham panel**：固定 canvas 中放 native-scale ROI，旁边放全图的 phase-scrambled thumbnail；控制 panel、像素量和直方图。
4. **Zoom+sham panel**：与 3 完全相同的 canvas 和 sham context，只把同一 ROI 放大；3→4 主要测试 resize/scale。
5. **Zoom+true-context panel**：与 4 的放大 ROI 完全相同，把 sham thumbnail 换成真实全图 thumbnail；4→5 主要测试补回 global context 是否救回阴性判断。

`phase-scrambled thumbnail` 保持低级频谱和直方图但破坏解剖结构；若实现太复杂，冻结 Gaussian-blurred thumbnail 作为 fallback，并明确它只能排除一部分低级视觉差异。

### 5.3 三种 prompt；每个 render 像素逐 bit 相同

- **Neutral**：现有 `Does this chest X-ray show [finding]?`
- **Random provenance**：说明该视图由程序在查看异常之前均匀随机产生，选择行为本身不构成临床证据；仍问同一 finding。
- **Suspicious provenance**：等长地说明该视图是在发现 `[finding]` 可疑区域后被选中；仍问同一 finding。

所有条件继续读取 supported/refuted/undetermined 三个固定答案 token 的 logits，不使用生成文本 parse。

为排除“random 说明只是让模型整体更保守”，同一组三种 prompt 必须同时作用于 Full 和 crop renders。定义 crop 特异的 provenance interaction：

\[
\Gamma_{\rm random}
=
[m(\text{crop},q_{\rm neutral})-m(\text{crop},q_{\rm random})]
-[m(\text{full},q_{\rm neutral})-m(\text{full},q_{\rm random})].
\]

背景解释：第一项是 random-provenance 说明在 crop 上降低了多少阳性 margin；第二项扣掉同一句话在完整图上造成的普通保守偏移。这个 difference-in-differences 是标准统计对照，不作为理论贡献。

### 5.4 预注册判定

先按 image 做 5,000 次配对 bootstrap；所有阈值在 development 冻结。

**Observation-policy pragmatics GO 必须同时满足：**

1. 全阴性 confirmation 上 `Gamma_random > 0.25 logit`，且 95% CI 下界 > 0；
2. random provenance 相对 neutral 使 crop FP 绝对下降至少 10pp，但 Full FP 变化不超过 3pp；二者差的 95% CI 排除 0；
3. clear-positive bbox crops 的 recall 下降不超过 1pp，排除统一说 No；
4. 在 Hulu 原样复现，且多数合格 findings 同向。

**竞争机制路由：**

- `Zoom+true-context` 比 `Zoom+sham` 降低 FP ≥10pp、而 provenance interaction 失败：结论是 **context loss**，关闭 pragmatics 主线；
- `Zoom+sham` 比 `Native+sham` 提高 FP ≥10pp：结论是 **resize/scale OOD**；
- random prompt 对 Full 与 crop 同幅降低：结论是普通 **criterion shift / prompt calibration**，关闭；
- 只有 suspicious prompt 抬高 FP：属于已知 prompt-induced hallucination 医学复现，关闭；
- 所有 render 都高 FP 且无可分离效应：记录为 partial-view ambiguity，不继续命名。

即使 GO，也只确认一个机制；必须再用自然图像局部 VQA、至少一个通用 VLM、不同 crop 比例和 fixed-K OE 证明普遍性，才有论文主线资格。

## 6. 若 GO，自然推出的极简 training-free 修复

### 6.1 Truthful Provenance Neutralization

任何程序生成或模型选择的 crop 都附上一句真实来源说明：

> “This view was produced by an automated search. Its selection is not independent clinical confirmation; use only visible evidence.”

它不改模型、不需要小模型，也不删除 claim。优点是简单、通用、与机制严格对应；缺点是本质仍是 prompt engineering，单独不够 ICLR。

### 6.2 Context-preserving rendering

不要把 ROI 伪装成一张新的完整胸片，而把放大 ROI 与原始全图共同呈现，或使用 blur-outside-ROI 保留解剖坐标。若 true-context panel 同时降低 FP、保持小病灶 recall，这会是更可靠的默认实现。

但该方法与 Fine-Grained Visual Prompting 的 blur reverse mask、local/global visual prompting、HALC/AGLA 高度相邻；不能包装成新算法。论文价值只能来自一个更高层、跨任务成立的规律：**VLM 诊断不仅条件于看到什么，也条件于模型认为“为何有人让它看到这些”。**

不采用“两个 prompt margin 相减”作为主修复，因为本项目已多次发现 response change 不等于 clinical evidence；这种 subtraction 会重回 NCD/ISD 的失败模式。

## 7. 严格创新性与 ICLR Oral 判定

| 维度 | 分数（0–3） | 证据 |
|---|---:|---|
| Importance | 3 | crop/zoom 是大量 training-free hallucination 和 active-perception 方法的共同操作，当前随机 crop 已造成 +54.8pp FP |
| Mechanistic value | 2 | 四机制有可区分预测与同像素干预，但语用项尚无正实验证据 |
| Novelty space | 1 | visual prompting、PIH、BCEA、informative observation 已分别覆盖所有组件；只剩联合分离规律 |
| Executability | 3 | 现有 62 图、crop 和单步 scorer 可直接复用，首次致死实验 <2k scorings |

默认加权分为 `2.3/3`：**值得一次低成本致死实验，不足以作为当前 ICLR Oral idea。**

严格结论：

1. 现象强、问题自然，比继续修 region-count search law 更值得解释；
2. 但最可能的简单解释是 context removal + resize OOD，且 FGVP 已提供保留上下文的成熟对照；
3. provenance interaction 若失败，应立即关闭“observation policy is a prompt”，不要把 8.1%→62.9% 重新命名；
4. provenance interaction 若跨两模型成立，它可成为一篇有启发性的机制论文核心之一，但仅靠一句 provenance prompt 修复仍达不到 Oral；还需要证明该规律统一解释 crop、zoom、retrieved views、医生选图和 tool-use acquisition，并能预测何时 local enhancement 降低或增加 hallucination。

## 8. 已核实参考文献

1. Yang et al. *Fine-Grained Visual Prompting*. NeurIPS 2023. <https://proceedings.neurips.cc/paper_files/paper/2023/hash/4e9fa6e716940a7cfc60c46e6f702f52-Abstract-Conference.html>
2. Chen et al. *HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding*. ICML 2024. <https://proceedings.mlr.press/v235/chen24bi.html>
3. Woo et al. *Black-Box Visual Prompt Engineering for Mitigating Object Hallucination in Large Vision Language Models*. NAACL 2025. <https://aclanthology.org/2025.naacl-short.45/>
4. Park et al. *SECOND*. ICML 2025. <https://arxiv.org/abs/2506.08391>
5. Kogilathota et al. *Mitigating Hallucinations in Vision-Language Models through Image-Guided Head Suppression*. EMNLP 2025. <https://aclanthology.org/2025.emnlp-main.631/>
6. McGee et al. *Multimodal Pragmatic Inference in Vision-Language Transformers*. CogSci 2025. <https://escholarship.org/uc/item/5pf870ff>
7. Rudman et al. *Mechanisms of Prompt-Induced Hallucination in Vision–Language Models*. ACL 2026. <https://aclanthology.org/2026.acl-long.1941/>
8. Xu et al. *Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable Vision-Language Models*. 2026. <https://arxiv.org/html/2606.16667v4>
9. Sisk et al. *Informative presence and observation in routine health data: A review of methodology for clinical risk prediction*. 2021. <https://pmc.ncbi.nlm.nih.gov/articles/PMC7810439/>
10. Davis et al. *A framework for understanding label leakage in machine learning for health care*. 2023. <https://pmc.ncbi.nlm.nih.gov/articles/PMC10746313/>
11. Zhang et al. *Mitigating Bias in Radiology Machine Learning: 2. Model Development*. 2022. <https://pmc.ncbi.nlm.nih.gov/articles/PMC9530765/>
12. Lin et al. *Shortcut Learning in Medical Image Segmentation*. MICCAI 2024. <https://papers.miccai.org/miccai-2024/695-Paper0423.html>

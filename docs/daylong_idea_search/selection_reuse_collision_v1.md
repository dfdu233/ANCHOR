# Selection–Reuse Inflation / Select–Validate–Calibrate：敌对文献碰撞 v1

> 检索日期：2026-08-12。仅使用已打开核实的论文主页、正式会议页面或全文；“未检索到”不等于不存在。本文不占 GPU，不修改 baseline。

## 1. 冻结问题

我们审查的窄命题是：

> VLM 先在许多区域/尺度中挑出让 claim 得分最高的区域，再用同一个模型或高度相关的内部信号验证、放大并解码，是否会把选择噪声误当成视觉证据，制造 selection–reuse inflation？

研究问题：

1. 是否已有工作明确提出“自适应选区改变校准/交换性，必须把选择策略纳入校准”？
2. AGLA、HALC、SECOND、VGA、Perception Magnifier 等是否已经占据“选局部区域再增强/解码”的方法空间？
3. 若仍有缝隙，它足以支撑 ICLR Oral，还是仅是一条机制审计规律？

## 2. 严格结论

**当前 Select–Validate–Calibrate 作为主方法应 Reject and Pivot。**

致命碰撞是 2026 年的 **BCEA**。它已经明确证明：若先用基础分数校准、测试时再自适应 crop/zoom/claim-specific intervention，会破坏交换性并使实际风险超出目标；修复方法正是把**完整 acquisition policy 纳入 post-acquisition score 后重新校准**。这与“选择后的响应不能直接当新证据，必须在选择后校准”在问题、机制、修复三层都高度同义。[Xu et al., 2026](https://arxiv.org/abs/2606.16667)

因此：

- “adaptive selection 后重校准”已被直接占据；
- “独立 selector / verifier”属于经典 sample splitting / cross-fitting，并有 HALC 的外部 detector、VGD 的轻量 verifier 等邻近实现；
- “最大响应存在 winner's curse”是经典 selective inference；
- sqrt(2 log M)、conformal、DID、普通 cross-fitting 都不能作为核心数学贡献。

尚存的窄缝隙是：**系统证明当前局部幻觉缓解方法会选择并再次放大同一视觉噪声，而且该效应以最终病例级错误、而非仅内部 max-score 的形式跨方法复现。** 这是待验证的机制规律，不是现成的新算法，当前不具备 Oral-ready 程度。

## 3. 最直接的同义碰撞

### 3.1 BCEA 已覆盖 Select–Validate–Calibrate

BCEA 研究 answer / abstain / acquire more visual evidence 三态决策，acquisition 明确包括 zoom、crop 和 claim-specific visual intervention。它指出：在 acquisition 前校准阈值、测试时换用更高的 post-acquisition score，会使 calibration 与 test 不再 exchangeable；把完整 acquisition policy 当作 score function 的一部分，并在 post-acquisition score 上校准，可恢复有限样本保证。[论文全文](https://arxiv.org/abs/2606.16667)

| 候选术语 | BCEA 对应对象 | 碰撞 |
|---|---|---:|
| Select | borderline claim 触发 model-guided crop / structured intervention | 直接 |
| Reuse inflation | naive acquisition 破坏 exchangeability，风险超标 | 直接 |
| Validate | post-acquisition evidence score | 直接 |
| Calibrate | 整个 acquisition policy 内嵌进 score 后重校准 | 直接 |
| “何时 acquisition 真有用” | 固定风险下 coverage 改善当且仅当 ROC 改善 | 直接 |

若论文只说“把区域选择纳入校准”，审稿人可将其概括为 BCEA 的医学应用。

### 3.2 一般统计问题也早已存在

Liang、Zhu 与 Barber 研究了用同一 hold-out 集选择最小 conformal set 的模型、再用它校准导致 coverage loss，并给出 post-model-selection conformal 方法。[Liang et al., 2024/2026](https://arxiv.org/abs/2408.07066)

在线场景的 CAP 直接把问题称为 **Calibration after Adaptive Pick**，给出 selection-conditional coverage / false coverage-statement rate 控制。[Bao et al., JMLR 2025](https://www.jmlr.org/papers/v26/24-0452.html) 另有工作指出错误的 selective-conformal 策略会破坏 selected test datum 与 calibration data 的 exchangeability。[Sale & Ramdas, 2025](https://arxiv.org/abs/2503.16809)

它们不等价于医疗局部视觉机制，但封死了把“post-selection 校准”包装为新数学的路径。

## 4. 局部选择/增强的方法空间已拥挤

| 工作 | 选择对象 | 后续干预 | 与候选关系 |
|---|---|---|---|
| **HALC, ICML 2024** | Grounding detector 给局部上下文，并采样多个 FOV | focal-contrast decoding + beam | 已占“自动选局部区域并对比解码”；selector 可来自外部 detector。[论文](https://proceedings.mlr.press/v235/chen24bi.html) |
| **AGLA, CVPR 2025** | image–prompt matching 找 prompt-relevant local features | 原图 global view 与 local augmented view 做 logit calibration | 已占“局部选择 + 全局/局部融合”。[论文](https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html) |
| **SECOND, ICML 2025** | attention/entropy 引导 coarse-to-fine patch 累积 | 多阶段 expert–amateur contrastive decoding | 已占“精细化选择 + 多尺度 + 解码”；作者也承认 attention selection 不可靠时会失败。[论文](https://arxiv.org/abs/2506.08391)，[代码](https://github.com/AIDASLab/SECOND) |
| **Perception Magnifier, ACL 2026** | 多轮 attention map 选区并扩大覆盖 | structure-preserving magnification 后重解码 | 已占“迭代搜索并放大验证”。[论文](https://aclanthology.org/2026.acl-long.2059/) |
| **VGA, CVPR 2026** | visual-token logit 的 VSC；object score 明确用 patch-wise max | 用同一 grounding 引导 decoder attention | 与 selection–reuse 风险最接近：max_i c_i(o) 后继续用该 grounding 干预。[论文](https://arxiv.org/abs/2511.20032) |
| **BCEA, arXiv 2026** | claim-specific acquisition / model-guided crop | 完整 policy 纳入 post-acquisition calibration | 直接占据“选后校准”。[论文](https://arxiv.org/abs/2606.16667) |
| **VGD, arXiv 2026** | 每个正在形成的 object mention | 轻量 verifier 后 rollback、屏蔽同义词、局部重生成 | 已占“selective verify then correct”，且评价 coverage 与长度。[论文](https://arxiv.org/abs/2607.27823) |
| **SPIN, EMNLP 2025** | 每个 query token 动态选 vision-attending heads | 抑制低视觉注意头 | 占动态选择再干预的 head 路径，但不处理选区复用偏差。[论文](https://aclanthology.org/2025.emnlp-main.631/) |

### FOCUS 名称核查

检索到的正式 NeurIPS 2025 **FOCUS** 是 referential segmentation 驱动的交互式编辑统一模型，不是 hallucination/local-evidence mitigation 方法。[FOCUS, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3028874b502e8f088f7f5c47baa6d36b-Abstract-Conference.html) 若项目材料把它列为局部幻觉缓解前作，应核对是否与其他同名工作混淆。2026 年另有 *Focus Matters: Phase-Aware Suppression* 预印本，但不是该 NeurIPS FOCUS。[arXiv:2604.03556](https://arxiv.org/abs/2604.03556)

## 5. 医学与小病灶本身也不新

- ICCV 2025 的 *Seeing the Trees for the Forest* 已指出医学 VLM 的背景 token 高范数、global token 对局部病灶表征不足，并用 disease-aware map 放大病灶、压制背景。[Huy et al., 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Huy_Seeing_the_Trees_for_the_Forest_Rethinking_Weakly-Supervised_Medical_Visual_ICCV_2025_paper.html)
- AAAI 2026 的 ARCD 使用 anatomy mask，在 token、attention、logit 三层进行区域引导 contrastive decoding，覆盖 CXR、CT、MRI 与超声。[Liang et al., 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37620)
- IJCAI 2025 已系统显示 small-object fine-grained hallucination 更严重，并测试 resize 等 training-free 方法。[Sun et al., 2025](https://www.ijcai.org/proceedings/2025/212)

所以“只看病灶区域”或“小病灶需要局部放大”不足以构成新问题。唯一可能的新对象是：**局部搜索本身会制造假证据。**

## 6. 尚存缝隙

### 6.1 未检索到完全等价的系统规律

未找到一篇工作同时完成：

1. 将 AGLA / HALC / SECOND / VGA / magnification 统一成 selector → selected view → correlated verifier；
2. 在 **all-negative claims** 上直接测 selector–verifier noise correlation；
3. 只增加“无新临床信息”的候选区域/尺度，因果证明最终 FP 随搜索空间增长；
4. 在 fixed-length / fixed-K 开放生成中证明修正不是靠少说、拒答或统一否定。

这是**机制审计缝隙**，不是算法缝隙。

### 6.2 最小数学命题：相关选择—验证偏差身份

背景：一张没有 claim \(c\) 的图像被划为 \(M\) 个候选区域。对区域 \(j\)：

- \(S_j\)：selector 分数，如 attention、VSC 或 entropy-guided saliency；
- \(V_j\)：verifier/decoder 的局部 claim 分数；
- \(J=\arg\max_j S_j\)：被选中的区域。

若负例上的 \((S_j,V_j)\) 独立同分布、标准化后联合高斯，相关系数为 \(\rho\)，则高斯条件期望满足

\[
\mathbb E[V_j\mid S_j]=\rho S_j.
\]

因此

\[
\boxed{\mathbb E[V_J]=\rho\,\mathbb E[\max_j S_j]}.
\]

含义：所有区域都没有病灶时，只要 selector 与 verifier 共享正相关噪声（\(\rho>0\)），选择最像病灶的区域后再用相关通路验证，平均 verifier 分数仍会被抬高；条件独立（\(\rho=0\)）时该项消失。

这比只写 sqrt(2 log M) 更适合实验，因为它指出可被干预的变量是 **selector–verifier noise coupling \(\rho\)**。但它仍是经典高斯回归与 selective inference 的直接推论，**只作诊断工具，不作理论贡献**。

论文级命题必须来自真实规律：局部增强造成的病例级错误增量，是否被 \(\rho\) 与有效搜索规模跨模型、跨方法共同预测，并独立于长度、Yes-rate 和架构。

### 6.3 一个合理但不新的聚合 baseline

若每个区域分数可校准成 likelihood ratio \(L_j\)，并假设阳性图像至多有一个未知位置病灶、位置先验均匀，则整图的边际 likelihood ratio 是

\[
L_{\mathrm{image}}=\frac{1}{M}\sum_{j=1}^{M}L_j,
\]

而不是 \(\max_j L_j\)。在全负例下 \(\mathbb E_0[L_{\mathrm{image}}]=1\)，不会因候选数增加而自动累积证据。但 average likelihood-ratio detection 与 scan statistic 的比较是经典工作，不能当新理论。[Chan & Walther, 2011/2013](https://arxiv.org/abs/1107.4344)

它仅应作为强 baseline：若 max/top-k local enhancement 被边际证据聚合稳定击败，才支持“问题来自搜索复用”。

## 7. 三个致死实验

### A. Null-view multiplication：唯一第一门

**目的**：只增加搜索机会，不增加临床证据，观察最终幻觉是否增加。

- 数据：VinDr 某 finding 的 0/3 reader-negative；3/3 clear-positive 作对照。
- 操作：同一图像生成 \(M\in\{1,4,9,16,25\}\) 个 area/scale-matched 候选；新增候选来自与目标病灶无关区域，只做轻微 label-preserving jitter。
- 方法：原生、VGA-like max VSC、SECOND-like selection、AGLA/HALC 忠实端口中至少三种。
- 主终点：最终 claim FP / BAcc；同时记录 selector max、verifier margin、负例 \(\rho\)。
- 对照：随机区域、固定区域、同计算量不选择、候选顺序打乱、完全相同区域重复。
- **GO**：至少 2 模型、3 方法中的多数显示 FP 随 \(M\) 单调上升；\(M=25\) 相对 \(M=1\) 的绝对 FP 增量 ≥5pp，image-bootstrap 95% CI 排除 0；不能被普通 Yes-rate 移动解释。
- **NO-GO**：只有内部 max/margin 变化而最终错误不变，或只在一个模型/方法出现，即关闭。

### B. 只切断 coupling，不改变像素预算

**目的**：区分“局部像素提供证据”与“选择噪声被复用”。

- **reuse**：同一模型/同一信号族选择并验证。
- **cross-fit**：预先冻结、不可读取 verifier 输出的 selector 选同大小区域，再由目标 VLM 验证；可用外部小视觉模型或两个预注册低相关 encoder/view，禁止事后挑 split。
- **sham**：计算量和区域大小相同，置换 selector 分数与 region ID。
- 主终点：0/3 negatives 上的 \(\rho\)、selected margin、最终 FP；3/3 positives 上的 recall。
- **GO**：cross-fit 同时显著降低 \(\rho\) 和 FP（FP 相对下降 ≥20%），clear-positive recall 下降 ≤1pp，且 2 模型复现。
- **NO-GO**：仅内部相关或 Yes-rate 下降，最终 FP 不变；或 FP 改善由 recall 下降解释。

### C. fixed-K OE 真实效用门

**目的**：排除删 claim、缩短回答、统一阴性带来的伪改善。

- 任务：VinDr OE abnormality listing；每张图 positive claim 数 \(K\) 与 greedy 相同。
- 对比：greedy、原始 local enhancement、切断 coupling 版本；matched length、matched \(K\)、matched refusal。
- 终点：positive-content hallucination、omission recall、reader-vote Brier、claim 数和长度；无医生评审时只称 benchmark proxy。
- **GO**：2 模型上 hallucinated positive claims 相对下降 ≥20%，omission 不增加，Brier 相对改善 ≥5%，paired bootstrap CI 排除 0。
- **NO-GO**：仅 CE 有效，或 OE 增益来自更多阴性/hedge，不能称 hallucination mitigation。

## 8. ICLR Oral 判断

| 版本 | 新颖性 | 判定 |
|---|---:|---|
| winner's curse in region max | 低：经典 selective inference | 不投 |
| 选择后 conformal 校准 | 无：BCEA 直接碰撞 | 关闭 |
| 独立 selector / verifier | 低：sample splitting + HALC/VGD 邻近 | 不足 |
| 医疗小病灶局部放大 | 低：ICCV 2025/ARCD/IJCAI 2025 已占 | 不足 |
| 跨方法 selection–reuse 病例级错误定律 + null-view 因果干预 + fixed-K OE | 中等、未验证 | 只值得跑一次致死实验；更像机制/评测主会 |
| 进一步发现跨分辨率、跨模型相变，且 coupling 统一预测 FP/FN 边界 | 潜在较高 | 只有强结果、规律普遍时才可讨论 Oral |

**执行建议**：不要把 Select–Validate–Calibrate 当算法主线放量。只运行实验 A 作为最低成本生死门；A 不过即关闭。A 通过后才做 B，B 通过后才做 fixed-K OE。诚实定位应是“局部增强是否在选择自己的噪声？”，而不是“发明 post-selection 校准”。

## 9. 已核实参考文献

1. Xu et al. *Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable Vision-Language Models*. 2026. <https://arxiv.org/abs/2606.16667>
2. Liang, Zhu, Barber. *Conformal Prediction After Data-Dependent Model Selection*. 2024–2026. <https://arxiv.org/abs/2408.07066>
3. Bao et al. *CAP: A General Algorithm for Online Selective Conformal Prediction with FCR Control*. JMLR 2025. <https://www.jmlr.org/papers/v26/24-0452.html>
4. Sale, Ramdas. *Online Selective Conformal Prediction: Errors and Solutions*. 2025. <https://arxiv.org/abs/2503.16809>
5. Chen et al. *HALC*. ICML 2024. <https://proceedings.mlr.press/v235/chen24bi.html>
6. An et al. *Assembly of Global and Local Attention*. CVPR 2025. <https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html>
7. Park et al. *SECOND*. ICML 2025. <https://arxiv.org/abs/2506.08391>
8. Zhao et al. *Vision-Guided Attention*. CVPR 2026. <https://arxiv.org/abs/2511.20032>
9. Yang et al. *Verifier-Guided Decoding*. 2026. <https://arxiv.org/abs/2607.27823>
10. Kogilathota et al. *Image-Guided Head Suppression*. EMNLP 2025. <https://aclanthology.org/2025.emnlp-main.631/>
11. Huy et al. *Seeing the Trees for the Forest*. ICCV 2025. <https://openaccess.thecvf.com/content/ICCV2025/html/Huy_Seeing_the_Trees_for_the_Forest_Rethinking_Weakly-Supervised_Medical_Visual_ICCV_2025_paper.html>
12. Liang et al. *Anatomical Region-Guided Contrastive Decoding*. AAAI 2026. <https://ojs.aaai.org/index.php/AAAI/article/view/37620>
13. Sun et al. *Understanding Visual Detail Hallucinations*. IJCAI 2025. <https://www.ijcai.org/proceedings/2025/212>
14. Chan, Walther. *Detection with the Scan and the Average Likelihood Ratio*. 2013. <https://arxiv.org/abs/1107.4344>

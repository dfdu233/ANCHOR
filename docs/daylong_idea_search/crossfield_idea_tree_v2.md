# 医学 VLM 幻觉：跨领域机制树 v2

日期：2026-08-12  
任务边界：只用现有缓存或单卡 30 分钟以内的 pilot 设计候选；本轮不占 GPU、不改 baseline。  
检索口径：2023--2026 的 VLM/医学近邻优先；经典数学只作背景，绝不把标准定理包装成原创。

## 0. 研究问题与证据约束

本轮回答三个问题：

1. 哪个跨领域对象能同时解释“小病灶更易漏”和“许多 mitigation 只移动 Yes-rate”这两个看似相反的现象？
2. 哪个对象能自然导出一个 **training-free、简洁、不会靠少说获益** 的方法？
3. 哪些漂亮公式其实只是换名，或已经被本地结果直接证伪？

任何候选都必须服从以下本地事实，而不是重新讲一遍已关闭的故事：

| 已有事实 | 对新候选的约束 |
|---|---|
| 两模型、开发/确认集都出现病种内 bbox 面积--final margin 正相关；确认集 Huatuo/Hulu 相关分别约 0.323/0.415 | 可以研究局部、稀疏、尺度与搜索；不能直接声称 bbox patch 已被模型正确定位 |
| final 之上的全局视觉增量 CI 跨 0；非最终层 macro AUROC 比 final 低 0.109 | 关闭“全局 pooled feature 已藏有额外答案”和“早层普遍更好” |
| lesion erase drop 0.029，CI `[-0.049,0.109]`；mirror/relocation 出现反向或过冲 | 任意 mask/paste 的 response 不能自动叫临床 evidence |
| 风格翻转率低于 5%，style drift error AUROC 0.425，final margin 0.798 | 关闭 style/DG invariance 主线；跨视图只能当控制，不能复活风格故事 |
| NCD/common-mode 双中心化 AUROC 0.736→0.655 | 关闭“再做一次去均值/quotient 就得到特异证据” |
| patient-matched RAG 不超 placebo；head suppression/LET 常统一推高某一答案 | 外部 context 或干预引起响应不等于病例级信息 |
| LLaVA CXR-VisHal 上 strict 与 official proxy 的方法排序有 5/10 pair 反号 | 所有新方法必须同时报告 strict truth、工作点、长度/parse；不能只挑一个 judge |

这里采用三个构造路径，但只转移其思考方式：

- **Chinchilla 式资源比**：把“局部方法是否有效”改写为信号强度、病灶面积、搜索空间和验证相关性的相变问题；
- **SigLIP 式去偶然耦合**：质疑“用同一个响应先选区域、再证明该区域有证据”这一偶然耦合；
- **Model Collapse 式玩具动力学**：先用可解析 null 解释 FP 如何随搜索增长，再问真实 VLM 是否遵循该规律。

## 1. 候选总表

`保留` 只表示值得一个致死实验，不表示已达到 ICLR；`关闭` 表示现有本地结果已经反驳精确机制；`降级` 表示可作评测或控制，不能作主算法。

| 排名 | 候选 | 跨领域基本对象 | 最接近碰撞 | ≤30min 致死门 | 决定 |
|---:|---|---|---|---|---|
| 1 | Selection--Reuse Inflation / Search-Calibrated Evidence | selective inference、winner's curse、极值统计 | SECOND、AGLA、CEBC；尚未检索到同一 selection--reuse law | patch cache 上扩大 claim×region 搜索，检验 null response 与最终 FP 是否随搜索增长 | **保留，第一优先** |
| 2 | Anatomy-Conditional Randomization | conditional randomization test、有效视觉 null | VCD/SECOND 的扰动；NeurIPS 2025 causal image editing | 32 个 vote-0/3 图，解剖位置匹配替换 vs mean-fill，检验 null rank 与阳性 power | **保留，高风险** |
| 3 | Cross-Fit Visual Confirmation | data splitting、post-selection inference | LENS/VTI/跨视图一致性；未见“先选后独立验”的同机制方法 | 用两套 patch response 交叉选/验，检验 inflation 是否随 selector--validator 相关性衰减 | **保留，依赖可用第二响应** |
| 4 | Open Report as Online FDR | online testing、e-values、alpha wealth | ConfLVLM、CEBC、Principled Detection via Multiple Testing | 冻结 ontology 大小 4/8/16/32，fixed output K，检验 raw FP 是否随候选数增长 | 降级；现象可测，方法碰撞强 |
| 5 | Anytime Visual Evidence | SPRT、test martingale、confidence sequence | BCEA、adaptive zoom、sequential diagnosis | patch cache 按冻结顺序累计，比较固定风险下平均读取 patch 数 | 降级；很可能只是校准/早停 |
| 6 | Observation--Computation Frontier | Blackwell order、value of information | BCEA、active perception、多视图 CXR | IU-Xray 64 study：第二真实 view vs 同图 beam vs wrong-study view | 条件保留；逐 view 真值不足 |
| 7 | Anatomical Unbalanced OT Residual | optimal transport、normal prototype anomaly | 2025 OT anomaly detection、医学 registration | patch cache 上当前图到 vote-0 正常 bank 的 spatial OT cost 是否超 final margin + max/mean | 低优先；易退化为 anomaly detector |
| 8 | Persistent Evidence Islands | persistent homology、scale-space | TOHA、PHG-Net、multiscale scan | patch map 的 component lifetime 是否在控制 max/scan 后仍 +0.02 AUROC | 降级；大概率被 scan 吸收 |
| 9 | Block-Jackknife Evidence Stability | algorithmic stability、influence | VTI、LENS、LeHaCE、FOCUS | 删除同面积 patch block 后，claim score 方差能否超 final margin 预测错误 | 降级；style stability 已失败 |
| 10 | Visual MDL / Diffusion Bayes Factor | MDL、likelihood ratio、score matching | DEEM、DeGF、diffusion classifier | 16 图×正反 claim 的冻结扩散重建/score gap；若不超 final margin 即停 | 关闭优先级；碰撞与算力都重 |
| 11 | Evidence Transport Equivariance | causal transport、commuting intervention | causal counterfactual editing、VCE | 复用 relocation/erase artifact 检验病灶移动是否只移动对应 claim | **关闭**：已有 overshoot/错方向 |
| 12 | Fisher--Rao Evidence Quotient | information geometry、nuisance quotient | VTI、latent steering；代数上接近 NCD/ISD | 复用 common-mode artifact 看 quotient 是否增量 | **关闭**：AUROC 0.736→0.655 |
| 13 | Counterexample Image Retrieval | contrast sets、case-based falsification | Visual Evidence Prompting、MARINE、image RAG | BioMedCLIP cache 中找最近相反标签图，差值能否超 final margin 与 shuffled neighbor | 低优先；易成为普通 KNN/RAG |
| 14 | Diagnostic Channel Coding | rate--distortion、successive refinement | WindowNet、多窗 CXR、多尺度 SECOND | 复用 5-render cache，等 token budget 下多窗互补是否超最佳单窗 | **关闭当前版本**：fingerprint delta -1.25pp |
| 15 | Credal Clinical Decoding | partial identification、imprecise probability | CEBC、selective prediction、MRIP 类问题定义 | 用 reader votes 构造可识别区间，检查收益是否只来自扩大 uncertain | 只作问题定义；不是 mitigation |

## 2. 逐候选敌对审计

### C1. Selection--Reuse Inflation：视觉搜索税

**自然现象。** 小病灶越小，模型 margin 越低；于是局部方法会搜索更多区域、尺度或 claims 来找证据。但在完全阴性的图上，候选越多，最大的随机响应也越大。局部增强可能把这个“噪声赢家”再次放大，这能同时解释 recall 上升和 FP 上升。

**一句话方法。** 允许 VLM 搜索局部区域，但只有在完整 claim×region×scale 搜索流程于 vote-0 图的经验 null 中仍显著时，才允许该局部响应改变原始输出。

**数学背景。** 令 `A_i` 是第 `i` 个候选区域的选择分数，`B_i` 是选择后用于确认的响应，`M` 是候选数，`rho` 是 `A_i,B_i` 的相关性。若 absent claim 下每对近似双变量高斯，选择 `J=argmax A_i` 后：

\[
E[B_J]=\rho E[\max_{i\le M}A_i]\approx \rho\sqrt{2\log M}.
\]

直观上，即使每个固定区域的确认响应均值为 0，只要“选”和“验”共享噪声 (`rho>0`)，被选中的确认响应就会随搜索规模增长。这是经典 winner's curse / post-selection inference，不是我们的定理；参考 [Fithian, Sun & Taylor, 2014](https://arxiv.org/abs/1410.2597)。真正可能新的，是该项能否统一预测 VLM 局部 mitigation 的 FP。

**碰撞。** [SECOND, ICML 2025](https://proceedings.mlr.press/v267/park25c.html) 和 [AGLA, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html) 都会选局部再增强；[CEBC, ACL 2026](https://aclanthology.org/2026.acl-long.2142/) 已用 conformal detector 约束输出。剩余 delta 只能是 **selection--reuse inflation 的跨方法规律**，不是 max、scan 或 conformal 本身。

**致死门。** 等待现有 `patch_scores_*`，只做 CPU：在所有目标 claims 均为 0/3 的图上嵌套扩大 `M=claim数×区域数×尺度数`；要求 Huatuo/Hulu 的 raw max、selected validation response 和最终 positive margin 都单调增长，bootstrap CI 排除 0，并由 `rho E[max A]` 至少方向性预测。若增长不传入最终 margin/FP，主线立即关闭。

### C2. Anatomy-Conditional Randomization：有效视觉 null 不是噪声图

**自然现象。** VCD、mean-fill、mirror、relocation 都能显著改变模型，却没有稳定的 claim-specific 正确方向；这提示失败可能不是“对比不够精细”，而是 null image 本身不在临床图像条件分布中。

**一句话方法。** 不用噪声或灰图制造负视图；固定其余解剖，只从同位置、同体位、同 nuisance stratum 的 vote-0 影像中抽取可交换替代区域，原图响应只有比这些条件重采样更极端才算证据。

**数学背景。** 对待检区域 `S`，保持图像其余部分 `X_-S`，从无该 claim 时的条件分布抽样 `X_S^(1),...,X_S^(B)`。若原图在 null 下与这些样本可交换，则任意黑盒分数 `T` 的 rank p-value

\[
p=\frac{1+\sum_{b=1}^{B}\mathbf 1[T(X^{(b)})\ge T(X)]}{B+1}
\]

在 null 下不会系统性过小。背景是 Conditional Randomization Test；有效性来自可交换性，不来自 VLM 分数准确。困难也恰在这里：位置匹配 patch 若有接缝或不保留 anatomy，就没有保证。

**碰撞。** [Counterfactual Image Editing, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/54e1419b1cbca8ada96a87c96567e954-Abstract-Conference.html) 已研究因果一致图像编辑；VCD/SECOND 已做扰动对比；BCEA 已指出自适应 acquisition 会破坏 exchangeability。未检索到把 **条件视觉 null 的有效性** 作为 VLM hallucination 对比解码核心对象的等价工作，但这不是新统计定理。

**致死门。** 单卡 32 图（16 vote-0、16 vote-3），每图仅 2 个 anatomy-matched replacements；与 mean-fill、同面积随机位置、wrong-patient replacement 比。先要求 vote-0 rank 接近均匀且 replacement 不造成统一 Yes shift，再要求 vote-3 原图在 rank 上显著更极端。任一 manipulation check 失败即停，不上 diffusion 生成器救结果。

### C3. Cross-Fit Visual Confirmation：选证据和验证证据必须分账

**自然现象。** “response is not evidence”最直接的统计原因，是同一相关响应被使用两次：先挑最强 patch，再拿它证明自己正确。

**一句话方法。** 用响应 A 只负责选区域，用尽可能去相关的响应 B 只负责验证该固定区域；A 不再参与最终证据值。

**数学背景。** 沿用 C1 的 `A_i,B_i`。若在 null 下通过真正独立的 view、传感器或严格 sample splitting 使 `rho=0`，则 `E[B_argmax A]=0`，候选数不再制造正均值。若 `rho` 只是变小而非 0，膨胀应按 `rho` 连续下降。这一等式是标准 data splitting 直觉，不是新理论；新命题只能是 VLM 的 selector--validator correlation 是否实测决定 FP。

**碰撞。** [VTI, ICLR 2025 Spotlight](https://openreview.net/forum?id=LBl7Hez0fF) 使用视觉稳定性 steering；LENS 使用 medical counterfactual view stability；SECOND 跨尺度对比。它们没有明确把“选择与验证共享噪声”作为控制变量，但如果最终只剩跨视图一致性，就与 LENS 直接碰撞。

**致死门。** 在两模型 patch cache 完成后，A/B 可先由跨模型 response 或开发集冻结的特征半空间形成，不重新跑 GPU。比较 same-map、cross-map、random-validator 三档 `rho`；全阴性 selected-B inflation 必须随 `rho` 下降，而 vote-3 bbox enrichment 不消失。若 cross-fit 只是降低所有响应或 true power 同比例下降，关闭。

### C4. Open Report as Online FDR：每个新增 claim 都是一项发现

**自然现象。** 开放式 prompt 会生成更多 claims；现有评测又会因长度、parse policy 而改变方法排名。报告不是一个二分类，而是一串自适应提出的临床假设。

**一句话方法。** 把每个准备提交的 claim 视为在线检验，只在它拥有足够 evidence wealth 时提交，同时固定最终 claim 数或一换一，避免靠少说获益。

**数学背景。** e-value 是在 null 下期望不超过 1 的非负证据量；[e-BH](https://academic.oup.com/jrsssb/article/84/3/822/7056146) 可在任意依赖下控制 false discovery rate，[e-LOND, AISTATS 2024](https://proceedings.mlr.press/v238/xu24a.html) 处理按顺序到来的假设。难点不是公式，而是 VLM 自适应生成的 claim 是否有有效 e-value。

**碰撞。** [ConfLVLM, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.576/)、CEBC，以及 [Principled Detection of Hallucinations via Multiple Testing](https://arxiv.org/abs/2508.18473) 已占据风险控制/多重检验。没有新的 dependent adaptive-generation theorem 时，只能作 baseline。

**致死门。** 用 VinDr ontology 构造候选数 4/8/16/32，输出 positive 数固定；CPU 复用已存 margins，测试 raw FP 是否随 ontology 增大，而校准后 fixed-K precision/recall 是否同时改善。若候选数现象不存在，关闭；若只靠删 claim，判失败。

### C5. Anytime Visual Evidence：证据够了就停，不够就保留 unknown

**自然现象。** 小病灶需要看更细区域，清晰大病灶不需要扫描全图；固定算力对所有病例可能既浪费又诱发更多 search FP。

**一句话方法。** 按冻结顺序读取 patch/scale，累计支持与反驳的 evidence process；跨过风险边界即停止，否则到预算后输出 undetermined。

**数学背景。** sequential likelihood ratio 或 test martingale 在 null 下保持期望受控，因而允许任意停止；e-value 正是这类“可以边看边停”的证据。真正需要证明的是 patch 顺序下的增量能构成有效 process，而不是把相关 patch logits 相乘。

**碰撞。** [BCEA, 2026 arXiv](https://arxiv.org/abs/2606.16667) 已做 answer/abstain/acquire 的预算 conformal 决策，并明确讨论 acquisition 后重校准；adaptive zoom/active perception 更早已有。仅改成胸片 patch 顺序不新。

**致死门。** CPU 用 patch cache 比较 raster、coarse-to-fine、claim-score order；在相同 error 下是否减少 ≥30% patch，同时 vote-0 FP 不随最大预算增长。若不能明显优于一次性 calibrated scan，关闭。

### C6. Observation--Computation Frontier：再想一次还是再看一张

**自然现象。** 层融合、风格变换、RAG placebo 都没有创造病例证据；真正的新 lateral view 可能有同图后处理永远没有的信息。

**一句话方法。** 对低置信 claim 比较“再算一次同图”和“获取真实新观测”的单位成本收益，只在新观测预期信息增益超过 compute 时采集。

**数学背景。** Blackwell order 说：如果新观测可经随机降质变成旧观测，则对所有决策问题，新观测的最优风险不会更差；同图后处理受 data-processing 限制。但这是标准决策论，只能支撑边界表述。

**碰撞。** BCEA 已覆盖同图 crop/zoom；active perception 与多视图 CXR 已覆盖真实 acquisition。剩余 delta 是 **医学 VLM 幻觉的 observation--compute phase law**，不是 acquisition 策略本身。

**致死门。** 已准备 IU-Xray 64 study：view0 greedy、匹配 FLOPs 的同图 beam、真实 view1、wrong-study view1。真实 view1 必须比最佳同图 compute +3pp BAcc 或相对 Brier -5%，且超 wrong-study +2pp；否则关闭。共享报告非逐 view truth，即使过门也只能称 proxy。

### C7. Anatomical Unbalanced OT Residual：把正常解剖对齐后看剩余质量

**自然现象。** bbox 面积与 margin 相关，但对侧镜像不是干净 null；病人的正常解剖位置、心影和体位变化会淹没直接 patch 差。

**一句话方法。** 将当前图的 patch measure 以空间约束 optimal transport 对齐到 vote-0 正常原型；不能低成本运输掉的局部质量才作为异常 evidence。

**数学背景。** 普通 OT 在所有运输计划 `pi` 中最小化 `sum pi_ij * cost(i,j)`；unbalanced OT 允许质量生成/消失并支付惩罚，因此可区分位置变化与局部异常。数学和算法均成熟，不是贡献。

**碰撞。** [Local/Global Prototypes with OT, 2025](https://arxiv.org/abs/2508.12927) 已用于无监督异常定位，[Mass-Repulsing OT, TMLR 2025](https://openreview.net/forum?id=PPGJ3EvENv) 已定义 OT anomaly score，医学 registration 也大量使用 OT。若只是把 anomaly score喂给 VLM，属于应用。

**致死门。** patch cache 上 7 findings，各自以开发 vote-0 建 prototype，CPU 网格搜索一个固定空间 cost；确认集比较 `finding+final+mean+max` 与再加 OT residual。要求两模型 macro AUROC +0.02 且 CI 排除 0；否则关闭。

### C8. Persistent Evidence Islands：真病灶应跨阈值/尺度存活

**自然现象。** 真病灶可能形成一片连续但不极强的 patch；噪声最大值通常是孤立尖峰。

**一句话方法。** 对 claim patch field 从高阈值逐步降阈值，记录连通分量的 birth--death lifetime；只保留跨阈值、跨尺度持续的 evidence island。

**数学背景。** persistent homology 追踪拓扑结构在 filtration 中出现和消失的尺度；长 lifetime 表示结构不依赖某个任意阈值。它不会自动证明临床真实性。

**碰撞。** [TOHA, 2025](https://arxiv.org/abs/2504.10063) 已用 attention graph topology 检测 LLM hallucination，[PHG-Net, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Peng_PHG-Net_Persistent_Homology_Guided_Medical_Image_Classification_WACV_2024_paper.html) 已把 persistent homology 用于医学图像分类；multiscale scan 已覆盖“跨尺度区域”。

**致死门。** CPU 对同一 patch field 同时算 max、scan、H0 persistence；persistence 必须在 `final+mean+max+scan` 之上 +0.02 AUROC 且 bbox enrichment 更好。若被 scan 完全吸收，不独立立项。

### C9. Block-Jackknife Evidence Stability：删一个小块后证据是否崩溃

**自然现象。** 噪声赢家依赖单个 patch，而真实弥散证据可能对删除任一小块稳定；反过来，极小真病灶也可能不稳定，因此应出现面积相关边界。

**一句话方法。** 对每个 claim 做 leave-one-block-out，使用最坏 score drop 或 influence 分布区分单点伪证据与集体证据。

**数学背景。** algorithmic stability 衡量数据小变化导致输出变化的上界；jackknife 通过逐块删除估计影响。但“稳定”既可能来自真实冗余，也可能来自顽固语言先验，所以必须相对 image-null 和 final margin 做增量检验。

**碰撞。** [VTI, ICLR 2025](https://openreview.net/forum?id=LBl7Hez0fF) 已以视觉特征稳定性做 steering；LENS 已做 counterfactual-view stability；[LeHaCE, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c9b551a2e195a209fc0b280de2f7f781-Abstract-Conference.html) 控制描述长度带来的不稳定。风格稳定在本地已经失败。

**致死门。** patch cache 上无需重跑 VLM，模拟删除 2×2 score blocks，比较 instability 对错误的 AUROC；必须超 final margin +0.02 且 focal/diffuse 出现预注册交叉。否则作为 scan 的消融，不立项。

### C10. Visual MDL / Diffusion Bayes Factor：一个 claim 是否真的解释像素

**自然现象。** 语言模型可以流畅地说“有结节”，但若 claim 对图像生成分布没有解释力，它只是语言承诺而非像素证据。

**一句话方法。** 用冻结图像生成模型比较图像在 `claim` 与 `not claim` 条件下的码长/score，只有正 claim 更能压缩原图时才允许确定断言。

**数学背景。** 若生成模型给出 `p(x|c)`，log Bayes factor `log p(x|c)-log p(x|not c)` 衡量两种假设对像素的相对解释；MDL 把较短编码视为更好解释。扩散模型通常只能近似 likelihood/score，因此近似误差可能比病灶信号大。

**碰撞。** [DEEM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a8399aace3dfa6dfb8b635117748c561-Abstract-Conference.html) 已让 diffusion model 作为 LLM 的视觉感知器并减少 hallucination；DeGF 已把回答生成回图像再反馈；diffusion classifier 也已是标准用法。

**致死门。** 只跑 16 张 vote-0/3 图、正反两个固定 prompt、冻结 checkpoint；Bayes/score gap 若不超 final margin 的 AUROC，或正常 anatomy 的重建误差主导 finding 差异，就关闭，不扩大。

### C11. Evidence Transport Equivariance：病灶移动，证据也应移动

**自然现象。** 若一个局部 patch 真承载某 claim，把它移动到合理解剖位置时，对应证据应随之变化，而 unrelated claims 不应同步变化。

**一句话方法。** 只接受对 lesion intervention 近似 equivariant、对 placebo transport 不变的 claim response。

**数学背景。** equivariance 要求“先变换图像再测量”与“先测量再按规则变换证据”近似 commute；这是标准 group/causal transport 语言。

**碰撞。** NeurIPS 2025 causal image editing 已直接研究 counterfactual consistency；[VCE, 2026](https://arxiv.org/abs/2604.19412) 使用 contrastive visual perturbation 分解 hallucination subspace。

**致死门与决定。** 本地 lesion relocation 已 overshoot `+0.291`，mirror erase 方向也异常，普通 bbox erase 没有稳定 claim-specific effect。当前 intervention 不满足 manipulation check，**精确版本已经关闭**；只有获得可验证的因果一致医学编辑器才可重开，不能靠换变换续命。

### C12. Fisher--Rao Evidence Quotient：从分布几何中去 nuisance

**自然现象。** 多种干预产生 common-mode response，可能想把答案分布投影到与 nuisance 正交的 quotient manifold。

**一句话方法。** 在概率 simplex 的 Fisher--Rao 度量下移除图像无关切向量，仅保留 claim-specific geodesic displacement。

**数学背景。** Fisher information 为概率分布族提供局部 Riemannian metric，使“同样大小的概率变化”按统计可区分性衡量；投影/商空间本身是标准几何。若 nuisance 子空间由行列均值估计，这在代数上仍是 NCD/ISD 的双中心化换皮。

**碰撞。** VTI、latent steering、正交 editing 已高度拥挤；本地 common-mode canary 已给出 AUROC `0.736→0.655`。因此无论换欧氏还是 Fisher 度量，若输入对象相同都没有新的实证支柱。

**致死门与决定。** 复用同一 artifact，比较 Fisher-normalized 与普通 centered response；未达到 final margin +0.02 且两模型 CI 排除 0 就关闭。鉴于原对象已失败，**不值得新 GPU**。

### C13. Counterexample Image Retrieval：检索最相似的反例，而不是相似报告

**自然现象。** patient/text RAG 不超 placebo，可能因为它补充的是语言先验；真正有诊断价值的检索应是“长得几乎一样但 claim 相反”的图像反例。

**一句话方法。** 对每个草稿 claim，检索最近的 positive 与 negative 图像各一张，用当前图相对二者的距离差作为反证，不把任何报告文本喂给 VLM。

**数学背景。** contrast set 的核心是局部 decision boundary：如果最近相反标签样本比最近同标签样本更近，当前 claim 缺少局部支持。它是 metric learning/KNN 的标准思想，没有新定理。

**碰撞。** [Visual Evidence Prompting, ACL 2025](https://aclanthology.org/2025.acl-long.205/) 已用小视觉模型为 LVLM 提供细粒度证据；[MARINE, ICML 2025](https://proceedings.mlr.press/v267/zhao25j.html) 用外部 image-grounded guidance；反例 VQA 可追溯到 [Making the V in VQA Matter](https://arxiv.org/abs/1612.00837)。

**致死门。** 用已有 BioMedCLIP embedding 和 VinDr labels，CPU 做 image-disjoint counterexample retrieval；距离差必须在 final margin + finding identity 上 +0.02 AUROC，并超 shuffled/同来源匹配 placebo。否则归为普通 KNN baseline。

### C14. Diagnostic Channel Coding：固定视觉带宽怎样分配最值

**自然现象。** CXR 是灰度图，却常复制到 RGB；不同窗宽可能携带互补低对比结构，但历史 style/render ensemble 无增益。

**一句话方法。** 在固定 visual-token budget 下，将通道分给互补窗宽而非重复灰度，像 successive refinement 一样先传全局、再传残差。

**数学背景。** rate--distortion 研究在有限码率下最小化重建/任务失真；新贡献必须是可重复的医学 claim rate--distortion law，不是把三张窗图塞进 RGB。

**碰撞。** 多窗医学网络、WindowNet、SECOND 的多尺度视觉输入已经覆盖工程表面；本地 80-case five-render fingerprint 比 canonical 低 1.25pp，CI `[-7.50,+5.01]pp`。

**致死门与决定。** 复用五 render cache，在相同 token 数下比较 best single、mean ensemble、dev-frozen complementary pair；现有结果未显示 headroom，**当前版本关闭**。只有原始 DICOM bit-depth 的互补通道在两个模型显著通过才可重开。

### C15. Credal Clinical Decoding：图像不识别的世界不应被强行选一个

**自然现象。** reader disagreement 说明同一影像可能不能识别唯一临床状态；模型却输出单一确定句子。

**一句话方法。** 输出与所有可观测 reader distributions 相容的概率区间，而不是一点概率；只有整个区间都支持 claim 时才 definite。

**数学背景。** partial identification 不假设数据唯一决定参数，而给出 identified set；imprecise probability/credal set 表示多个仍与观测相容的分布。它能规范“不知道”，但不能创造遗漏病灶证据。

**碰撞。** CEBC/conformal selective prediction 已做风险边界；reader calibration 与不确定性预测已有大量医学工作。没有医生 review 时，自动 claim truth 也限制 OE 主张。

**致死门与决定。** 用 VinDr reader votes 构造区间并保持 matched coverage；若所谓改善完全来自扩大 uncertain/减少 positive，则只能作可靠性分析。它是重要的问题定义，**不是用户要求的 hallucination mitigation 主算法**。

## 3. Top-3 敌对排序

评分使用 `I/M/N/E`：Importance、Mechanistic value、Novelty space、Executability，均为 0--3；`R` 是竞争修正。高分只是优先级，不是接受结论。

| Rank | Candidate | I | M | N | E | R | Final | 最大致命风险 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Selection--Reuse Inflation | 3 | 3 | 2 | 3 | -0.2 | **2.60** | native patch score 可能根本不把搜索膨胀传到最终 FP |
| 2 | Anatomy-Conditional Randomization | 3 | 3 | 2 | 2 | 0 | **2.60** | 无法构造近似可交换的 CXR conditional replacement |
| 3 | Cross-Fit Visual Confirmation | 3 | 3 | 2 | 2 | -0.2 | **2.40** | 可用的两套 response 仍高度相关，退化为 LENS 式稳定性 |

排序解释：

1. **C1 最先跑**，因为它直接利用正在生成的 patch cache，并给正负结果都提供清晰结论：若 null inflation 存在，解释 local enhancement 的 recall--FP 交换；若不存在，整个 selective-inference 故事关闭。
2. **C2 是更高风险但更“自然”的备选**：它把此前任意 noise/mask/null 的失败统一解释为 null misspecification，方法也只有一句话；但 exact exchangeability 极难，不能先写保证。
3. **C3 只在能构造真正不同的 validator 时保留**。若 validator 只是同图另一次 prompt 或另一层，已有结果已说明这很可能仍是 common response。

三者不是堆模块关系：C1 校准搜索产生的选择偏差；C2 解决视觉 null 分布错误；C3 用信息分账降低选择--验证相关性。正式主线只能选择一个经过致死门的机制，不能把三个拼成系统。

## 4. 最小执行顺序

1. **零 GPU / 10--20 分钟：C1 Null Search Expansion。** patch cache 一旦落盘，先只跑全 0/3 negative images；不等 positive classifier，不调窗口。
2. **若 C1 的 raw inflation 不传入 final margin：关闭 C1，转 C2。** 单卡 32 图、每图 2 个 anatomy-matched replacements；先看 exchangeability/manipulation，不先看 accuracy。
3. **若 C1 inflation 成立但 calibration 后真信号消失：论文可留下“局部搜索制造 false evidence”的机制负结果，但不能叫 mitigation。** 此时 C3 只做 CPU selector/validator correlation 实验，判断是否存在保 power 的去相关验证器。
4. **只有任一候选同时满足：两模型同向、final margin 之上 +0.02 AUROC、null FP 可控、positive power 保留，才进入生成实验。** OE 必须 fixed-K；CE 必须 matched Yes-rate；任何通过删 claim、缩短回答或扩大 uncertain 获益的版本判失败。

## 5. 当前最高诚实结论

当前没有已完成的 ICLR Oral 方法。最值得探索的不是“再找一次病灶”，而是下面这个可证伪问题：

> **局部 VLM 方法是否把“搜索到的最大响应”误当作“独立视觉证据”，从而在病灶越小、候选越多时同时制造遗漏补偿与假阳性？**

它满足用户偏好的简洁性：一句话能讲清；它也有非平凡数学背景，但不会把经典定理据为己有。它的 oral 潜力只来自一个尚未证实的 VLM-specific phase law：`病灶稀疏度 × 搜索空间 × selector--validator correlation` 能否跨模型、跨局部方法、跨医学/自然图像预测 FP 与 FN。致死门失败后，不应再用 OT、拓扑、e-value 或 Fisher 几何给同一失败换名字。

## 参考文献（本轮承担碰撞结论的主要来源）

- Park et al. [SECOND](https://proceedings.mlr.press/v267/park25c.html), ICML 2025.
- An et al. [AGLA](https://openaccess.thecvf.com/content/CVPR2025/html/An_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_with_Assembly_of_CVPR_2025_paper.html), CVPR 2025.
- Mishra et al. [CEBC](https://aclanthology.org/2026.acl-long.2142/), ACL 2026.
- Xu et al. [BCEA](https://arxiv.org/abs/2606.16667), arXiv 2026.
- Li et al. [VISTA](https://openreview.net/forum?id=7BKcLeHQsm), ICML 2025.
- Liu et al. [VTI](https://openreview.net/forum?id=LBl7Hez0fF), ICLR 2025 Spotlight.
- Zhao et al. [MARINE](https://proceedings.mlr.press/v267/zhao25j.html), ICML 2025 Spotlight.
- Li et al. [Visual Evidence Prompting](https://aclanthology.org/2025.acl-long.205/), ACL 2025.
- Pan & Bareinboim. [Counterfactual Image Editing with Disentangled Causal Latent Space](https://proceedings.neurips.cc/paper_files/paper/2025/hash/54e1419b1cbca8ada96a87c96567e954-Abstract-Conference.html), NeurIPS 2025.
- Peng et al. [PHG-Net](https://openaccess.thecvf.com/content/WACV2024/html/Peng_PHG-Net_Persistent_Homology_Guided_Medical_Image_Classification_WACV_2024_paper.html), WACV 2024.
- Bazarova et al. [TOHA](https://arxiv.org/abs/2504.10063), arXiv 2025.
- Xu & Ramdas. [Online Multiple Testing with E-values](https://proceedings.mlr.press/v238/xu24a.html), AISTATS 2024.
- Wei et al. [LeHaCE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c9b551a2e195a209fc0b280de2f7f781-Abstract-Conference.html), NeurIPS 2024.
- DEEM. [Diffusion Models Serve as the Eyes of LLMs](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a8399aace3dfa6dfb8b635117748c561-Abstract-Conference.html), ICLR 2025.
- Fithian, Sun & Taylor. [Optimal Inference After Model Selection](https://arxiv.org/abs/1410.2597), 2014.（数学背景，不作新颖性来源）
- Wang & Ramdas. [False Discovery Rate Control with E-values](https://academic.oup.com/jrsssb/article/84/3/822/7056146), JRSS-B 2022.（数学背景）

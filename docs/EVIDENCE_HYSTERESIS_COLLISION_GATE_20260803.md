# Evidence Reveal–Removal Hysteresis：collision 与可识别性闸门

**冻结日期：** 2026-08-03  
**决定：** **HARD NO-GO；不运行 GPU。**  
**适用范围：** 单张医学影像、冻结 HuatuoGPT-Vision/Hulu/LLaVA-Med、无跨调用持久状态的 CE/OE 推理。

## 0. 一句话结论

在标准 stateless VLM forward 中，若 reveal 与 removal 在第 `k` 步具有完全相同的可见像素、prompt、chat template、processor tensor 和 decoding rule，则两次模型输入完全相同，所谓“路径差”在定义上只能为零；若人为保留历史、KV cache 或前一轮答案使其非零，研究对象已经变成 **multi-turn belief-update anchoring**，而不是单图医学 VLM 的证据迟滞，且该问题已被 2024--2026 的 belief revision、Evidence Update Prompting 与视觉自纠正工作直接占据。

VinDr 的三读者框确实提供充足的阳性 finding/ROI 数量，但只 truth `原图中 finding 是否被三位读者标出`，不能 truth `某个中间 mask 还保留了多少临床证据`。因此样本数通过、科学构念仍失败。这个候选不能通过增加 mask、换 baseline、做 phase averaging 或跑更多模型补救。

## 1. 先冻结“hysteresis”必须是什么

设原图为 `x`，固定 finding claim 为 `c`，冻结 prompt 为 `p`，`M_0 ... M_K` 是严格 nested masks。对统一背景算子 `B`，第 `k` 个可见图为

\[
x_k=M_k\odot x+(1-M_k)\odot B(x).
\]

冻结模型、processor、chat template 与 scoring contract 后，claim commitment 为

\[
q_k=Q_\theta(x_k,c,p).
\]

真正的路径迟滞要求系统还存在一个由过去输入决定、且没有被当前输入完全确定的状态 `h_k`：

\[
h_k=G_\theta(h_{k-1},x_k),\qquad q_k=Q_\theta(h_k,x_k,c,p).
\]

此时同一 `x_k` 才可能因 `h_k^\uparrow != h_k^\downarrow` 而得到

\[
q_k^\uparrow != q_k^\downarrow,
\]

并定义 loop area 或 switching-threshold gap。**历史依赖的内部状态是 hysteresis 的必要条件，不是可省略的实现细节。** 2021 年关于遮挡物体识别的 recurrent-network 工作之所以能产生 perceptual hysteresis，正是因为网络在遮挡序列之间保留 recurrent state，而非把同一张图独立 forward 两次（[JOV/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8684313/)）。

## 2. 为什么当前 VLM 中路径变量不存在

### 2.1 Stateless forward 的零效应是恒等式

对当前单图 VLM adapter，常规调用是

\[
F_\theta(\operatorname{processor}(x_k),\operatorname{tokens}(p,c)).
\]

每一张 masked image 独立运行，调用结束后 hidden states 与 KV cache 被释放。只要 reveal/removal 在同一 `k` 上产生 bit-identical processor tensor，确定性 teacher forcing 或 greedy decoding 必然给出

\[
q_k^\uparrow=q_k^\downarrow.
\]

因此：

- “从低证据向上首次越过阈值”与“从高证据向下首次越过阈值”若读取同一条离散曲线，只是同一个 crossing 的两种遍历写法；
- 非单调 `q_k` 是普通 occlusion/response curve，不是 hysteresis；
- 若 sampling decoding 得到差异，两条路径的条件分布仍完全相同，观测差只是 Monte Carlo noise；
- 若两方向采用不同 mask、不同背景、不同 prompt 或不同阈值，则比较的是两种 intervention，不再是同一状态上的路径效应。

### 2.2 自回归生成并没有自动提供跨 mask 的状态

单次回答内部当然有 autoregressive history，但第 `k-1` 张 masked image 的回答不会自动进入第 `k` 次独立调用。已有 [PAS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Hoang_PAS_Prelim_Attention_Score_for_Detecting_Object_Hallucinations_in_Large_CVPR_2026_paper.html) 已表明 hallucinated object token 可过度依赖此前生成的 prelim tokens；这是 **within-generation language lock-in**，不能给 reveal/removal 跨 forward 的路径差提供状态。

### 2.3 四种可实现的“状态”都改变了研究问题

| 实现 | 同一当前可见图上能否产生路径差 | 为什么不属于当前主线 |
|---|---:|---|
| 每步独立 forward | 否 | 无持久状态，差异恒为零 |
| 把前轮 image/answer 放入 multi-turn history | 是 | 完整输入已不同；测 conversational anchoring/belief revision |
| 跨图强行复用 KV cache | 可制造 | cache 编码了旧 image tokens；替换视觉 token 后是 cache contamination/OOD，而非合法模型语义 |
| test-time adaptation、external memory、recurrent/video wrapper | 是 | 研究新 stateful system，不是冻结单图医学 VLM |

若把 nested masks 作为一组多图同时输入，模型看到的仍是不同序列前缀与位置编码；路径效应不能再归因于当前图像证据。若把旧视觉 KV 保留、只替换 pixels，cached keys/values 并不会自动重新计算，任何 persistence 都首先由 stale cache 解释。

## 3. Collision audit：两个可能的重写都已被占据

### 3.1 若无状态，它退化为 masking / insertion–deletion

| 最近或基础邻居 | 已覆盖的对象 | 对本候选的约束 |
|---|---|---|
| [RISE, BMVC 2018](https://arxiv.org/abs/1806.07421) | masked-input probing 与 deletion/insertion curve | 独立 forward 的 nested reveal/removal 首先是成熟 saliency faithfulness protocol，不是新机制 |
| [Improving Interpretation Faithfulness for ViTs, ICML 2024](https://proceedings.mlr.press/v235/hu24k.html) | input perturbation 下 explanation/prediction stability | 仅报告 mask 曲线稳定性或 saliency consistency 不构成新问题 |
| [Evaluating Reasoning Faithfulness in Medical VLMs using Multimodal Perturbations, ML4H 2025 proceedings/PMLR](https://proceedings.mlr.press/v297/moll26a.html) | 胸片 VQA 的受控 image/text perturbation、attribution 与 confidence calibration；六个 VLM、放射科 reader study | “医学图像扰动后答案/解释是否改变”已是直接邻居；需要额外、可识别的 state mechanism 才能越过 |
| [Same Attention, Different Truths, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Same_Attention_Different_Truths_Put_Logit-Lens_over_Visual_Attention_to_CVPR_2026_paper.html) | mask 高注意区域后，visual-uncertainty hallucination 消失，而 contextual-prior hallucination 持续并发生 attention drift | “移除证据后 claim 仍持续”已被解释为 contextual prior persistence；无路径变量时不能重命名为 hysteresis |
| [Questioning the Stability of VQA, arXiv 2025](https://arxiv.org/abs/2511.11206) | pixel shift、rescale、paraphrase 等 benign perturbation 下 VLM 答案不稳定 | pixel shift/轻微变换必须作为 nuisance control，不是贡献 |
| [Phase Marginalization for Patch-Grid Instability, arXiv 2026](https://arxiv.org/abs/2606.08132) | 将 patch-grid phase 定义为 nuisance，并用 training-free 多 phase aggregation | shift/phase averaging 已是明确方法类，只能作为控制或 baseline |

所以无状态版本最多产生一篇医学 masking sensitivity / saliency audit；它不能支撑 “evidence commitment hysteresis” 的机制标题。

### 3.2 若加入状态，它退化为 belief revision / anchoring

| 邻居 | 已覆盖的对象 | 碰撞强度 |
|---|---|---:|
| [Belief Revision, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.586/) | 新 premise 到来时更新/不更新结论；约 30 个 LM，并揭示 update/no-update trade-off | 高：文本层 belief-update 基础问题已建立 |
| [Do VLMs Revise Beliefs or Just Rationalize? Evidence Update Prompting, CVPR 2026 CogVL workshop poster](https://openreview.net/forum?id=36Sde2FKCU) | 先给有限视觉证据、再给新增证据；错误初判的 37--62% 未修正，belief-state prompt 降低 stubbornness 13--18pp | **直接碰撞**：视觉证据序列、anchoring、confidence inflation 均已测；注意该文是无 workshop proceedings 的 poster，证据级别应如实标注 |
| [VISCO, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_VISCO_Benchmarking_Fine-Grained_Critique_and_Correction_Towards_Self-Improvement_in_Visual_CVPR_2025_paper.html) | 1,645 个视觉推理回答的细粒度 critique/correction；模型自评常有害，LookBack 重看图像可改善 | 高：视觉自纠正与 refusal-to-say-no 已是显式任务 |
| [Evaluating VLMs on Bistable Images, CMCL 2024](https://aclanthology.org/2024.cmcl-1.2/) | 29 张双稳态图、121 种 manipulation、12 模型；模型不像人类表现 continuity bias，prompt/label prior 影响更强 | 中高：视觉 ambiguous persistence 与人类 continuity 已被直接比较，且结果不支持简单类人迟滞假设 |

把胸片 ROI nested masks 放入多轮对话可以形成一个窄的 **Clinical Evidence-Update Anchoring** benchmark；但它偏离当前单图 OE hallucination 主线，并且与 EUP 的任务定义只剩医学数据与局部 mask 差异。按本项目 collision 规则，这不是 ICLR oral 级机制空位。

## 4. VinDr CPU-only truth 与 power census

### 4.1 已执行的只读审计

数据来自 `/workspace/vinbigdata/train.csv` 与 15,000 张 train DICOM。CSV 共 67,914 行；R8/R9/R10 三位固定读者均出现的图像为 5,501 张。对每个 `(image_id, finding)`：

1. 三位读者均标正并提供 bbox；
2. 每位读者恰好一个 bbox；
3. 三对 box 的平均 IoU 至少 0.5；
4. 用 DICOM `Rows/Columns` 检查几何尺度。

结果为：三读者均有局部框 5,112 个 claim-image，三位各恰一框 3,988 个，平均 pairwise IoU≥0.5 后剩 3,529 个，分布在 2,657 张独立图像；所有相关 DICOM header 均可读取。

| Finding | 三读者各一框且 mean IoU≥0.5 | 448 输入上 consensus-box 短边≥2 patches | 短边≥4 patches |
|---|---:|---:|---:|
| Aortic enlargement | 1,628 | 1,628 | 231 |
| Cardiomegaly | 1,252 | 1,251 | 757 |
| Pleural effusion | 173 | 151 | 132 |
| Pulmonary fibrosis | 154 | 93 | 40 |
| Nodule/Mass | 90 | 53 | 30 |
| Lung Opacity | 51 | 50 | 40 |
| Infiltration | 34 | 33 | 27 |
| Pneumothorax | 32 | 32 | 29 |

按 `patch=14` 的保守几何 screen，至少 **6 个 findings** 在四 patch 短边下仍有 `n≥30`，所以“≥3 findings × 关键阳性 cell≥30”的**数量门通过**。这也说明 NO-GO 不是数据没下载或样本太少造成的。

### 4.2 但 intermediate-mask truth 不存在

VinDr box 是 lesion extent 的粗矩形标注，不是 segmentation、visibility rating 或 causal sufficient set。它不能告诉我们：

- 只显示 box 中央 20% 时，一位放射科医生是否仍支持 finding；
- 擦除 box 外圈是否删除了证据，还是只删掉正常背景；
- box 外 anatomy、对称性、全局心胸比例、胸膜轮廓是否仍提供决定性证据；
- 一个 masked render 应属于 `supported / refuted / unobservable` 哪一态。

尤其 cardiomegaly、aortic enlargement 与 pleural effusion 是全局/边界性 finding，把一个矩形逐步揭示并不对应临床证据的自然剂量。Nodule/Mass 的框更局部，但 bbox 仍混入大量背景，且小 lesion 在 patch 化后只有少数离散 token。

因此 VinDr 可以 truth `原图的 finding 与读者定位共识`，不能独立 truth `mask level k 的 evidence support`。把模型 score 自己当作 evidence curve、再用该 curve 定义阈值，是 truth leakage。

此外，本地 DICOM 缺少可用 patient/study ID；当前只能 image-disjoint，不能证明 patient-disjoint。对纯 ROI screen 这不是最先失败点，但不满足正式论文独立性合同。

## 5. 遮挡 artifact 为什么无法被一个漂亮的 mask schedule 消除

### 5.1 每种 background 都引入新的解释

| Mask background | 主要 confound |
|---|---|
| black/white/mean fill | 极强 OOD 边界、曝光与直方图变化 |
| Gaussian blur | 保留低频轮廓且改变纹理/噪声谱；不是“无证据” |
| noise fill | 改变 acquisition texture，可能激活域偏差 |
| matched donor patch | 移植另一患者 anatomy/pathology，破坏图像一致性 |
| generative inpainting | inpainting 模型可能生成或删除 finding；truth 转移给另一个模型 |
| crop/zoom reveal | 同时改变分辨率、上下文、position 与 patch-grid phase |

soft alpha boundary 可以减少锐利边缘，却不能使 intervention 成为临床有效的证据剂量。多 background 一致性只能降低某些 artifact 的可能性，不能创造 intermediate truth。

### 5.2 nested masks 与 ViT patch grid 强耦合

当 reveal 边界跨过一个 patch，visual token 会离散突变；resize/crop 的一像素位移还会改变 patch partition。因而 apparent switching threshold 可能只是 patch-grid phase。2025 的 VQA stability 工作已系统显示 benign pixel shifts 足以改变答案；2026 Phase Marginalization 已把 phase 定义为 nuisance 并提供多 phase aggregation。任何此类实验至少需要 `4 phase × ≥3 backgrounds × shifted-box controls`，但做完也只得到更可信的 perturbation curve，不会得到路径机制。

## 6. 若仅作为 falsification，唯一合法的固定 output contract

本候选已经 NO-GO，下面合同只说明怎样避免将来误跑；它不授权 GPU。

1. **原子 claim 固定：** 每个 finding 使用一条冻结 declarative proposition，不生成 OE 草稿，不换 synonym。
2. **原生模板固定：** 每个模型使用其官方 image placeholder/chat template；processor version、resize、normalization、dtype 全部 hash-bound。
3. **主读数为三序列 teacher forcing：** 对相同 prefix 计算完整序列的长度归一化 log-likelihood：`supported`、`refuted`、`undetermined`。不得混用某个模型的单 token `Yes/No` 与另一个模型的生成文本。
4. **确定性：** `model.eval()`、固定 kernel、无 sampling；同一个 processor tensor 重复 5 次必须 bit-identical logits（若硬件仅允许数值确定，则最大差需预注册到 `1e-6`）。
5. **路径盲化：** scorer 只接收 current tensor；任何 path label、文件名、turn index 不可进入 prompt。
6. **identity test：** 对每个 `k`，reveal/removal 的 PNG/DICOM-derived array hash、processor tensor hash、input IDs 必须逐项相同。若相同而 score 不同，先判 nondeterminism/hidden cache bug，不判机制。
7. **OE 只可作 secondary：** 若以后测 native generation，必须固定 greedy、claim count/length、nonempty、cap-hit、refusal 与 exact-template diversity；不能用少说或 hedge 当 mitigation。

在此合同下，stateless 主假设的预注册期望就是 `A_loop=0`。它是软件单元测试，不是需要 7B 模型 GPU 实验的新科学问题。

## 7. 真正 stateful 版本需要的双向 counterfactual（仅用于说明为何偏题）

若研究目标明确改为“对话中的临床证据更新”，最小设计不是一条 reveal 曲线，而是冻结最终当前图 `x_k` 后交叉两个历史：

| 历史 state | 最终证据支持 `c` | 最终证据反驳 `c` |
|---|---:|---:|
| 先前诱导支持 `c` | persistence cell | required reversal cell |
| 先前诱导反驳 `c` | required reversal cell | persistence cell |

必须再加入：

- identical final image + history-reset control；
- 相同 token 数、轮数、措辞与 confidence 的 support/refute histories；
- history-only、image-only、random-history、same-support different-image controls；
- 两方向均以独立临床 truth 评分，而不是把“是否改答案”当准确性；
- 不复用 stale KV；每个完整 conversation 从头合法重算；
- primary estimand 是 `history × final-evidence` interaction，不是 marginal answer change。

这可以排除普通 prompt priming 的一部分，但完整 inputs 仍然不同，结论只能是 conversational anchoring/belief revision。EUP 已用有限证据→新增证据、stubbornness、confidence inflation 和 belief-state prompt 直接覆盖核心现象；医学化本身不足以立项。

## 8. 必须预先排除的替代解释

1. **确定性/采样噪声：** seed、sampling、kernel 或 quantization nondeterminism。
2. **hidden-state 泄漏：** adapter 全局变量、未清空 cache、复用 conversation object。
3. **mask OOD：** fill 值、锐利边缘、频谱与曝光改变。
4. **patch-grid phase：** resize、shift、mask 边界导致 visual token 相位变化。
5. **box truth 不充分：** bbox 不是 segmentation；关键证据可能在 box 外。
6. **global disease cue：** 心胸比例、对称性、胸膜轮廓与共病仍可支持 claim。
7. **language/contextual prior：** 移除视觉区域后 claim 持续，正是 CVPR 2026 已描述的 contextual-prior mechanism。
8. **prompt/history anchoring：** stateful 版本可能只是前轮答案的语言重复。
9. **within-generation prelim dependence：** PAS 所描述的已生成 token lock-in。
10. **parser/threshold artifact：** 连续 score 很小的变化被离散阈值放大；不同方向搜索算法产生伪 gap。
11. **output contraction：** OE 生成更短、统一否定、拒答或 hedge。
12. **finding difficulty/size：** 大框与小框、局部与全局 finding 混合造成曲线差异。

## 9. CPU-only one-day fail-closed gate 与本次结果

| 时段 | CPU artifact / 判据 | 本次结果 |
|---|---|---|
| 0--1h：state identifiability | 写出 `h_k` 的存储位置、更新方程与 reset；同 current input 下历史是否仍不同 | **FAIL**：标准单图 forward 无 `h_k`；引入 history/KV 即改变任务 |
| 1--3h：collision | 检索 mask insertion/deletion、medical perturbation、belief revision、visual update、patch phase 及官方 code | **FAIL**：无状态撞 saliency/perturbation；有状态直接撞 EUP/belief revision |
| 3--5h：truth census | ≥3 findings、关键 cell≥30、独立 truth、patient/study split | **COUNT PASS / TRUTH FAIL**：6 findings 在严格几何 screen 下≥30；中间 masks 无 reader truth，patient grouping 缺失 |
| 5--7h：render identity | 双路径同 `k` 的 array/tensor/token hashes 全等；4 phase 与≥3 background 只作 nuisance screen | **NOT RUN**：前两道 hard gate 已失败；即使通过也只证明输入一致 |
| 7--9h：fixed output contract | 三态 teacher-forcing adapter 单元测试、重复性、native template | **NOT RUN**：没有可识别 estimand，不应消耗模型算力 |
| 9--12h：causal falsification pack | 双向 counterfactual、interaction estimand、替代解释相反预测 | **FAIL for mainline**：合法双向设计必须变成 multi-turn conversational task |

根据 fail-closed 合同，Gate 0 或 Gate 1 任一失败就停止；不得因为数据数量漂亮而继续跑 GPU。**本次没有启动任何 GPU process。**

## 10. 最终研究决定

### 删掉的 claim

> 医学 VLM 对同一 finding 的 commitment 存在 reveal/removal evidence hysteresis。

对当前 stateless 单图模型，该 claim 不可定义；对 stateful wrapper，它又不是当前模型/任务的属性。

### 不应包装成贡献的观察

- claim score 随 mask level 非单调；
- 移除 reader box 后模型仍报告 finding；
- insertion 与 deletion AUC 不同但两者使用了不同图片集合/背景；
- phase averaging 让曲线更平滑；
- 多轮先说“有”以后不愿改口。

前四项属于 perturbation/saliency、mask artifact 或 contextual prior；最后一项属于 belief revision/anchoring，且已有直接工作。

### 可保留为 baseline/control 的部分

- VinDr 三读者 consensus boxes 仅可作为未来合格视觉机制中的 ROI sensitivity **secondary control**；
- same-mask processor hash identity 可加入任何视觉 counterfactual 的软件测试；
- 4-phase marginalization、shifted-box、multi-background 可作为 mask 方法的 nuisance baselines；
- “移除 ROI 后 claim 是否保持”可帮助区分 perceptual limitation 与 contextual prior，但不能称 hysteresis。

### 对主线的实际价值

这次 NO-GO 给出一个应写入后续所有机制方案的硬规则：

> **凡声称 path dependence，必须先指出跨 observation 持久化的合法 state；若模型调用是 stateless，则 path effect 要么严格为零，要么来自不同输入、随机性或实现泄漏。**

本次 NO-GO 只否定 Evidence Hysteresis，**不提供复活 Two-Plane 的理由**；Huatuo frozen residual gate 已失败，并已在 `PAPER_SCOPE_GATE` / ICLR portfolio 中被 `Reject and Pivot`。当前应优先完成 Specificity / CECD 的独立 truth gates；PCEM 仅保留为 access-blocked 候选，不为 Evidence Hysteresis 新开 GPU 支线。

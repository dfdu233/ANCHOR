# Answer-Space Evidence Substitution：机制碰撞审计

日期：2026-08-02  
判定：**KILL 作为论文主机制；不授权实现、GPU canary 或 baseline-scale 实验。**

## 1. 被审计的候选

候选主张是：

> 疾病词或候选答案进入 prompt 后先形成 claim 方向；decoder 后层把这条文本方向误当成视觉支持，因此同一图像、同一原子 claim 在 CE、OE 与报告生成中出现不一致。

原计划用 prompt-end 的 text-only claim vector 与 image residual 分解，并在固定
claim identity、polarity 和真值的条件下做因果 patching。

这不是一个空洞现象：prompt 中的候选答案、错误建议和 false presupposition
确实会压过图像；任务 framing 也确实会改变视觉处理。但截至当前原始论文检索，
该叙事已经能被更早、更加直接的机制解释覆盖，新增的 residual 分解主要是实验工具
重组，而不是不可替代的新机制。

## 2. Closest-work matrix

| 工作 | 同图/同语义真值 | 跨 answer space | 医学 | 内部机制 | 因果干预 | 对候选的占据 |
|---|---|---|---|---|---|---|
| [Tinted Frames (2026)](https://arxiv.org/abs/2603.19203) | 是：同一 semantic query 三种 framing | OE / Yes-No / MCQ | 否 | 中层开始的 visual-energy、ROI-attention 下降 | attention steering；soft-token realignment | **最直接碰撞**：framing 改变视觉处理并导致跨 framing 不一致，已完成机制与修复闭环 |
| [Mechanisms of Prompt-Induced Hallucination (ACL 2026)](https://aclanthology.org/2026.acl-long.1941/) | 同图、错误计数 prompt | prompt assertion 对视觉计数 | 否 | 少量 model-specific PIH heads 介导 prompt copying | head ablation 至少降低 40% PIH | **直接碰撞**：prompt direction 压过 image evidence 的稀疏因果回路已被定位 |
| [When Prompts Override Vision / HalluScope (2026)](https://arxiv.org/abs/2604.21911) | 同图；存在性与属性 presupposition | recognition / false-presupposition OE | 否 | 行为上区分 perception、co-occurrence、instruction prior | HalluVL-DPO | **现象直接碰撞**：prompt 将不存在对象预设为事实并诱发属性幻觉 |
| [Med-CP (EACL 2026)](https://aclanthology.org/2026.eacl-industry.67/) | 医学 image-question 上加入正确/错误临床信息 | 医学 VQA prompt variants | **是** | 无 layerwise 分解 | cross-modal reflection SFT | **医学直接碰撞**：Med-VLM 会盲从 noisy user prompt，无论其是否正确 |
| [ReMedQA (EACL 2026)](https://aclanthology.org/2026.eacl-long.124/) | 同一医学问题及真值 | OE / MCQ / option perturbations | **是，但 text-only** | 无 | 无 | **评测直接碰撞**：候选项和格式使同一医学真值不一致；仅看选项也可高分 |
| [MedVIGIL (2026)](https://arxiv.org/abs/2605.07919) | 医生构造的 gold、false premise、ROI counterfactual | paired MCQ / OE | **是** | 无 hidden-state mechanism | 无 | **医学 substrate 碰撞**：paired OE、候选集与 false-premise trap 已存在 |
| [MedVH (2024/2025)](https://physionet.org/content/medvh/1.0.1/) | 同一 MC-VQA 加入随机错误 suggested answer | MC answer / long justification；另含 report | **是** | 无 | 无 | **早期行为碰撞**：错误答案进入上下文后，测试模型能否反驳并给出替代；报告样本并非同 case |
| [Vision-Default, Prior-Override (2026)](https://arxiv.org/abs/2606.28273) | 同一 counterfactual image、不同 grounding prompt | visual answer / parametric-prior answer | 否 | residual、head、MLP 三粒度 activation patching；稀疏晚层 routing/writing heads | 双向 patching 与 ablation | **方法学直接碰撞**：同图异 prompt、固定竞争答案的文本/视觉因果分解已被实施 |
| [System-Mediated Attention Imbalances (Findings ACL 2026)](https://aclanthology.org/2026.findings-acl.1940/) | binary visual tasks | Yes/No | 否 | late system attention 促成 coarse default 与 yes-bias | 跨 modality causal attention redistribution | 占据 CE yes-bias 的另一充分解释；也警告 CE 与 autoregressive hallucination 机制可能不同 |
| [FINER (2026)](https://arxiv.org/abs/2603.17662) | 真元素与单个细粒度假元素共现 | negative queries / what questions | 否 | 无 layerwise 机制 | FINER-DPO | 占据“candidate detail 提高错误肯定”的细粒度行为现象 |
| [FundusGround (2026)](https://arxiv.org/abs/2605.22414) | 共享 structured lesion evidence | open / closed / single-choice / multi-choice | **是** | 无 | evidence grounding | 占据从同一医学 lesion evidence 派生多格式问题的数据构造；未做同 claim 机制 |
| [MEDA (ICML 2026)](https://openreview.net/attachment?id=VzZxoRiU3G&name=originally_submitted_PDF) | 非同 case / claim | medical VQA 与 report generation | **是** | manifestation 与 diagnostic-principle activation editing | activation steering | 占据医学跨任务 activation editing，但不提供 prompt/image provenance 分解 |

## 3. 为什么不是“只差医学化”

候选想同时占据四件事：跨格式不一致、prompt 压过图像、内部因果定位、医学跨任务
有效。它们虽然没有被单篇论文全部打包，却分别已由高度相邻的工作完成：

1. `Tinted Frames` 已把 **同语义 OE/YN/MCQ -> 中层视觉 disengagement ->
   因果 attention restoration** 做成完整闭环。
2. ACL 2026 PIH 已把 **prompt direction -> sparse prompt-copying heads -> ablation
   恢复视觉纠正** 做成完整闭环。
3. Med-CP、MedVIGIL、MedVH 已把错误 clinical information、false premise、错误
   suggested answer 放入医学 VLM 语境。
4. ReMedQA 和 FundusGround 已占据医学共享真值/共享 lesion evidence 的开放与受限
   answer-space 构造。
5. Vision-Default/Prior-Override 已占据同图异 prompt、固定竞争答案、component-level
   patching 的方法学空间；MEDA 又占据医学 activation editing 的应用空间。

因此“把这些元素组合到同一病例、同一原子 claim、CE/OE/report”会得到一个更严格
的 benchmark/control contract，但不会自动产生新机制。ICLR reviewer 可以自然地把
它描述为：`Tinted Frames + PIH in medical datasets, evaluated with a unified claim parser`。

## 4. 不可替代差异审计

目前唯一可能不可替代的差异不是 `evidence substitution`，而是更窄的
**Evidence-Provenance Misbinding**：

> 模型晚层保留 claim semantics，却选择性丢失“该语义来自 prompt 还是 image”的
> source tag；错误不是简单 text dominance、visual disengagement 或 prompt copying。

但要让这个差异成立，必须同时满足：

1. 在同图、同原子 claim、同 polarity 下，prompt semantic direction 与 image evidence
   source 可被正交、交叉泛化地解码；不能把 token identity 当 source tag。
2. 晚层 claim semantics 仍在，而 source decodability 显著下降；下降不能由
   answer length、position、attention magnitude 或 framing token 解释。
3. 只 patch source component 就能纠正回答；patch claim direction、统一增加视觉
   attention、PIH-head ablation 和随机范数匹配不能解释效果。
4. 在保持 claim identity、polarity、正 claim 数量和长度时，CE、OE、report 至少两类
   任务出现同方向效应。
5. 该机制必须优于 `Tinted Frames` 的 attention restoration、ACL PIH-head control、
   text-only answer prior、image swap、system-attention redistribution 和 generic
   activation steering。

这组条件需要一个真实的 source variable，而现有 VinDr reader votes 只提供支持度，
不提供 prompt/image provenance ground truth。报告又没有天然和 CE/OE 共享同一显式
命题。因此当前 substrate 尚不能识别该机制；直接运行 hidden-state probe 极易把
framing、position 或 token-copying 误命名为 source erasure。

## 5. GO / KILL

### 当前决定

- `Answer-Space Evidence Substitution` 作为宽泛机制：**KILL**。
- “同病例同原子 claim 的 CE/OE/report 一致性”作为评测协议：**可作为 control，不能
  headline mechanism**。
- prompt-end text-only vector 减 image residual：**不授权**；它不能单独区分
  Tinted-Frames visual disengagement、PIH prompt copying 和 generic text dominance。
- `Evidence-Provenance Misbinding`：**仅保留纸面上的独立候选，不立项**；除非先有
  source-preserving stimulus contract 和一个无需 GPU 即可验证的 construct gate。

### 唯一重新 GO 的条件

先构造 source-preserving tetrad，并在行为层通过下列 preregistered gate：

1. claim wording、answer space、长度与 truth 全固定，只改变 evidence source；
2. 正确答案在 image-only、text-only、congruent、conflict 四格中可定义；
3. 同一机制效应跨两个模型，并超过 same-support image swap 与 prompt paraphrase 漂移；
4. 现有三种充分解释——visual attention loss、prompt copying、system yes-bias——均被
   独立控制后仍有剩余效应。

若这四项中任一失败，永久关闭该分支。通过也只授权一个小型 CPU/单 batch
representational canary，而不是整套 GPU 实验。

## 6. 对当前研究主线的影响

本次 mechanism-research 审计避免把一个真实但已被充分解释的现象包装成新机制。
它进一步强化当前项目的研究纪律：

- CE 中疾病词进入 prompt 后的肯定偏差，不能自动推广为 OE/report hallucination；
- 跨 answer-space 的统一原子 claim 真值是必要评判规则，但属于 measurement quality；
- 下一条主线必须带来新的可识别 latent variable 或临床 construct，而不是把已有
  attention、prompt-copying 和 activation-editing 元件重新组合。


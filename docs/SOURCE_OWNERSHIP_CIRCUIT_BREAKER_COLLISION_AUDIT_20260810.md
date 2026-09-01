# Source-Ownership Circuit Breaker：机制与公式碰撞审计

> **2026-08-10 superseding collision:** TrustNLP 2026 *Ghost Context* 已直接覆盖
> wrong-context misattributed grounding、source-blind faithfulness 的结构性盲点、mask-and-rerun
> 因果归因与后处理修复。故“错误来源影响答案”及 span masking 不是剩余 novelty。候选仅在
> patient-owner × clinical-predicate 的内部变量绑定、layer/path 衰减、双向因果搬运和
> fixed-coverage multimodal mitigation 全部成立时保留；否则判 cosmetic medical extension。

第二轮独立 red-team 进一步排除以下宽泛贡献：

- ContextCite（NeurIPS 2024）已覆盖 statement-level context attribution 与 pruning；span attribution/
  deletion 不是方法增量。
- Wu et al.（ICML 2025）已用 residual addressable memory、routing heads 与 causal intervention
  解释 variable binding；“找到 owner binding subspace”本身只是领域移植。
- Pronoun Fidelity（2026）已研究 group/entity binding、分布式 causal subspace 和 competing copying
  routes；owner-binding direction 不能单独成贡献。
- CoDA（ACL Findings 2026）已研究中晚层 context-selective routing 变弱与 parametric influence；
  “晚层 context/source evidence 消失并恢复 routing”也不是剩余主张。
- Taming Knowledge Conflicts（ICML 2025 Spotlight）的 JuICE 已对叠加 contextual/parametric
  information 的高影响 head 做 test-time attention steering；source-aware head/direction selection 不新。
- Representation Before Retrieval（2026）已在 clinical RAG 中指出 raw retrieval 可制造包括错误
  test attribution 在内的 hallucination，并用结构化 provenance artifact 防御；显式 patient tag 或
  metadata defense 不能作为算法贡献。

因此最终 go/no-go 必须同时满足：两模型存在 OTHER polarity transport；早层 owner×predicate
binding 显著高于末层而 predicate polarity 保留；双向 binding-only patch 对错误有选择性因果效应；
edge intervention 优于 Ghost/ContextCite mask、SPIN、provenance prompt、CoDA/JuICE-style
context routing。否则候选 KILL。

**日期：** 2026-08-10  
**候选：** 用 same-report polarity twins 找到 donor-state odd heads，再结合 current-image vs null
的 visual response，只在“高 odd、低视觉或与视觉冲突”的 heads 上投影掉 odd component。

## 裁决

> **按当前公式作为新 mitigation algorithm：KILL。作为“source ownership binding failure”机制：有条件 KEEP，但必须重构识别实验。**

当前方法可以被准确概括为：

> SPIN/V-ITI 式动态 head selection + CAA 式 contrast-pair direction + LEACE 式 concept projection。

这三个表面组件均已有强工作，且 ICLR 2025 已用 counterfactual causal mediation 定位
hallucination heads 并做 training-free targeted intervention。因此，仅凭“医学、RAG、逐 head
投影”不形成 ICLR delta。

但现有工作大多区分 visual vs textual influence，尚未检索到工作把**临床命题内容**与**患者归属**
拆成独立变量，并证明模型把 OTHER_PATIENT 的状态错误绑定到 CURRENT_PATIENT claim。若能建立
这个 source-binding circuit，并只切断 `donor-state -> current-claim` 的边而不是压制整个 head，
则存在真实机制增量。

## 关键公式及其已有组件

设第 \(h\) 个 head 在报告 \(R\) 与 polarity twin \(gR\) 下的输出为 \(a_h(R)\)。当前候选近似为：

\[
u_h=\operatorname{norm}\,\mathbb E[a_h(R)-a_h(gR)],
\]

\[
G_{h,t}=\mathbb 1[O_{h,t}>\tau_o\ \land\
(V_{h,t}<\tau_v\ \lor\ \langle o_{h,t},v_{h,t}\rangle<0)],
\]

\[
a'_{h,t}=a_{h,t}-G_{h,t}\,u_hu_h^\top(a_{h,t}-\mu_h).
\]

| 组件 | 最近已有工作 | 碰撞 |
|---|---|---|
| 找 hallucination heads 并定向干预 | [Understanding and Mitigating Hallucination via Modular Attribution and Intervention](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8001c3568152d134d821cd46d4d84768-Abstract-Conference.html)，ICLR 2025 | 已用 counterfactual causal mediation 发现偏文本 hallucination heads，并对这些 heads 做 training-free intervention；**机制外壳高度重合** |
| 按 image attention 动态压制 heads | [SPIN](https://aclanthology.org/2025.emnlp-main.631/)，EMNLP 2025；[代码](https://github.com/YUECHE77/SPIN) | 每个 query token 保留高 image-attention heads、抑制低 image-attention heads；直接覆盖“低视觉 head suppression” |
| 先检测 visual neglect、再选择性 intervention | [V-ITI](https://arxiv.org/abs/2512.03542)，2025 | head-level probe 检测何时忽略视觉，并仅在需要时用 visual activation 调制；覆盖“when-to-intervene + head probe + visual recall” |
| visual/text conflict 驱动动态 attention intervention | [Owl](https://arxiv.org/abs/2511.09018)，AAAI 2026；[代码](https://github.com/CikZ2023/OWL) | 用 visual-to-textual contribution ratio 量化冲突，token/layer-wise 调制 attention，并做双路径解码；覆盖“视觉弱/文本强时干预” |
| 用 probe/gradient 找 causal hallucination heads 并 suppress | [VIB-Probe](https://arxiv.org/abs/2601.05547)，ACL 2026 | 从 head activation 检测幻觉，再以 gradient 识别有因果作用的 heads 并抑制；覆盖 probe-to-suppression pipeline |
| 正负 pairs 的 activation difference | [Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/)，ACL 2024；[代码](https://github.com/nrimsky/CAA) | 直接以 paired positive/negative activation mean difference 定义行为方向；same-report twins 只是更干净的医学 contrast pairs |
| 从 activation 投影掉目标概念 | [LEACE](https://papers.nips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html)，NeurIPS 2023；[代码](https://github.com/EleutherAI/concept-erasure) | closed-form 最小改变 concept erasure，并已在每层做 concept scrubbing；覆盖 odd-direction projection |
| 稀疏 attention-head truth steering | [Inference-Time Intervention](https://proceedings.nips.cc/paper_files/paper/2023/file/81b8390039b7302c909cb769f8b6cd93-Paper-Conference.pdf)，NeurIPS 2023 | 在线性 probe 最强的少量 heads 上沿 truth direction 改 activation；覆盖 sparse head-selective steering |
| RAG 中 external-context/parametric routes 的 head 机制及干预 | ReDeEP，ICLR 2025（[论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/7daf60e805e596c3bd1e843e72ea5560-Paper-Conference.pdf)） | 定位 Copying Heads 与 Knowledge FFNs，并通过 Add Attention Reduce FFN 调制两条来源路径；已占据 RAG source-route mitigation 的宽泛叙事 |
| with/without retrieval 的 token-level internal difference | [CORTEX](https://arxiv.org/abs/2606.31033)，arXiv 2026 | 比较有无 retrieved documents 的内部表征并跟踪信息沿 prefix 传播；覆盖 context-induced internal signal 的检测 |

没有检索到 exact duplicate 同时使用 `same-report polarity twin + patient ownership + image conflict +
donor-to-current edge projection`。但当前候选实际上还没有测到 ownership，因此暂时不能使用这个
窄 novelty。

## 更精细的 Source-Typed Attention Firewall

更精细版本先用 RadGraph/小模型标记 `OTHER_PATIENT_STATE` spans，仅调制这些 source tokens 到
current-image clinical claim query 的 attention edges；同一 head 对知识 tokens、image tokens 和
其他文本 tokens 保持不变。动态强度由 same-report twin odd edge flow 与 claim-specific image
evidence 决定。它明显优于整 head suppression，但仍有强近邻：

- [TAF: Taming the Phantom](https://ojs.aaai.org/index.php/AAAI/article/view/37768)（AAAI 2026）已经在
  token level 识别误导视觉理解的 `phantom textual tokens` 与关键 `anchor visual tokens`，并以
  training-free asymmetric filtering 直接调制中间 attention maps。这是**最强表面机制碰撞**：
  都是“只削弱危险文本 token influence、保留/增强视觉 token influence”。
- [AttentionRAG](https://arxiv.org/abs/2503.10720) 已用 query-to-context attention 做 retrieved-token
  pruning；PruneHal 等工作也根据视觉注意分配动态裁剪 KV tokens。它们目标主要是 context/KV
  pruning，不保留同一 token 的无害子空间，也不研究 patient ownership。
- [SECOND](https://arxiv.org/abs/2506.08391) 逐步选择、整合并对比多尺度 object-centric visual
  information。它同样证明“细粒度选择比全局删除好”，但干预对象是图像尺度/区域，不是
  source-typed text-to-output edges；**不是公式同构**。
- 一般 provenance/taint RAG 在系统层保留 chunk/source IDs；它通常不修改 Transformer 内部的
  source-token-to-sink edge，因此也不是 exact duplicate。

所以 Firewall 不能把“token-level asymmetric attention filtering”列为贡献——TAF 已占据；其
剩余差异必须同时依赖：**显式患者类型、反事实 odd flow、current-claim target edge、以及只移除
state subspace 而非删除 token。** 少任何一个限定，都会退化为 TAF/SPIN/AttentionRAG。

还有一个实现陷阱：`OTHER_PATIENT_STATE span` 并不等于若干显式 polarity words。

```text
No pleural effusion.        # polarity 有显式 no
Small pleural effusion.     # positive polarity 没有独立词，状态分布在整个 noun phrase
```

若直接 mask 整个 `small pleural effusion`，可运输术语和解剖知识也被删除，效果可能等价于 context
pruning；若只 mask `no`，又无法处理 affirmative state。因此 RadGraph 只能定义候选 source span，
真正干预仍应是该 span 经 polarity twin 识别出的 **odd value/output subspace**。必须与以下基线
比较：删除整句、mask 整 span、仅 mask negation、AttentionRAG pruning、TAF、以及同能量 random edge。

另一个可识别性问题是“current-image-claim query”在自回归生成前并不天然已知。论文必须冻结
一种不偷看答案的定义，例如所有 report FINDINGS 生成 positions，或由已生成 prefix 的结构状态
触发；不能用事后 RadGraph 标出 hallucinated output token 再回放并声称在线 mitigation。

## 当前版本最致命的问题：测到的是 polarity，不是 ownership

same-report polarity twins 只改变：

```text
OTHER patient: pleural effusion present
OTHER patient: pleural effusion absent
```

它们的 activation difference 测量的是 **effusion present vs absent**。这条方向可能正是模型回答
当前图像所必需的临床极性表示。它没有回答模型是否知道这句话属于 OTHER patient。

若论文称其为 `Source-Ownership Circuit Breaker`，必须至少有 \(2\times2\) factorial：

| source | polarity + | polarity - |
|---|---|---|
| CURRENT_PATIENT | “This patient's image shows effusion.” | “This patient's image shows no effusion.” |
| OTHER_PATIENT | “Another patient's report shows effusion.” | “Another patient's report shows no effusion.” |

在词汇、finding、长度和位置匹配后，对每个 head 分解：

\[
P_h=\frac14[(a_{C,+}-a_{C,-})+(a_{O,+}-a_{O,-})]
\]

为一般 polarity，

\[
S_h=\frac14[(a_{C,+}+a_{C,-})-(a_{O,+}+a_{O,-})]
\]

为 source identity，而

\[
B_h=\frac14[(a_{C,+}-a_{C,-})-(a_{O,+}-a_{O,-})]
\]

才是 **source × polarity binding**。论文需要定位的是错误的 donor-state 运输路径，而不是所有
polarity-sensitive heads。

如果只用 \(P_h\) 做投影，最可能发生的结果是：减少 positive hallucination 的同时增加 omission
或 false negative；这与“切断来源污染”不是一回事。

## 第二个关键升级：切 edge，不切整颗 head

一个 attention head 可同时承担语法、医学术语复制、图像整合与 donor-state 搬运。整 head suppression
已经被 ICLR 2025、SPIN、V-ITI、VIB-Probe 充分占据，也会造成明显 collateral damage。

更窄且更合理的干预对象是 donor tokens 对当前 answer query 的直接贡献：

\[
c^{D\rightarrow q}_{h,t}
=\sum_{j\in D}A^h_{t,j}V^h_jW^O_h.
\]

只对该 edge contribution 投影掉 dev-estimated source-bound donor-state subspace：

\[
\tilde c^{D\rightarrow q}_{h,t}
=c^{D\rightarrow q}_{h,t}
-G_{h,t}U_hU_h^\top c^{D\rightarrow q}_{h,t},
\]

其他 source positions、同一 head 的语法功能、current-image contribution 和 activation norm 均保持。
这才配得上 `Circuit Breaker`；全 head projection 只是 concept erasure。

## current-image vs null 不能直接定义视觉真值

任意 null image 往往是 OOD。\(a_h(X)-a_h(X_{null})\) 只能说明“这个 head 对图像变化有反应”，
不能证明它携带正确的 effusion 证据，也不能判断 donor polarity 与图像冲突。

最低限度应同时使用：

- current image vs same-support image swap；
- current image vs matched opposite-reader-support image；
- norm-matched corruption 与 mean-token null；
- claim-specific teacher-forced margin，而非总 activation norm；
- 在 VinDr reader votes 上验证 visual direction 随 0/3、1/3、2/3、3/3 单调变化。

只有映射到同一 claim-polarity coordinate 后，\(\langle o_h,v_h\rangle<0\) 才可解释为临床冲突。

## 与 provenance / taint RAG 的边界

常见 RAG provenance 保存 document/chunk ID 或给输出 claim 配 citation；NeuroTaint 等工作跟踪
untrusted source 到 privileged sink 的语义传播，[Lookback Lens](https://arxiv.org/abs/2407.07071)
和 REFIND 则量化输出对上下文的依赖。这些工作说明“来源追踪/污染传播”不是新问题。

仍可能保留的差异是：外部 metadata 能告诉我们内容来自哪个患者，却不能保证冻结 VLM 在内部
继续尊重这个归属；这里研究的是 **ownership tag 在生成路径中何处与 clinical state 脱绑定**。
该主张必须有 layerwise binding loss 与 path intervention，不能只展示 attention heatmap。

## ICLR delta 的 KEEP / KILL 边界

### KILL

- “找到 hallucination heads 并压制它们”；
- “视觉注意低时动态干预”；
- “用正负样本 activation difference 找方向”；
- “把不需要的方向投影掉”；
- “在医学 RAG 中首次做 source-aware attention”；
- 全 head suppression/projection 作为主要算法贡献。

### CONDITIONAL KEEP

将论文中心改成：

> **Clinical source ownership is represented separately from claim polarity, but their binding fails along a sparse donor-to-answer circuit; hallucination occurs when donor polarity survives after the ownership signal decays.**

这需要以下三项缺一不可：

1. **Factorial mechanism：** source × polarity 完整设计证明 source identity 与 clinical content 可分，
   且错误病例出现 ownership decay 而 polarity 仍存活；
2. **Path causality：** patch ownership component 能阻止 OTHER claim 搬运，却保留同样内容作为 CURRENT
   evidence 的作用；随机/同范数/path-matched patch 无效；
3. **Edge-selective rescue：** 只切断 donor-token-to-current-answer 的 state component，优于 SPIN、
   ICLR-2025 hallucination-head intervention、V-ITI、Owl、ITI、LEACE 和直接删 RAG，且固定 OE coverage
   后 omission 不增加。

## 生死实验

1. **四格 ownership gate：** 两模型、至少三个 findings 中，OTHER 与 CURRENT 的 polarity transport
   显著不同；若没有 source × polarity interaction，停止该方向。
2. **错误特异性：** donor-induced FP/FN 中 binding score 应显著异常，而 matched TP/TN、no-RAG 和
   irrelevant-donor control 不异常；image-cluster bootstrap CI 排除零。
3. **因果双向性：** 把 CURRENT ownership patch 到 OTHER state 应增加 transport；把 OTHER ownership
   patch 到 CURRENT state 应降低 transport，且 polarity identity 不变。
4. **edge vs head：** edge projection 必须优于整 head suppression、random edge、same-energy projection
   和 SPIN top-k；否则新颖性退化为已有 head intervention。
5. **实际 hallucination：** CE 只是定位；OE/report 固定 claim 数、长度与 ontology coverage，fabrication
   相对 raw-RAG 下降至少 20%，omission 不增，non-polarity knowledge 收益保留。

## 最终评分

| 版本 | 机制新颖性 | 方法新颖性 | ICLR 判断 |
|---|---:|---:|---|
| odd head + low visual + whole-head suppression | 低 | 低 | **KILL** |
| odd direction + selective head projection | 中低 | 低；CAA + LEACE + ITI | **不够** |
| source×polarity binding + visual conflict gate | 中高，但需实证 | 中 | **边缘 KEEP** |
| ownership decay 的 layer/path 规律 + donor→answer edge causal breaker + OE 无遗漏交换 | 高潜力 | 中高 | **有 ICLR delta，值得先做 gate** |

一句话：**“选择性投影 head”并不新；“模型保留临床极性却丢失它属于谁，并能只切断错误来源到
当前患者的那条边”才可能新。**

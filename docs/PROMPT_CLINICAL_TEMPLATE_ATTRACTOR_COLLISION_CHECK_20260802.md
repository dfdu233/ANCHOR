# Prompt-Triggered Clinical Template Attractor：窄碰撞审计

日期：2026-08-02  
判定：**作为独立论文主线 NO-GO；作为 Clinical Presupposition screen 的诊断性失败模式保留。**

## 1. 当前本地现象

Huatuo 的 VinDr generation-only job 采用 label-blind image selection、固定三种 pragmatic prompts、greedy decoding，并明确标记 `formal_clinical_claim_evaluation=false`。因此当前输出只能作为机制 lead，不能直接作为临床结论。

在任务仍运行时的只读快照中，已经出现很强的 prompt-conditioned template concentration：

- existential prompt 的主导 10-token prefix 是 `the chest x-ray shows a right-sided pleural effusion, which is ...`；
- negative-obligation prompt 的主导 prefix 是 `this chest x-ray shows no common abnormalities such as consolidation, ...`；
- neutral prompt 的主导 prefix 是 `the chest x-ray shows clear lungs, normal heart size and ...`；
- 同一个完整的 right-sided pleural-effusion 报告，在独立 reader panel 的 VinDr `0/3` 与 `3/3` pleural-effusion images 上都逐字出现；一个快照中的最常见版本覆盖 54 个已完成 existential outputs，其中 vote bins 为 `0/3:46, 1/3:3, 2/3:3, 3/3:2`。

这说明“跨标签图像进入同一输出 basin”是真实而强的候选现象。但它尚未说明是视觉感知失败、prompt/system attention、训练语料模板记忆，还是 autoregressive prefix lock-in。

## 2. 直接碰撞

### 2.1 医学 phenomenon 已被 2026 Template Collapse 直接命名并量化

[Generating Reports or Repeating Templates?](https://arxiv.org/abs/2605.30984) 已把医学 VLM 在不同扫描上输出少数重复报告正式定义为 **Template Collapse**，并用 clinical fidelity、output diversity、normal-template bias、dominant-template concentration 和 rare-finding survival 评估。

其最危险的结果不是语义相似，而是 exact-output collision：CT-CHAT 在 1,564 个 CT 中把同一个完整 normal report 逐字输出 452 次（28.9%），作者明确写道这些报告“regardless of what pathology is present in the scan”；SAMF 为 20.8%。图像直接输入的 VLM ablation 仍有超过 25% 的报告是同一 template。该工作还提出 CLarGen，通过显式 pathology detection、pathology-guided retrieval 与冻结 LLM synthesis 缓解 collapse。

因此以下 claim 均已碰撞：

- 医学报告会退化为少数模板；
- 完全相同报告会跨 pathology-discordant images 出现；
- visual conditioning 本身不能阻止 template collapse；
- decouple `what to say` 与 `how to say` 可缓解它。

Huatuo 的 right-sided effusion 模板是 2D CXR、positive template，而上述论文主要是 3D CT、normal-template bias；这属于重要复现和边界扩展，不足以把“Template Collapse”重新作为新问题。

### 2.2 Prompt mechanism 已被 ACL 2026 大幅占据

[Mechanisms of Prompt-Induced Hallucination](https://aclanthology.org/2026.acl-long.1941/) 在受控 VLM 实验中证明 prompt 可以压过图像，并定位少量 prompt-induced-hallucination attention heads；消融这些 heads 可把 PIH 降低至少 40%。虽然其任务是一般图像 object counting，不是医学 OE report，但“prompt 触发 hallucination、内部 heads 介导 prompt copying、head ablation 恢复视觉纠正”这一机制叙事已经存在。

[System-Mediated Attention Imbalances Make VLMs Say Yes](https://aclanthology.org/2026.findings-acl.1940/) 又把 system attention、coarse default representation 与 yes-bias 做了 causal attention redistribution。它特别指出 binary yes-bias 与 autoregressive hallucination 可能机制不同；这允许我们研究 OE，但也意味着仅把 dominant prefix 归因于 system/prompt attention 不够新。

所以“找到一组 prompt/template heads 并 ablate”很可能只是 ACL 2026 PIH 的医学复刻。

### 2.3 Cross-image subtraction 已被 Pensieve 直接覆盖

[Pensieve](https://arxiv.org/abs/2403.14401)（ECCV 2024）已经提出 training-free 的 `retrospect-then-compare`：检索语义/外观相似的真实参考图像，在相同文本上下文下计算候选分数，并从 test-image logits 中减去 reference-image confidence scores，以抑制相似图像共同诱发的 hallucination。

这与“用 in-distribution cross-image common template distribution 做 subtraction，只保留 image-specific residual”在算法骨架上等价。把 COCO references 换成 VinDr、把 common object 换成 pleural-effusion template、把随机/相似图换成 label-discordant CXR，都只是 domain-specific reference construction，不能独立支撑方法新颖性。

VCD、ICD 等工作又分别覆盖 distorted-image 与 disturbed-instruction contrastive decoding。因此当前命名下的 `in-distribution cross-image template subtraction` 应当停止，不投入正式 baseline-scale 实验。

### 2.4 Templated professional language 也已被评价工作识别

[X-ray Made Simple](https://aclanthology.org/2026.findings-acl.1726/) 指出专业放射报告的 templated nature 会让模型通过复制 common patterns 人为抬高 BLEU 等 lexical metrics，并用 layman-style data 减少模板依赖。这进一步压缩了“现有指标掩盖模板复制”的新颖空间。

## 3. 最终科学定位

当前现象最合理的定位不是一个新主线，而是：

> Huatuo 在特定 OE pragmatic task 下表现出 prompt-conditioned Template Collapse；这一失败说明任何 hallucination mitigation 必须先通过 template-collapse admission，否则所谓改善可能只是从一个模板切到另一个模板。

它可为 Reader-Grounded Two-Plane 主线提供边界证据：如果模型的 reader-vote polarity 在生成前表征中都不能超过同支持 image-swap 漂移，则它是 perception-limited；如果 polarity 可解码但输出跨 vote bins 仍坍缩为同一模板，则它属于 decoder/template-collapse-limited。二者不能用同一个 commitment projection 处理。

## 4. 允许继续的最小验证

只完成低成本诊断，不立项新方法：

1. 等 600/600 generation 完成，再冻结 snapshot；当前移动中的百分比不得进入论文。
2. 对每个 prompt 报告 normalized exact Top-1 share、10-token prefix Top-1 share、unique-report rate、T80 和语义-template concentration。
3. 对每个模板报告 reader-vote-bin distribution、跨 `0/3↔3/3` exact collision rate，以及 `I(template; reader_bin | prompt)`；按 image cluster bootstrap。
4. 排除相同 DICOM/hash、相同病人/检查、renderer collision、prompt echo、长度截断和 stop-template artifact。
5. 在至少第二个医学 VLM 复现；单 Huatuo 只作为 model-specific failure。
6. 用 neutral/existential/negative-obligation 的 within-image transition matrix 区分 prompt-triggered basin switch 与一般低 diversity。

以上通过后，仍只得到一个高质量 diagnostic result。除非出现下述更强证据，否则不做 hidden-state 方法：

- reader-bin polarity 在 pre-generation representation 中显著可解码；
- 随生成模板 prefix 增长，image-specific causal effect 在某一 token/layer 突然坍缩，而非从 encoder 起就缺失；
- patching 该转折前后的状态能选择性恢复 image-conditioned clinical claim，且不通过缩短、拒答、统一正常或换模板获益；
- 该转折机制跨两个模型成立，并在机制上区别于 PIH-head ablation 与 Pensieve score subtraction。

达到这些条件后可把它称为 **clinical autoregressive lock-in**，但仍更适合作为现有主线的模型边界或一个强 mechanistic section，而非目前直接升级为 ICLR oral 主问题。

## 5. 决策

- `Prompt-triggered clinical template attractor` 作为新 phenomenon：**NO-GO，Template Collapse 已直接覆盖。**
- `report-template memorization / identical OE answers across label-discordant images`：**NO-GO 作为新 claim；保留为强本地复现与 admission metric。**
- `in-distribution cross-image template subtraction`：**NO-GO，Pensieve 构成直接方法碰撞。**
- `clinical autoregressive lock-in after preserved early visual evidence`：**仅作条件性机制 screen；在出现 layer/token causal transition 前不立项、不命名为贡献。**

## 6. 600/600 冻结诊断结果

完整 generation hash 为 `1335d7c...cd4d81`；冻结 reader-template
diagnostic v2 hash 为 `a3960f2...ba7518`。所有数字均来自 200 个 label-blind
图像的三条件完整 triplet：

| prompt | exact unique | exact Top-1 | exact T80 | prefix-10 unique | prefix-10 Top-1 |
|---|---:|---:|---:|---:|---:|
| neutral | 91 | 29.0% | 51 | 15 | 80.0% |
| existential | 62 | 29.0% | 22 | 12 | 62.0% |
| negative-obligation | 45 | 36.0% | 12 | 2 | 99.5% |

200 个原始 DICOM file hash 与 200 个 renderer pixel hash 均唯一；无 prompt
echo、无字面 refusal，仅 existential 有一个 cap hit。另一方面，官方去标识
DICOM 的 PatientID、Study/Series/SOP UID 全为空，因此不能证明 collision 来自
不同患者/检查。这个缺失使第 4 项 fail closed，而不是被 image hash 替代。

对 `3 prompts × 8 findings` 的 template-reader mutual information 做 2,000 次
label permutation 后再进行 Benjamini--Hochberg 校正，仅
`neutral × lung_opacity` 存活（`q=0.04798`）。因此数据既不支持“纯模板、完全
无 reader 信息”，也不支持“reader 信息控制生成”：正确边界是少量 reader
signal 与强 prompt-template concentration 并存。由于 patient/study linkage、
第二模型与 token/layer causal transition 三项均缺失，
`paper_mechanism_authorized=false`；不运行 hidden-state canary。

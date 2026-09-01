# Clinical Claim-Boundary Re-grounding Failure：substrate 与碰撞审计

日期：2026-08-03  
判定：**当前本地 MIMIC/OE substrate 严格 NO-GO；不计算 position--error 曲线，不启动 GPU。**

## 1. 被审计的问题

候选问题不是“回答越长越容易错”，而是一个更窄的状态转移假设：

> 当模型从一个已完成的临床命题进入下一个原生临床 claim 时，本应重新读取
> 图像；幻觉是否来自新 claim 边界处未能恢复 image-causal dependence？

设第 (j) 个原生 claim 的起始边界为 (b_j)，在固定病例、模型、prompt、
claim 数和长度后，定义边界恢复量

\[
R_j=D(x,c_j\mid y_{<b_j})-
    D(x,c_{j-1}^{\mathrm{tail}}\mid y_{<b_j}),
\]

其中 (D) 必须来自 original / same-support image / opposite-support image 的
边界局部因果差，而不能由 attention 大小、单个 null image 或整段平均 NLL
替代。真正的机制预言是：supported 新 claim 在边界处有正的 (R_j)，而
unsupported claim 独有 `failure-to-recover`；绝对 token position 的平滑下降
不能产生这个 boundary-local reversal。

这一定义在看任何 claim-position outcome 前冻结。当前审计只检查能否识别该
量，不匹配生成 claim 与参考 claim，也不读取 sealed confirmation。

## 2. Outcome-blind 本地 substrate 结果

确定性审计 artifact：
`corrected_runs/claim_boundary_regrounding_audit_v1/substrate_audit.json`，
fingerprint
`5c04dfaaeaf698af53a6dc19246f3bbd30039d18aed8f4943dabd942e6fe29fc`。

### 2.1 顺序可恢复，但没有独立视觉真值

MIMIC `claim_action_audit_mimic_v1` 有 694 个对齐病例。RadGraph records 没有
显式 `claim_ordinal`，但可由每个 component 的 `start_ix` 无冲突恢复文本顺序：

| 方法 | 自动抽取 claim | 可恢复文本位置 | 显式 ordinal |
|---|---:|---:|---:|
| greedy | 212 | 212 | 0 |
| DoLa | 208 | 208 | 0 |
| PAI | 263 | 263 | 0 |

这只证明文本顺序存在，不证明 claim 的图像支持状态。参考侧是**单份 MIMIC
报告经过同一自动 RadGraph/ontology converter 的结果**；既有分析明确声明
`evidence_grade=C`，claim ceiling 是 “single reference report plus automatic
RadGraph extraction; not clinical truth”。

尤其不能误用 `ClinicalClaim.state`：代码中的 `supported/refuted` 只是由**该
文本自身**的 present/absent polarity 和 uncertainty 派生，并没有将 claim 与
图像、读者或参考报告核对。因此把它当作 visual truth 会构成循环标签。

结论：`independent_per_claim_visual_truth=false`，单这一项已经触发 NO-GO。

### 2.2 真正的多 finding 序列几乎不存在

| 方法 | 至少 2 个抽取 claim 的报告 | 至少 2 个不同 image-grounded finding 的报告 | 至少 3 个 claim |
|---|---:|---:|---:|
| greedy | 98 | **2** | 14 |
| DoLa | 97 | **1** | 13 |
| PAI | 119 | **2** | 12 |

绝大多数所谓多 claim 序列是同一 finding 的重复提及，而不是从一个临床命题
切换到另一个命题。用这些 rows 估计 claim-boundary recovery，会把 restatement
误写成 proposition transition；固定病例/claim 数后也没有统计支持。

预冻结 prevalence gate 为每个模型至少 100 个含两个不同 image-grounded
findings 的病例。当前分别只有 2/1/2，明确失败。

### 2.3 多 claim 子集本身由少数 exact templates 支配

| 方法 | 多 claim 子集的 exact unique reports | 最大 exact template share | 不同 ordered claim sequences |
|---|---:|---:|---:|
| greedy | 9 | **64.29%** | 6 |
| DoLa | 8 | **64.95%** | 5 |
| PAI | 18 | **65.55%** | 6 |

因此即使自动参考给出一个 ordinal error slope，它也可以完全由 dominant
template 中固定的 claim 顺序产生。只有 1--2 个真正多 finding 病例时，加入
template fixed effect 或排除重复模板都会耗尽可识别样本。预冻结 gate 要求
最大 exact-template share 不超过 20%；三模型全部失败。

### 2.4 有图像干预，但粒度不对

本地找到三类已生成的 MIMIC image interventions：

| artifact | n | 已有干预 | 逐 claim CF answer/score |
|---|---:|---|---:|
| `huatuo_evidence_dg/report_n16_controls_v1` | 48 | zero-visual、整段 teacher-forced mean NLL | 0 |
| `section_substitution/huatuo_mimic_n24_v1` | 24 | shuffled image、整段 impression mean NLL | 0 |
| `information_mismatch/huatuo_token_dependence_v1` | 52 | correct/shuffled/zero、预设 token-group mean NLL | 0 |

整段平均 NLL 不能定位某一个 native claim boundary，也不能知道 changed image
是否改变了该 claim 而保持其它 claim、claim 数和长度不变。三个 artifact 均无
counterfactual generated answer，也无逐 claim counterfactual score。因此
`per_claim_image_counterfactual=false`。

## 3. 为什么不能报告一个“稍有信号”的位置曲线

四个必要 gate 的当前值为：

```text
exact_case_alignment                              true
independent_per_claim_visual_truth                false
per_claim_image_counterfactual                    false
minimum_100_multiclaim_cases_per_model            false
largest_exact_template_share_at_most_0_20         false
formal_mechanism_analysis_authorized              false
gpu_authorized                                    false
```

在这些条件下，`hallucination ~ claim ordinal` 的任意回归都会混合：

1. 单份参考报告的未提及/报告习惯与真实图像支持；
2. 自动抽取错误和 ontology 漏检；
3. dominant template 的固定内部顺序；
4. 相同 finding 的重复提及；
5. 绝对 token length 与一般语言退化；
6. 方法之间不同的输出长度和可抽取 claim 数。

固定回答长度或 claim 数不能补救缺失的真值和反事实。当前不生成 effect size、
p 值或方向性图，避免事后用弱标签误导后续研究。

## 4. 机制级碰撞边界

| 最近工作/本地分支 | 已覆盖的机制 | 对本候选的判定 |
|---|---|---|
| [PAS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Hoang_PAS_Prelim_Attention_Score_for_Detecting_Object_Hallucinations_in_Large_CVPR_2026_paper.html) | 对每个 object token，hallucination 与高 prelim dependence、低 conditional image information 相关；用 prelim attention 做 detector | **直接占据弱版本**。“后序 claim 更少看图/更易错”只是 PAS 的医学与 ordinal 迁移。 |
| [More Thinking, Less Seeing, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/777db387a5ccb131ba8c7cd155166b85-Abstract-Conference.html) | 随生成/推理链变长，visual attention 和 perception fidelity 下降 | 任何平滑 ordinal/length slope 都是直接长度解释；必须出现 boundary-local recovery/reversal 才能区分。 |
| 本地 `Clinical Autoregressive Lock-in v5` 条件协议 | 自然序列上随 prefix 增长，固定 embedded claim 的视觉残差何时消失 | 高度相邻。新 delta 只可能是**完成 claim 后本应重新 grounding 的 reset 失败**，而不是固定 claim 的 prefix lock-in。v4 已因非自然拼接被否决，不能复用其结果。 |
| [Template Collapse](https://arxiv.org/abs/2605.30984) | 不同医学影像退化为少数 exact reports，包含 normal-template bias 与 rare-finding loss | 当前多 claim 子集 64--66% 被单一 exact template 支配，位置效应首先属于此解释。 |
| [OPERA, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_OPERA_Alleviating_Hallucination_in_Multi-Modal_Large_Language_Models_via_Over-Trust_CVPR_2024_paper.html) | 生成中对 summary/self-attention patterns 的过度信任与 retrospection-based mitigation | 仅显示既有 token 的 attention concentration 或重复不能构成新边界机制。 |

PAS 的数学对象已经是 (I(v;Y_k\mid y_{<k},t))，所以换成临床 claim、把
object token 聚合成句子或按 ordinal 作图都不是新机制。剩余的唯一实质 delta
必须同时满足：

- 新的独立临床 claim 开始时，supported claim **确实恢复** image-causal
  dependence；
- hallucinated claim 在相同长度/位置下独有 failure-to-reset；
- claim boundary 的局部变化超过 matched non-claim sentence boundary、标点
  boundary 和 within-claim pseudo-boundary；
- boundary-only causal intervention 恢复图像依赖和 claim correctness，同时
  保持 claim identity、polarity、claim 数和回答长度。

否则本候选应永久归入 `PAS + long-text drift + template collapse`，不另命名。

## 5. 唯一允许的未来最小实验

只有新 substrate 同时通过以下 admission，才能重新打开 CPU mechanism probe：

1. 至少两个模型；每模型至少 100 个自然 OE/report 病例，每例至少两个**不同**
   image-grounded claims；病例与患者在 dev/test 间隔离。
2. 每个原生 claim 由两名独立临床读者判定 `supported/refuted/undetermined/
   unobservable`，冲突经独立 adjudication；自动 RadGraph 只给 span，不给 truth。
3. claim boundary 在 reviewer 看任何 dependence score 前冻结；保留原生顺序，
   不删除 claim，不按结果选择句子。
4. 对同一 teacher-forced native sequence 记录 original、same-support image swap、
   opposite-support swap；只在真实图像可观察 claim 上分析。任意 null 仅作 OOD
   sensitivity，不作主证据。
5. 边界恢复量与 matched punctuation/non-claim/pseudo-boundaries 比较；模型中含
   case fixed effect、absolute token position、local/full length、claim count、
   exact/semantic template ID、finding 与 section。
6. 预注册双向检验：supported↔refuted swap 必须使 claim-local dependence 与
   polarity 双向变化，same-support swap 仅定义 nuisance floor。
7. 只有在 held-out 数据中 `boundary × clinical-support` interaction 显著、且
   PAS、平滑 length trend 和 template fixed effects都不能解释时，才进入 causal
   patching。patch 不得删 claim、缩短回答、统一阴性或增加拒答。

关键 falsifier：若所有 claim boundary 都不出现视觉依赖恢复，或恢复与普通
句号相同，则不存在 claim-specific re-grounding process；观察到的后序恶化归于
一般 autoregressive/length drift。负结果仍有意义，但不能催生 mitigation。

## 6. 最小可复现实验

本阶段唯一合法的“实验”是 outcome-blind substrate audit：

```bash
cd /home/dbw/ANCHOR
python anchor/corrected_sgta/audit_claim_boundary_regrounding_substrate_v1.py \
  --reference corrected_runs/claim_action_audit_mimic_v1/reference_claims.json \
  --claim-action-summary corrected_runs/claim_action_audit_mimic_v1/summary_complete_n694.json \
  --prediction greedy=corrected_runs/claim_action_audit_mimic_v1/greedy_claims.json \
  --prediction DoLa=corrected_runs/claim_action_audit_mimic_v1/DoLa_claims.json \
  --prediction PAI=corrected_runs/claim_action_audit_mimic_v1/PAI_claims.json \
  --counterfactual zero_visual=corrected_runs/huatuo_evidence_dg/report_n16_controls_v1/raw.jsonl \
  --counterfactual section_swap=corrected_runs/section_substitution/huatuo_mimic_n24_v1/raw.jsonl \
  --counterfactual token_dependence=corrected_runs/information_mismatch/huatuo_token_dependence_v1/raw.jsonl \
  --output /tmp/claim_boundary_substrate_audit.json
```

正式 repository artifact 使用相同输入写入
`corrected_runs/claim_boundary_regrounding_audit_v1/substrate_audit.json`。auditor
拒绝覆盖已有输出，保存全部输入/代码 SHA-256，并明确记录未做 reference match、
未算 position outcome、未读 sealed confirmation、未使用 GPU。

## 7. 决策

- 当前 `Clinical Claim-Boundary Re-grounding Failure`：**NO-GO**。
- 当前可报告结果：**本地 MIMIC 输出不具备识别该机制所需的逐 claim 真值、
  反事实粒度、多 finding prevalence 与 template diversity。**
- 禁止结果：后序 claim hallucination rate、ordinal slope、boundary recovery
  effect、任何 decoding mitigation gain。
- 未来保留的窄问题：在独立临床 truth 下，原生新 claim 是否存在区别于 PAS 和
  length drift 的 image-grounding reset，以及该 reset 是否可被 claim-preserving
  causal intervention 恢复。


# MedVIGIL 是否解锁 Evidence-Source Erasure：substrate 审计

日期：2026-08-02  
结论：**NO-GO。MedVIGIL 不能作为 Evidence-Source Erasure 的正式机制 substrate。**

## 1. 冻结问题与判定标准

待检验机制不是一般的 image-vs-text reliance，而是：对于同一个原子临床 claim，模型是否先保留证据来自 `current_image / history / prior_study / knowledge` 的身份，随后在 decoder 中抹掉该身份，因而把来自错误来源、过期来源或不可观察来源的内容写成当前图像事实。

要解锁表示层实验，release 至少必须同时提供：

1. 同一原子 claim 的独立临床 source labels；
2. 每个 source 对 claim 的 `support / refute / unavailable` 状态；
3. 保持 claim、polarity、certainty 和答案语义不变，只交换 evidence carrier 的 source-preserving pairs；
4. patient/image-group-disjoint 的 dev/test 划分；
5. 可审计的图像与文本来源、许可和医生 adjudication。

MedVIGIL 在上述五项中只部分满足第 5 项，因此不能进入 hidden-state probe、activation patching 或 mitigation。

## 2. 审计对象与可复现快照

- 论文：[MedVIGIL, arXiv:2605.07919v2](https://arxiv.org/abs/2605.07919)，2026-05-22 修订。论文声明 300 cases、2,556 probes、240 counterfactual triplets、四位放射科医生监督，并提供 paired open-ended variant。
- 数据：[官方 Hugging Face release](https://huggingface.co/datasets/jhq0709/MedVIGIL)。审计了 `manifest.csv`、`grounding.csv`、`probes_mcq.csv`、`probes_open.csv`、`triplets.csv`、`CXR_RECONSTRUCTION.md`、`DATASHEET.md`、`PIPELINE.md` 和完整 repository tree。
- 代码：[官方 GitHub evaluation harness](https://github.com/hq0709/MedVIGIL)，本地审计 commit `dc7ad9d2e9ca2cc78515b764b3e932f4035a0d44`。GitHub README 明确说明只发布 evaluation code，不发布 construction-side tooling。
- 小型 metadata 的本地 SHA-256 已记录在审计临时目录；没有运行模型或 GPU。

## 3. Source-label 审计

正式 release 的关键 schema 为：

```text
manifest:
case_id, source_dataset, modality, image_file, question, gold_answer,
risk_tier, text_only_answerable, annotator_notes

grounding:
case_id, answer_rationale, roi_pointer, roi_bbox_norm, differential_set,
indication, modality_detail, laterality_dependent, gold_after_lr_flip

triplets:
case_id, anchor_question, anchor_gold, tcf_question, tcf_gold,
vcf_condition, vcf_question, vcf_gold, image_file, validation_flags
```

没有以下任何字段：`claim_id`、原子 claim tuple、`evidence_source`、source-specific support、source span、prior-study identifier、history timestamp、current/prior study link 或 source-rater votes。

`source_dataset` 只是 VQA-RAD、SLAKE、ROCO、CXR 的上游数据集 provenance，不是临床证据来源。`indication` 是单个上下文字段，不提供 history 的 claim-level 支持，也没有与 current image 或 prior study 的独立对应关系。

因此 release 中可以可靠标注的只有“当前图像 ROI 是否被遮挡”等 image-evidence condition，不能标注“这个 claim 来自当前图像、历史、既往检查还是一般知识”。

## 4. 为什么现有 probes 不能构造 source-preserving causal pairs

### 4.1 `knowledge_only` 是任务重写，不是证据来源交换

[官方说明](https://huggingface.co/datasets/jhq0709/MedVIGIL#layer-c--probe-battery-probescsv)把它定义为“rewritten to be answerable from textbook knowledge alone”，并允许使用新的 knowledge gold。实际 release 有 283 个 knowledge-only probes：

- 原问题与 knowledge-only 问题的 token-set Jaccard 中位数只有 `0.10`；`258/283` 不超过 `0.25`；
- 只有 `148/283` 的 expected answer 字符串完全相同；相同答案也不代表相同 proposition；
- 例如“这张胸片是否提示主动脉夹层？”被改成“主动脉夹层通常能否仅靠胸片确诊？”——前者是个体图像 claim，后者是一般诊断知识；
- knowledge-only probe 仍带与 original 相同的 `image_file`，并没有显式移除图像或标注模型实际使用的来源；
- `text_only_answerable` 在 manifest 中为 `No=240, blank=55, TRUE=4, FALSE=1`，与 283 个 knowledge-only probes 不构成一致的正式 source label。

所以 original↔knowledge-only 同时改变了问题、命题及部分答案，不能用于 patching 后宣称只改变了 evidence source。

### 4.2 Counterfactual triplets 改变假设，不交换来源

[官方数据说明](https://huggingface.co/datasets/jhq0709/MedVIGIL#layer-d--counterfactual-triplets-tripletscsv)明确写明：T-CF 是同图 paraphrase；V-CF 保持图像不变，在问题中加入 hypothetical condition，并改变 gold；没有合成 counterfactual image。

240 个 triplets 使用 240 个 image filenames，但不存在第二个临床 source，也没有同一 claim 从 image 搬运到 history/prior/knowledge 的 pair。T-CF 适合测措辞稳定性，V-CF 适合测 hypothetical blindness；二者都不是 source-preserving evidence transport。

### 4.3 ROI probes 只改变图像可见性

`roi_only` 与 `roi_masked` 可测试当前图像区域的必要性和充分性；它们仍属于单一 `current_image` source。ROI mask 可以作为 directional admission 或 grounding control，但无法识别 history/prior/knowledge source identity，更无法证明 source identity 在层间被 erased。

### 4.4 “Open” 是同一 probe 的自由回答封装

`probes_open.csv` 与 MCQ 的 2,556 个 `(case_id, probe_id, axis, kind, question, behavior, explanation, image_file)` 公共字段逐项完全相同，公共字段 mismatch 为 `0/20,448`。它扩展的是回答格式，不是报告生成、长期病历输入或多来源 annotation。开放回答不会凭空补出 source label。

### 4.5 历史/既往字段实际不存在

manifest 的 300 个原始问题中，`prior / previous / history / comparison / follow-up` 等词的命中为 0。全部 probes 中只有 15 行、11 个 case 命中；主要是 hallucination trap、条件问题或偶然知识问题，没有 prior image/report object、时间关系或 claim-level source adjudication。它们不能组成纵向 source dataset。

## 5. Release 完整性、split 与泄漏风险

### 5.1 图像并非 300 例自包含

HF repository tree 实际有：

- `images/`: 240 files；
- `images_perturbed/`: 540 files；
- 60 个 `source_dataset=cxr` 的原图缺失。

这与 README/DATASHEET 中“300 original / self-contained / ~900 perturbed”的描述不一致。`CXR_RECONSTRUCTION.md` 正确说明 60 个 CXR 来自 credentialed MIMIC-CXR/CheXpert，不能再分发；但它引用的 manifest “per-case provenance field”和 triplets `provenance_anchor` 在实际 schema 中不存在，且 `scripts/reconstruct_cxr.py` 标为 camera-ready forthcoming，官方 GitHub 中也没有该脚本。当前 release 无法重建这 60 例。

### 5.2 声明的 split 与原始标注目录未发布

HF README 列出 `splits/`、`splits/text_only_subset.json` 和 `raw_clinician/`；完整 tree 中均不存在。DATASHEET 同时明确该 benchmark 没有 train/val/test split。HF viewer 显示的 auto-generated `train` 不能当作机制学习的官方 split。

任何 learned probe 或 steering vector 都必须由我们自行创建 dev/test，而且必须先解决图像重复组。

### 5.3 byte-identical 图像可跨 case 泄漏

240 个已发布 original image filenames 只有 191 个唯一 LFS byte OID：

- 28 个重复图像组；
- 77 个 case 位于重复组；
- 最大重复组含 11 个 case。

DATASHEET 声称 `image_group` metadata 捕获此问题，但实际 manifest 没有该字段。若按 `case_id` 随机切分，等同一张图可能同时进入 probe fitting 与 test。最低要求是按 image byte hash 分组，并进一步按 patient/study 分组；后者目前缺 provenance，无法完整保证。

### 5.4 MCQ answer-position 泄漏

2,556 个 MCQ 的 correct-letter 分布为：`A=1,723, E=792, B=37, C=3, D=1`。其中 original、TCF、negation、specificity-drop、knowledge-only、ROI-only 全部把正确答案放在 A；hallucination traps 与 ROI-masked 全部放在 E；只有 laterality flip 出现少量 B/C/D。

因此 MCQ 可被强 position/probe-family shortcut 污染，不能用来定义 hidden-source truth 或比较细小机制效应。若作辅助行为测试，必须独立打乱 answer order 并报告 order-randomized sensitivity。

## 6. 许可边界

- HF dataset card 顶层声明 `CC-BY-NC-SA-4.0`。
- GitHub `LICENSE` 声明 harness code 为 MIT，annotations/metadata 为 CC-BY-4.0，source images 保留上游许可。
- GitHub README 又列出 VQA-RAD CC0、SLAKE CC BY-SA 4.0、ROCO CC BY-NC-SA 4.0、MIMIC-CXR/CheXpert credentialed access。

保守执行方式是：代码按 MIT；annotations 至少遵守 CC-BY-4.0 attribution；整个 HF package 和任何衍生分发按更严格的 CC-BY-NC-SA-4.0/per-source 条款处理；MIMIC-CXR/CheXpert 图像不再分发。不能把 300 例当作统一、自由再发布的图像集。

## 7. 最近机制碰撞

| 工作 | 已覆盖内容 | 对 Evidence-Source Erasure 的边界 |
|---|---|---|
| [Medical Context Distorts Decisions in Clinical VLMs](https://arxiv.org/abs/2605.17436) (2026) | 在 MIMIC-CXR 上系统交换 image/text，加入 1–5 个无关历史报告，发现文本覆盖图像、无关 history 造成 negative flips；代码公开 | 任何只报告“text/history dominates image”的行为工作已经严重碰撞。新工作必须是同 claim、source-controlled、layerwise 和 causal。 |
| [SDLS](https://arxiv.org/abs/2602.23676) (2026) | 定位并 steering “historical comparison” latent axis，抑制不存在 prior 时的比较幻觉 | 任何“找 prior direction 然后减掉”的方法已直接碰撞。真正未覆盖的是：模型在 prior 确实存在时能否保留并正确使用 source identity，以及 identity 在何层消失。 |
| [CMC-Bench](https://aclanthology.org/2026.magmar-main.3/) (2026) | 一般领域 image/text 的 aligned、image-correct、text-correct、both-wrong 四格冲突 | 一般的双模态 source arbitration 已有。医学贡献需四类临床来源、原子 claim 和医生 source adjudication。 |
| [Treble Counterfactual VLMs](https://aclanthology.org/2025.findings-emnlp.1000/) (EMNLP Findings 2025) | 用视觉、文本与跨模态 counterfactual direct effects 做因果 hallucination mitigation | 任意粗粒度 modality ablation 不足以构成新机制；必须区分临床 source identity，而不只是 image/text modality。 |
| [HalluTrace](https://aclanthology.org/2026.alvr-main.29/) (ALVR 2026) | 把一般 LVLM 幻觉归因为 visual grounding failure、language-prior dominance、cross-modal conflict，并 source-target decoding | 其“source”是 failure source，不是临床 provenance；但已占据通用 causal-source attribution 叙事，迫使我们把 claim 缩到临床 evidence provenance erasure。 |
| [CXR-ReDonE](https://arxiv.org/abs/2210.06340) (2022) | 清理不存在 prior 的报告引用并重训 | 仅删除 prior-reference 的数据清理不是新机制，且会损害真实纵向能力。 |

此外，[Harrison.Rad 1.5](https://arxiv.org/abs/2607.05880) 已明确把 images、priors 和 clinical context 作为真实报告输入，说明多来源问题本身重要；它不提供公开的 claim-level source-erasure substrate。

## 8. 最终判定与允许用法

**Evidence-Source Erasure 维持 halt。不得基于 MedVIGIL 启动 layerwise source probe、source-direction steering，或写出“source identity 在后层消失”的论文主张。**

MedVIGIL 仍可作为以下辅助用途：

1. 用 240 个可用病例做 broken-image behavioral admission；
2. 用 ROI-only/ROI-masked 检查模型是否真正使用当前图像；
3. 用 original/TCF 检查 prompt robustness；
4. 用 hallucination traps 测 false-premise refusal；
5. 所有统计按 image-byte-hash cluster bootstrap，MCQ 必须随机化 option order，60 个缺图 CXR 单独标记 unavailable。

这些用途只能支持“行为边界/入场测试”，不能支持 hidden-source mechanism。

## 9. 重新解锁所需的最小数据契约

若以后基于 MIMIC-CXR 纵向记录或另一合规数据集重建 substrate，manifest 至少应包含：

```text
patient_group, study_group, claim_group, claim_id,
finding, polarity, uncertainty, anatomy, attributes,
source_slot in {current_image, history, prior_study, knowledge},
source_state in {support, refute, unavailable},
source_span_or_roi, source_timestamp, source_observability,
reader_1_source_state, reader_2_source_state, adjudicated_source_state,
pair_id, pair_role, image_hash, split
```

每个 source pair 必须形成同 claim 的最小四格：both-support、A-only、B-only、neither；正式冲突实验再加入 A-support/B-refute 与相反方向。pair 内冻结 claim wording、polarity、certainty、输出长度预算和患者层真值，只交换承载证据的 source。pilot 至少需要六个 source pairs 各 50 个独立 claim groups；formal gate 应扩到各 100 个，并按 patient/study/image hash 切分。

在这个契约出现前，最可信的决定不是“用 MedVIGIL 凑一个 source probe”，而是明确 no-go，避免把 task rewrite、hypothetical reasoning 或 ROI corruption 错写成 evidence-source erasure。

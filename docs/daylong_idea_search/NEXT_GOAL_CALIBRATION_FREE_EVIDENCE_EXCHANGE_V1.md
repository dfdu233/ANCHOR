# 下一阶段 Goal：Calibration-Free Evidence Exchange

## Objective

在不影响 baseline 队列的前提下，用 24 小时发现并验证一种面向开放式医学
VLM 生成的 training-free 直接纠错原语。冻结子问题为 **fabricated positive
finding**：报告生成了图像中不存在的阳性 finding，同时常伴随真正 finding
被遗漏。

允许相对上一 Goal 放宽且只放宽一项：可调用一个冻结、轻量、独立的医学视觉
专家提供病例级视觉分数；不得训练 VLM/专家，不得检索，不得使用测试标签，
不得让专家直接生成文本。

方法必须保持每例原始阳性 claim 数 `K`，只允许一换一纠错；不得靠删除、缩短、
拒答、统一阴性或增加 hedge 获益。目标是一个对 VLM 架构、专家分数单位和
tokenizer 均不敏感的证据交换原语，而不是新的评测器、校准器或 ensemble 权重。

## Seed hypothesis, not frozen method

异构 VLM 与小视觉专家的 raw score 不在同一尺度，直接加权没有自然含义。
先检验最小的 calibration-free 操作：若遗漏 claim `j` 在 VLM 病例证据与独立
专家证据上都严格高于已生成 claim `i`，则以 `j` 一换一替换 `i`。

数学对象是二维序数偏序：

`j ≻ i  iff  rank_VLM(j) > rank_VLM(i) and rank_expert(j) > rank_expert(i)`。

该关系在两路分数分别经过任意严格单调变换后不变；每次交换都让两路证据的
rank potential 同时上升，因此有限候选集上必然终止。不得把这两个标准的序数
性质冒充理论创新；新意只能来自医学开放生成中是否存在稳定、可利用的
“共同支配错误”，以及它是否给出无需权重的固定内容纠错规律。

## Work order

1. 公式级检索 VEP、CoEV、Pelican、CGD、product-of-experts、rank fusion、
   Pareto decoding、set prediction 与 fixed-cardinality correction；若已有等价
   claim-exchange 操作，立即关闭。
2. 使用 VinDr reader-vote truth、已有 VLM 输出/claim margin 与冻结视觉专家，
   在 dev/confirmation image-disjoint split 统计“共同支配错误”上限：实际生成
   false-positive `i` 是否被某个 omitted true-positive `j` 在两路证据上共同支配。
3. L0 生死门：Huatuo、Hulu 各自至少 20% 的 baseline FP 存在这样的真交换，
   image-bootstrap 95% CI 下界大于 10%；否则不写解码代码，关闭该方向。
4. L1 实现无标签交换：输入只含草稿 claims、固定 ontology、VLM 分数和专家分数；
   不读取 ground truth。与 expert-only top-K、加权和、product-of-experts、Borda/
   reciprocal-rank fusion、随机交换和 VEP-style prompting 比较。
5. L1 GO：固定 `K` 后，confirmation hallucination 相对下降至少 20%，omission
   同时下降，exact-set accuracy 不降，bootstrap 95% CI 排除 0；至少两模型成立。
6. 只有 L1 GO 后才用不超过 2 小时 GPU 扩展到自然语言报告：先抽取草稿 claims，
   交换后保持 finding 数与报告模板，重新表述；输出长度差控制在 ±5%。
7. OE/report GO：两个数据集上 fabricated positive 相对下降至少 20%，matched
   coverage 下成立，临床 finding recall 不降，clear-case 下降不超过 1pp。

## Hard stop and pivot

- 若专家只帮助 Huatuo、不帮助 Hulu，结论是弱模型补偿，不称通用方法。
- 若增益由 expert-only top-K 完全解释，降级为普通 specialist ensemble，关闭。
- 若只在 oracle `K`、oracle claim parser 或测试集阈值下成立，关闭。
- 若 Pareto seed 失败，下一分支只能增加真实病例信息（第二物理视图、DICOM
  acquisition metadata 或独立 modality）；不得退回层融合、attention、style/DG、
  NCD/DID、crop/mask、RAG、energy 或另一个 output-only 公式。

## Deliverables

交付公式级 collision 表、L0 headroom JSON、完整无标签实现、两模型 held-out
结果、fixed-K 错误配对样本，以及 GO/NO-GO 决定。没有通过上述门槛，不得使用
方法名或声称 ICLR-ready。

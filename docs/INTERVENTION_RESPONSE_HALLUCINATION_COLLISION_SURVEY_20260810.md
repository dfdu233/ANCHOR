# 干预响应能否成为医学 VLM 幻觉的新主线：碰撞调研（2026-08-10）

## 研究问题

1. 多种 prompt、RAG、图像变换或 decoder 的响应差异，是否已被用于幻觉检测与缓解？
2. 当前真实提分究竟支持普通 ensemble，还是支持新的因果/可辨识机制？
3. 在最新工作之后，还剩什么可验证且足够重要的研究空间？

检索覆盖通用 VLM、医学 VLM、LLM uncertainty、active evidence acquisition、逻辑
一致性与 test-time routing；只使用可核验的论文主页、论文全文或官方仓库。2026
preprint 只作为竞争证据，不等同于已通过同行评审。

## 证据分类

| 分支 | 代表工作 | 已覆盖内容 | 对当前项目的约束 |
|---|---|---|---|
| 语义扰动与一致性 | [SAC3](https://arxiv.org/abs/2311.01740)、[Prompt Multiplicity](https://aclanthology.org/2026.eacl-long.327/) | 对语义等价 query 的回答一致性做检测；后者证明检测器常学到一致性而非正确性 | 不能把 disagreement 直接称为 hallucination truth |
| 语义保持干预 UQ | [ESI](https://arxiv.org/abs/2510.13103) | 用干预前后 token 分布变化估计 epistemic uncertainty | “干预响应比单次置信度更有信息”本身不新 |
| 医学图像对比解码 | [VGS-Decoding](https://arxiv.org/abs/2603.20314) | 原图/失真图 token 分布差异，自适应抑制医学幻觉 | distorted-vs-original score 与响应融合直接碰撞 |
| 图像多视图 | [RITUAL](https://arxiv.org/abs/2405.17821)、[SECOND](https://proceedings.mlr.press/v267/park25c.html)、[AIR](https://arxiv.org/abs/2602.24041) | 随机变换融合、多尺度选择、选择性 patch 强化 | 多风格、病灶 patch、自动选择视图都不能单独主张新颖 |
| 逐样本选择 | [QueryBandits](https://arxiv.org/abs/2602.20332)、[V-ITI](https://arxiv.org/abs/2512.03542) | 按 query 选择改写策略；检测 visual neglect 后才干预 | CMP/普通 router 的核心命题已被占据 |
| 主动证据获取 | [BCEA](https://arxiv.org/abs/2606.16667) | 在预算内选择 zoom/crop/claim intervention，并重新校准风险 | value-of-information acquisition 不能只靠医学换皮 |
| 逻辑闭环 | [Logical Implications for VQA](https://arxiv.org/abs/2303.09427)、[V-Loop](https://arxiv.org/abs/2601.18240)、[CGD-PD](https://arxiv.org/abs/2604.06196) | 已知问题逻辑关系、医学双向验证、正命题/否命题三态投影 | “正问+反问+校验码”直接碰撞 |
| 医学扰动审计 | [PSF-Med](https://arxiv.org/abs/2602.21428)、[MedVIGIL](https://arxiv.org/abs/2605.07919)、[MetaRA](https://arxiv.org/abs/2605.19307) | 医学 paraphrase、negation/ROI、图像×问题 metamorphic testing | 医学措辞敏感、联合扰动和 ROI 证据审计都不是空白 |

## 与真实结果的交叉判断

当前 CXR 五折结果证明：患者对干预的配对响应含有可泛化预测信息。两模型 plain-only
stack 的增益不显著；加入两条 RAG 响应后 BAcc 提升约2.6pp，且跨患者打乱 RAG
配对会丢失约2.7pp。Knowledge-MIMIC 训练的线性响应基迁移到 CXR 后，也比
plain-only 高约2.75pp。

但特征消融又显示，绝大多数增益来自 RAG 最终 Yes/No，而非置信度曲率或隐藏几何。
因此当前证据首先支持“低相关、患者对齐的专家输出可被 stacking 利用”，尚不支持
新的内部机制。把它称为 intervention-response geometry 会夸大证据。

## 剩余研究空间

尚未检索到机制完全等价的工作是一个更窄的合取问题：

> 对具有独立多读者真值的 image-grounded clinical claims，能否设计一组临床语义
> 明确的干预，使其响应形成可辨识的最小基；该基的有效秩或类间距离能否在不看目标域
> 标签时预测哪些错误可被纠正，并通过患者错配、matched-compute、自一致性和因果搬运
> 对照证明它不是普通 ensemble？

这个方向只有满足以下条件才是“新机制/规律”，否则仍是 Kaggle stacker：

1. 干预基的选择准则只在 source/dev 拟合，却能预测未见 domain/model 的纠错收益；
2. 相同数量的随机采样、独立模型 plain 输出和普通 beam/self-consistency 无法解释增益；
3. 患者配对被破坏后增益消失，但每个专家边际准确率保持；
4. 至少一个因果搬运实验把被识别的状态送入最终生成，并同时降低 FP 与 FN；
5. OE/report 固定 claim coverage 后仍有效，而非只修二元 Yes/No；
6. 用 VinDr reader votes 分离清晰病例与真正 ambiguous 病例，不把一致性当真值。

## 结论

RQ1：答案是“已被广泛使用”，普通扰动、一致性、router、主动获取和逻辑校验均拥挤。

RQ2：当前 +2.6～2.75pp 是可信的竞赛信号，但最保守解释仍是 supervised linear
stacking；尚未获得因果机制。

RQ3：唯一值得继续的高层问题是**干预基的跨域可辨识边界**，而不是再发明一种融合
公式。其最小 source-only basis 实验、matched-compute 对照和多读者 causal test 将
决定该方向是否保留。

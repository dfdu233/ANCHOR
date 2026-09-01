# Patient-State Noninterference（PSNI）候选协议

## 核心问题

医学 RAG 检索到的报告通常来自**其他患者**。其中同时混有两类信息：

- 可迁移知识 `K`：疾病术语、典型影像征象、解剖关系；
- 不可迁移状态 `Z_r`：该患者“有/没有胸腔积液、左侧、小量”等事实。

对当前患者的 image-grounded claim，理想系统应满足：

\[
P(Y\mid X,K,Z_r)=P(Y\mid X,K,Z'_r),
\]

即替换其他患者的状态，不应改变当前患者结论；这对应信息流安全中的
**noninterference**。当前候选现象称为 **retrieval polarity transplantation**：
语义相关但属于别人的 present/absent 状态穿过 RAG 边界，被移植成当前患者事实。

这不是普通“检索不相关”：危险文本可以完全正确、与问题高度相关，只是主语属于另一
患者。它也不是继续寻找 DG 风格中心；研究对象从像素域偏移改为推理阶段之间的证据
所有权泄漏。

## 当前动机证据（仅 discovery）

- Hulu CXR 的 raw RAG 将 GT=No 的 FP 从10.33%增至16.57%（+6.24pp），同时对
  GT=Yes 的 FN 略有改善，符合阳性状态注入而非统一变差。
- plain↔RAG 分歧仅占29.85%，却覆盖92.90%的 stack rescues 与88.75%的 harms。
- 旧 patient-shuffle placebo 已被纠正：在完全相同 question 内交换 RAG 响应后，
  paired code 不再更好。因此 stacking 不再作为机制证据；PSNI 必须直接操纵检索文本
  的患者状态，而不是利用输出组合。
- 上述 Hulu CXR 输出未通过冻结质量门，故所有数字只授权设计 causal pilot，不授权
  论文结论。

## 最简方法：Clinical State Firewall

1. 将检索报告拆为原子 tuple：`finding + polarity + uncertainty + anatomy + attributes`。
2. 对其他患者的 tuple 将 patient-state polarity 标为 untrusted；知识术语和一般影像
   关系标为 transferable。
3. 构造 state-neutral context，或检索同 finding 的正/负 matched pair 使状态先验相消。
4. 训练时对同一图像随机交换 `Z_r`，最小化当前 image-grounded claim 对 `Z_r` 的
   条件依赖；knowledge claim 不施加该约束。
5. 部署时仍只生成一次；OE/report 先拆 claim，再按 evidence source 应用 firewall，
   不通过删 claim、缩短报告或统一拒答获益。

## Gate 0：文本极性传递（CPU，已完成；仅观察性通过）

- CXR-VisHal 与 Knowledge-MIMIC；Huatuo/Hulu；plain/RAG。
- 只用预注册 finding lexicon 和局部 negation；unknown 单列。
- 检验 retrieved polarity 是否预测 plain→RAG flip 的方向，并控制 question/finding、
  GT、长度、BM25 rank；image/study cluster bootstrap。
- 若两个模型/数据方向均不成立，立即淘汰 PSNI。

冻结规则后的一次性全量分析得到：CXR-VisHal 中，RAG 相对 plain 向检索
报告极性移动的比例，Huatuo 增加 13.12pp（cluster 95% CI 10.21--15.46），
Hulu 增加 8.26pp（5.49--10.24）。Knowledge-MIMIC 只有 negative/no-state
移植跨两模型稳定，positive-state 不稳定。因此 Gate 0 只支持“存在来源状态污染”这一
发现，不支持对称、普适的极性复制定律。现有 shuffle 仅有 5 个反极性有效样本，不能
提供因果确认；禁止将上述观察性数字写成方法效果。

## Gate 1：真正的因果交换

对同一 `X,K` 构造三种等长、同 finding、同写作风格 context：positive patient state、
negative patient state、state-neutral。只改变 `Z_r`，随机顺序并绑定哈希。

通过条件：

- polarity swap 使 claim log-odds 同方向移动，二模型、多个 finding、独立 confirmation
  成立；
- neutral context 显著减小该效应；
- unrelated-finding、同 polarity 改写、长度匹配和 shuffled-patient controls 不能解释；
- current image 被 same-label swap 时不应出现伪“患者特异”效应。

### Gate 1 substrate（2026-08-10，CPU 构建完成）

不读取 target、reader vote 或答案，共得到 108 个严格 donor pairs（VinDr 34、
CXR-VisHal 74）和 432 个四臂输入：present、absent、neutral、random deletion。
每个 pair 满足同 finding、query/present/absent 三患者互异、present/absent 字数差不超过
10%；中位长度差 4.04%，TF-IDF cosine 中位数 0.326。neutral 与 random deletion
逐 pair 删除相同字数，前者删除 target-state claim，后者保留 target-state claim，从而
区分语义状态移除与普通文本删除。

该 substrate 只证明可实施性，不是模型结果。aortic enlargement 与 pulmonary fibrosis
各只有一个且相似度约 0.04，正式主分析不得用它们凑 finding 数；emphysema、pleural
thickening、other lesion 无严格配对，已经按冻结规则停止而没有放宽匹配。

## Gate 2：竞赛效果与论文效果

比较 no RAG、raw RAG、MR-RAG/relevance purification、conflict-aware RAG、正负平衡
retrieval、Clinical State Firewall：

- CE：相对 raw RAG 的 FP 至少下降20%，相对 no-RAG 的 BAcc 提升，FN 不增加超过1pp；
- OE/report：固定 claim coverage，fabricated finding 减少且 omission 不增加；
- 无医生时 OE/report 只作 benchmark proxy，不称临床 hallucination truth；
- 至少两个模型、两个独立数据域；环境、输入和生成哈希全部绑定。

## Novelty kill list

必须直接对照 MR-RAG、KERM purification、TCR/conflict-aware RAG、CRAG、Stable-RAG、
contrastive RAG。若结果只能由更好的 relevance ranking、普通正负 exemplars、context
denoising 或 no-RAG 解释，则方法降级为工程 RAG baseline，不进入 ICLR 主线。

额外的直接碰撞包括 RULE（EMNLP 2024）、MMed-RAG（ICLR 2025）、RADAR
（ACL 2025）、FactMM-RAG（NAACL 2025）和 CF-RAG（ICLR 2026）。因此“RAG
会误导模型”“过滤不相关文本”“使用反事实检索”均不能作为新颖性。唯一暂存的研究
增量是：**疾病语义可跨患者运输，但患者状态不可运输**，并以 matched state swap
证明 source-role-specific noninterference，而非把 firewall 文本清洗本身作为贡献。

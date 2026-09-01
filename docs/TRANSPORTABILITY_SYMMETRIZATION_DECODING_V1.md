# Transportability Symmetrization Decoding（TSD）候选

> **2026-08-10 公式级裁决：作为主算法 KILL。** CAFP 2026 已对二元
> counterfactual 输入做同构的双查询输出平均；Frame Averaging/Reynolds operator 已覆盖
> 群平均，token-logit 平均也是标准 logarithmic opinion pool。以下内容仅作为机制假设与
> intervention baseline 保留，不得作为论文标题或主要算法贡献。唯一可继续的问题是：
> other-patient state 是否真形成 polarity-odd circuit，而可运输知识形成 even circuit。

## 一句话

对同一份其他患者报告构造“患者状态反转”的 twin context，在每个生成步平均原报告与
twin 的完整词表 logits；报告中的解剖和疾病知识不变，阳/阴性、侧别/程度等不可运输
患者状态成对相消，不再复制到当前患者。

## 从 DG 到医学 RAG 的核心抽象

其他患者报告同时包含：可运输的疾病语义 `K`，以及不可运输的患者状态
`Z in {-1,+1}`。先把报告解析成原子 clinical tuples，再由冻结、可逆的变换 `g` 只反转
其中 patient-state 字段，满足 `g(g(R))=R`；正文、术语和非状态知识保持相同。若下一
token 的 logits 可写成

\[
z(X,K,Z)=a(X,K)+b(X,K)Z+\epsilon(X,K,Z),
\]

则对同一报告的二元素轨道 `{R,gR}` 做群平均：

\[
z_{TSD}=\frac12\left[z(X,K,+1)+z(X,K,-1)\right]
       =a(X,K)+\frac12(\epsilon_++\epsilon_-).
\]

一阶 patient-state 项被精确抵消，图像证据与不变的疾病知识保留。这不是寻找所有
训练域的“中心”，而是只沿一个有明确临床来源含义的 nuisance 轴做对称化。部署冻结
模型，不训练参数；代价为两次前向。单次前向的 Clinical State Firewall 是其编译近似，
不是论文核心。

## 为什么可能比普通 RAG/过滤更强

- Raw RAG 只给一个病例：相似度高不代表该患者的阳性/阴性可用于当前患者。
- No-RAG 避免污染，但同时丢掉术语、视觉征象和解剖关系等潜在知识收益。
- 普通 relevance filter 无法删除“内容完全相关、事实也正确、但属于另一个患者”的
  状态。
- TSD 不需要判断该检索报告真不真实；它利用正负成对设计，使来源状态在数学上相消。

## 竞赛版与论文版

竞赛版先检索最相关报告，将其中所有 `OTHER_PATIENT_STATE` tuples 以冻结模板反转，逐
token 平均原 context 与 twin context 的完整 logits。这样只需两次前向，不随 OE claim
数量增长。可在 development OOF 上选择 retriever 和 top-k，但不在 test 上调阈值。

Matched positive/negative donor reports 只作为自然语料中的生态发现与外部有效性实验；
由于两个 donor 的 `K` 不可能完全相同，它们不能单独定义主算法或最严格因果结论。

论文版研究更一般的 **Cross-Patient Evidence Transportability**：

1. 只交换 donor state 是否会有方向地搬运当前图像 claim；
2. 这种搬运是否围绕 plain image evidence 近似反对称；
3. 群平均是否只消除不可运输状态，而保留可运输知识；
4. 哪些 claim（clear/ambiguous、present/absent、visual/knowledge）违反该规律。

## 预注册验证阶梯

### G0：方向与对称性

先以自然 matched donors 比较同图同问题的 present、absent、plain，再以同报告
polarity twins 作为正式操纵：

\[
T=E[m_+-m_-],\qquad
S=E[m_++m_--2m_0].
\]

`T>0` 表示 signed patient-state transport；`S` 接近 0 才说明简单群平均有依据。自然
donor 结果必须由 same-report twin 复现，才排除 donor 内容差异。必须跨
Huatuo/Hulu、至少 3/4 核心 findings 同向。当前离散生成 pilot 只作筛选；确认实验使用
teacher-forced Yes/No/Uncertain margins。

### G1：删除控制

从 positive donor 删除 target-state claim 应降低 Yes 倾向；随机删除相同 token 数但
保留 claim 不应复现。显式写“来自其他患者”仍不能消除效应，才说明来源提示不足。

### G2：算法效果

比较 plain、raw-RAG、no-RAG、正负答案投票、logit TSD、Clinical State Firewall、
RULE 与 MMed-RAG：

- CE：相对 raw RAG 的 FP 至少下降 20%，FN 不增加超过 1pp；相对 plain BAcc 下界
  不低于 -1pp，且最好有正增益；
- OE/report：固定 claim 数与长度，fabrication 下降且 omission 不增加；
- 非极性 knowledge probes 下降少于 1pp；
- 两模型、两个数据域；剩余 76 matched pairs 只打开一次。

### G3：DG/OOD 规律

donor 来源医院、报告风格、检索器和未见 finding 作为 held-out domains。若对称化只在
构造它的同一 BM25/MIMIC 风格上有效，则它是竞赛技巧，不是 DG 规律。

## 致命淘汰条件

- same-report twin 的 `T` 不跨两模型/三 findings，或 `S` 明显偏离 0，说明
  patient-state 不是可对称抵消轴；
- 自动 polarity inversion 不能在严格语义审计下达到足够 coverage，或 twin 变成明显
  不自然/矛盾文本；
- TSD 与普通投票、no-RAG 或删上下文无差别；
- FP 降低来自统一 No、uncertain/invalid 增加、回答缩短或 claim coverage 下降；
- knowledge 收益为零，此时最合理方法就是不使用 RAG；
- RULE/MMed-RAG 或 CF-RAG 在同设置已经实现相同 source-state symmetrization；
- 只能在 CE 成立，OE/report 固定 coverage 失败。

通过 G0 之前，TSD 只是一个高潜力候选，不是完成的方法；通过 G0/G1 但 G2 失败时，
论文最多保留“并非所有检索证据都可跨患者运输”的机制结果。

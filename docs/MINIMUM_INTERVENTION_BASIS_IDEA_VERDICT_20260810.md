# Minimum Intervention Basis：ICLR idea gate

## 1. First impression

- Paper type: Novel Method（当前版本）
- One-sentence story: 用源域标签选择少量干预输出，并线性组合以迁移到未见医学域。

## 2. Fatal-flaws audit

| # | Flaw | Severity | Evidence |
|---|---|---|---|
| 1 | 核心算法不是新的研究对象 | CRITICAL | Fisher/LDA 类间距离、best-subset search 与 logistic stacking 都是标准组件；SRM (ICML 2025)、TEP (ICML 2024)、MUSE (EMNLP 2025) 已覆盖 source-only/OOD ensemble、预测拓扑选子集或校准子集聚合。 |
| 2 | 核心选择规律被简单控制击败 | CRITICAL | Knowledge→CXR 原始 Spearman 0.969；控制 subset size、source-val performance、成员强弱和 error diversity/correlation 后 partial rho=0.375、p=0.125、delta R²=0.012，且 leave-one-arm-out 符号翻转。 |

## 7. Verdict

**Reject and Pivot.**

不能把 MIB/Fisher selector 作为论文核心，也不为它继续增加阈值或替换成 D-optimal
design。真实的竞赛增益保留为强 stacking baseline。只有出现以下不同研究对象才可重新
立项：同一模型、语义预注册的干预；干预机制与模型身份完全解耦；跨未见医院/任务；
并产生一个在控制单臂强弱和普通 ensemble 后仍成立的独特预测。

## Closest work

- [SRM, ICML 2025](https://proceedings.mlr.press/v267/qiao25b.html)
- [TEP, ICML 2024](https://proceedings.mlr.press/v235/qiao24a.html)
- [MUSE, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1551/)
- [QueryBandits, 2026](https://arxiv.org/abs/2602.20332)
- [VGS-Decoding, 2026](https://arxiv.org/abs/2603.20314)

本判决按 idea-evaluator 的 data-refuted mechanism 规则短路；因此不继续做五维打分或
论文逻辑模板。等新的候选通过 fatal gate 后再建立 ICLR skeleton。

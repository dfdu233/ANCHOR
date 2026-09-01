# Metric Calibration Side-Probe v2：n=8 两模型结果与停止决定

日期：2026-08-03 UTC  
结论：**STOP_AFTER_N8；不进入 n=97，不进入论文主线。**

## 构造与运行资格

- 只使用 corrected v2 manifest：`structured_neutral_v2` 与 `clinical_direct_v2`；v1 prompt 结果不进入判断。
- 8 张冻结 VinDr Nodule/Mass 图像，每模型 112 条 structured 与 32 条 direct。
- Qwen parent commit：`cc594898137f460bfe9f0759e9844b3ce807cfb5`。
- Huatuo medical commit：`451ac32400e36cfd07b41b62cbe63e6894895b38`。
- 两模型 structured JSON valid、nonempty 均为 1.00，cap-hit 均为 0；structured runtime gate 通过。
- Qwen direct 的 cap-hit 为 0.1875，集中在长篇 `header_unknown` 解释，因此 direct 只按独立 parser 与人工候选解释；Huatuo direct cap-hit 为 0。
- 本实验只评估 counterfactual law 与 patient-unit commitment legality，不评估真实 patient-mm accuracy。

## 主要结果

### 1. 结构化类型遵从高度模型特异

- Qwen oracle/vision：`missing` 与 `detector_only` 的 patient-mm overcommitment 均为 0；`header_unknown` 为 0.25。
- Huatuo oracle：`missing`、`detector_only`、`header_unknown` 均为 1.00。
- Huatuo vision：三者分别为 1.00、1.00、0.75。

Huatuo 即使被明确告知“只有 certified patient-plane calibration 才可输出 physical value”，仍在结构化 schema 中把不可识别状态写成 patient-mm；Qwen 基本遵从，只有 calibration provenance 不明时部分失败。这不是跨模型统一机制。

### 2. 自然 direct prompt 没有跨模型非法单位承诺

在 `missing`、`detector_only`、`header_unknown` 共 24 条/模型中：

- 两模型 explicit abstention rate 都为 1.00；
- 两模型 unqualified numeric-unit commitment rate 都为 0。

因此，结构化 schema 中的 Huatuo failure 没有迁移到 matched clinical direct prompt。最可能的解释是 prompt/schema interaction 或 instruction-following 差异，而不是稳定的自然临床 hallucination phenotype。

### 3. 合法 spacing arithmetic 不遵守 transformation law

- Qwen oracle 的 median log-value/log-scale slope 为 -1.572；Huatuo为 0.000，正确值应接近 1。
- x0.5/x2 的 median absolute log-ratio errors 明显非零。
- 两模型在 `certified_cm` oracle 下 expected-unit accuracy 均为 0：倾向把 cm metadata 重新写成 mm，而未做正确单位重表达。
- oracle endpoint drift 均为 0，说明失败发生在固定 dimensionless geometry 之后；但该 arithmetic/equivariance failure 已被 MedVision `scaledPS` 与 FactCheXcker/tool conversion 强覆盖，不能构成新颖贡献。

## 预注册 gate

- structured runtime：PASS；all-answer-contract runtime：**FAIL**（Qwen direct cap-hit 0.1875）；
- neutral structured overcommitment cross-model：PASS（由 `header_unknown` 达阈值）；
- dimensionless endpoint factorization：PASS（oracle endpoints冻结）；
- direct overcommitment cross-model：**FAIL**（两模型均为 0 < 0.20）。

最终 artifact：`corrected_runs/metric_calibration_probe_v2/two_model_pilot_decision_v3.json`，决定为 `STOP_AFTER_N8`、`n97_authorized=false`、`gpu_authorized=false`；它 supersede 只检查 structured runtime flag 的 v2 判定逻辑。

## 执行事故与修复

一个自动 full launcher 曾只检查 structured runtime gate，错误地把“接口合格”当成“科学扩展授权”。作业在首次终止后被 watchdog 短暂恢复，最终在 45/1358 条时再次终止；partial rows 永不进入分析。审计标记：`corrected_runs/metric_calibration_probe_v2/full_v2/ABORTED_GATE_VIOLATION.json`。

Launcher 已改为 fail-closed：除 runtime 外，必须同时满足预注册 decision 为 `EXPAND_TO_N97_DIAGNOSTIC_ONLY`、`n97_authorized=true`、`gpu_authorized=true`，并核对两份 analysis SHA；任一失败不启动进程。新增 STOP_AFTER_N8 集成测试并通过。

## 可保留的可信结论

两模型在固定端点下都不能可靠执行 spacing/unit transformation，但只有 Huatuo 在结构化三态 schema 中强烈违反 calibration type；当自然问题明确要求不可识别时不要猜，两模型都能拒绝。故当前证据支持“量化算术与 schema-specific 类型遵从缺陷”，不支持跨模型、自然临床的 calibration-state hallucination 机制。该方向保留为 negative side-probe，不再消耗 GPU。

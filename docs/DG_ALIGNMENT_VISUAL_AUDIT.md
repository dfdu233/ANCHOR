# DG 变换与跨模态对齐：当前问题定位

本轮不否定 DG 假设。相反，我们把问题改写为：**域/风格变换是否保留了临床证据，并且是否真的改变了视觉证据到 claim 的绑定，而不是只改变像素或解码 operating point。**

## 已实现的可视化

脚本：`anchor/corrected_sgta/visualize_dg_alignment_v1.py`

它是 CPU-only、后处理模块，不训练、不改写答案、不建立图像池。输入可以是：

- `style_phenomenon/*/raw.jsonl`（每个 style 有 yes/no logits、prediction、PSNR、edge correlation）；
- `huatuo_rule_mimic_feddg*/**/raw_generations.jsonl`（FEDD-G candidates、NLL、结构审计）；
- `huatuo_evidence_dg/**/raw.jsonl`（生成证据轨迹）。

输出：

- `dashboard.html`：自包含浏览器页面；
- `summary.json`：可复核的 variant 统计。

最小运行示例：

```bash
PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.corrected_sgta.visualize_dg_alignment_v1 \
  --input corrected_runs/style_phenomenon/huatuo_rule_mimic_n128_v1/raw.jsonl \
  --output-dir corrected_runs/style_phenomenon/huatuo_rule_mimic_n128_v1/dg_visual_audit \
  --image-root data/medheval/images
```

页面包含：

1. image change（`1-edge correlation`）— claim delta 散点图；
2. 每个 variant 的 prediction flip rate；
3. 每个样本的原图、claim delta、prediction、PSNR、edge correlation 和输出文本；
4. 机器可读的 interpretation，明确区分 evidence destruction 与 decoder/domain sensitivity。

当前 128-case style phenomenon smoke 已生成：
`corrected_runs/style_phenomenon/huatuo_rule_mimic_n128_v1/dg_visual_audit/dashboard.html`。

## 目前风格迁移对齐失败的具体位置

### 1. 变换参数的语义在不同实现中不一致

`methods.py` 中 `low_frequency_ratio` 是低频窗口半径，`source_ratio` 是窗口内的 source mixing；但 `frequency_alignment_source_spectrum.py` / `release2.py` 中同名参数被改成了**全频谱 residual 的 alpha**，并强制 `source_ratio == 0`。这意味着同一个参数名在不同 runner 中不是同一个干预：一个只改中心低频窗口，另一个混合全部非 DC 频率。

这会造成错误归因：实验以为在测试“低频风格”，实际可能在改变全频幅度。必须在可视化中同时显示实现版本、参数语义和实际频谱 mask。

### 2. 当前 source center 不是同一成像机制的可靠 target

FEDD-G runner 的默认 bank 是 `PubMedVision/train/ct__chest.npy`。将 CT chest 的频谱中心用于 CXR，会把 modality、重建方式和导出管线差异混进“style”。这不否定 DG，而是说明 source center 必须与目标 modality、projection 和 preprocessing 对齐，并记录 provenance。

### 3. 结构 gate 只测像素近似，不测临床证据保持

现有 gate 使用 PSNR 与 edge correlation。128-case 结果中，低频变换的 median PSNR 约 19，而 edge correlation 约 0.996；FEDD-G `source_ratio=0/0.5` 的 median PSNR 约 5/11、edge correlation 约 0.70/0.97。高 edge correlation 仍可能伴随强烈对比度、幅度和病灶可见性变化。

因此“结构 gate 通过”不能被解释为“病理证据不变”。需要增加 claim-specific evidence survival，而不是继续放宽 PSNR 阈值。

### 4. 输出变化主要可能是低 margin boundary，而非 DG 修复

style phenomenon 中 flip rate 约为 1.6%–3.1%，且既有结果显示 flips 集中在低 native margin。若只看 flip 数量，容易把边界敏感性误读成风格纠正。可视化必须同时显示 native margin、style drift 和 claim correctness。

### 5. 全局对齐没有传递到 decoder 的 claim binding

域对齐作用在像素/视觉表示；幻觉可能在 `visual evidence → claim commitment` 处发生，且生成历史 `H` 会提供语言先验。只对齐视觉均值或频谱中心，不能保证 decoder 在 cardiomegaly、effusion 等 claim 上使用了正确的条件证据。

## DG 保持不变时的下一步

先不改 Baseline，也不启动大规模 GPU。对每个 claim 做四格反事实：

```text
                 原始域       DG 域
证据保留          A             B
证据削弱          C             D
```

计算：

```text
style_effect_with_evidence    = B - A
style_effect_without_evidence = D - C
DG_interaction = (B - A) - (D - C)
```

解释：

- `B-A` 大、`D-C` 小：DG 可能改变了视觉证据读取；
- 两者都大：更像 decoder/domain prior；
- 只有 evidence 被破坏时变化：变换破坏了临床信息；
- 只在低 margin claim 变化：普通决策边界脆弱性。

只有当 interaction 在两模型、两个数据集上稳定预测 hallucination，且 geometry-insensitive control claim 不同步变化，才进入 inference-time claim calibration。否则 DG 仍然作为机制诊断工具，而不是被放弃。


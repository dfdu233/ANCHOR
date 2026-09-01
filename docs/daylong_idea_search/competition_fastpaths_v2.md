# Competition fast paths v2：CPU 缓存复用审计

## 结论先行

本轮没有占用 GPU，也没有停止 baseline。三个真实缓存实验得到一个可复用的竞赛增益、
一个更重要的共同失败边界、以及两个必须防止的评测取巧模式：

1. **finding-conditioned 双模型校准确有 +3.23pp BAcc**，但 FP 从 74 增至 91，
   它主要用更多阳性换回 FN，不能称为幻觉缓解。
2. **小病灶是跨模型共同瓶颈**：最小四分位中 Huatuo/Hulu 同时漏掉 26.7%，最大
   四分位仅 4.2%；即使使用知道真值的二模型 oracle，最小病灶 recall 也只有 73.3%。
3. **部分 mitigation 的改变是有选择性的，但仍不足以超过最好方法**：在保持
   `No→Yes`、`Yes→No` 数量完全相同的随机 placebo 下，OPERA/VCD 的样本选择分别
   多带来 +3.91/+4.65pp BAcc；然而二者相对 AvisC 的实际 BAcc 仍为 -1.46/-0.14pp。
4. **OE 的表面差异强烈混入“少说/不说”**：固定输出 claim 数量后，主要方法间
   claim-F1 差异的 patient-bootstrap CI 均跨 0；同时 K=1 覆盖率从 OPERA 的 3.7%
   到 PAI 的 91.0% 相差巨大。

因此，最值得带回主线的不是另一个 ensemble/router，而是两个机制判断：

- 方法是否有效，必须拆成“**改变多少阳性率**”和“**是否找对该改变的病例**”；
- 多模型互补不能解决共同的 sparse-evidence bottleneck，病灶越小，oracle headroom
  也越快耗尽。

## 实验一：相同 operating point 下，mitigation 是否真的找对病例

### 问题

一个方法把更多答案推成 Yes，可能提升 recall，但这不代表它找到了视觉证据。对每个
目标方法，构造 label-blind placebo：从 AvisC 的 No 中随机选出完全相同数量改成 Yes，
再从 Yes 中随机选出完全相同数量改成 No。这样 target 与 placebo 的 Yes-rate 和两个
方向的改动数一致，差异只来自“改了哪些病例”。

### 设置

- 数据：MedHEval CXR-VisHal。
- 模型：LLaVA-Med-v1.5-Mistral-7B。
- 方法：AvisC、DoLa、OPERA、PAI、VCD、VISTA。
- 只保留六方法都能由冻结 semantic diagnostic 解析为 Yes/No 的二元题：2,714 题、
  464 个 image clusters；这只是机制诊断，不替代严格主指标。
- 置信区间：5,000 次 image-cluster bootstrap。

### 结果

| Target（相对 AvisC） | 实际 BAcc 差 | 同方向 flip placebo 差 | 选择特异性 excess | 95% CI |
|---|---:|---:|---:|---:|
| DoLa | -6.55pp | -6.48pp | -0.06pp | [-0.32, +0.17] |
| OPERA | -1.46pp | -5.37pp | **+3.91pp** | **[+3.09, +4.72]** |
| PAI | -6.43pp | -6.46pp | +0.03pp | [-0.24, +0.30] |
| VCD | -0.14pp | -4.80pp | **+4.65pp** | **[+3.56, +5.79]** |
| VISTA | -6.55pp | -6.53pp | -0.02pp | [-0.25, +0.19] |

AvisC BAcc 为 56.88%，且 Yes-rate 高达 89.20%，所以绝对数不应被当成论文正式结果。
但负对照结论清楚：DoLa/PAI/VISTA 的改动没有超过“相同方向、随机选病例”，而
OPERA/VCD 确实找到了更有用的病例，只是 harms 仍抵消 corrections。六方法的
label-dependent oracle accuracy 比 AvisC 高 9.80pp `[8.75, 10.92]`，这是不可实现的
上限，说明互补存在但缺少无标签识别接口。

**科研机制：** mitigation 应分解成 criterion shift（改变工作点）与 selection
specificity（找对病例）。这是一种强审计原则，不是新的缓解算法。

## 实验二：病种条件专家与小病灶上限

### 问题

Huatuo 与 Hulu 的错误是否只是普通平均可消除，还是依赖 finding 类型与病灶大小？

### 设置

- 数据：VinDr-CXR，3/3 reader-positive 与 0/3 reader-negative。
- 开发 320 claims，confirmation 960 claims；两模型、8 findings 完全对齐。
- 只在开发集拟合两种 L2 logistic 校准：
  - pooled：Huatuo/Hulu 两个 final margin；
  - finding interaction：再允许每个 finding 对两种 margin 使用不同权重。
- 阈值也只在开发集选；confirmation 一次评估。
- 5,000 次 image bootstrap；同时报告 Yes-rate、FP、FN。

### 结果

| 方法 | BAcc | Yes-rate | FP | FN | 相对开发集所选 Hulu |
|---|---:|---:|---:|---:|---:|
| Hulu raw | 75.00% | 40.42% | 74 | 166 | — |
| Huatuo raw | 61.56% | 69.27% | 277 | 92 | -13.44pp |
| pooled 双 margin | 76.88% | 52.92% | 125 | 97 | +1.88pp `[-0.67,+4.41]` |
| finding interaction | **78.23%** | 47.19% | 91 | 118 | **+3.23pp `[+0.93,+5.60]`** |

病种条件增益是真实的 confirmation 正信号，但仍是标准 supervised calibration，且
FP 增加 17。它可作为竞赛模块或 upper bound，不能包装为 training-free hallucination
method。

在 480 个 reader-positive 且有框的 claims 上，按病灶 union area 四分位：

| 病灶大小 | Huatuo recall | Hulu recall | 二者任一正确 oracle | 二者同时遗漏 |
|---|---:|---:|---:|---:|
| 最小四分位 | 66.7% | 48.3% | 73.3% | **26.7%** |
| 次小四分位 | 75.0% | 70.8% | 87.5% | 12.5% |
| 次大四分位 | 90.0% | 70.8% | 95.0% | 5.0% |
| 最大四分位 | 91.7% | 71.7% | 95.8% | **4.2%** |

这里使用真实 reader box 与正类身份，只是诊断 oracle，不能用于推理。它说明：
模型互补对小病灶有部分 headroom，但最难的 26.7% 是共同盲区；简单 model ensemble
无法替代局部证据搜索。

**科研机制：** finding identity 决定模型偏置方向，而 lesion support size 决定两模型
共同失败概率。前者可被普通校准利用；后者才可能需要新的视觉证据机制。

## 实验三：固定 claim 数的 OE 比较

### 问题

开放报告中，一个方法可以靠短回答获得更高 precision。这里用固定的 14-concept
signed lexical extractor，把每份回答转成按出现顺序排列的
`finding:positive/negative` claims；比较任意两方法时，只保留二者和 reference 都至少
有 K 个 claims 的同一患者子集，并都截为前 K 个 claims。

### 设置

- 数据：Visual-MIMIC OE，490 图、490 患者。
- 模型与六种 mitigation 同实验一。
- K=1/2/3；每个 pair 单独使用共同 eligible 子集。
- 5,000 次 patient bootstrap。
- 该指标是确定性 lexical proxy，不是医生定义的 hallucination truth。

### 结果

首先，claim coverage 已经说明长度差异不能忽略：

| 方法 | 至少1 claim | 至少2 claims | native empty rate |
|---|---:|---:|---:|
| AvisC | 77.6% | 54.7% | 22.4% |
| DoLa | 39.8% | 39.0% | 60.2% |
| OPERA | 3.7% | 2.7% | 96.3% |
| PAI | 91.0% | 89.8% | 9.0% |
| VCD | 50.2% | 23.1% | 49.8% |
| VISTA | 6.3% | 5.5% | 93.7% |

在样本量至少 30 的 pairwise fixed-K 比较中，没有一个主要 pair 的 claim-F1 CI 排除
0。例如：

- AvisC−PAI，K=1，n=333：F1 +2.87pp `[-0.02,+5.81]`；
- AvisC−PAI，K=2，n=208：F1 +2.07pp `[-2.99,+6.99]`；
- AvisC−DoLa，K=1，n=150：F1 +2.13pp `[-2.75,+7.28]`；
- AvisC−VCD，K=1，n=186：F1 +2.64pp `[-1.20,+6.60]`；
- PAI−VCD，K=1，n=215：F1 +0.44pp `[-3.26,+4.03]`。

因此现有 OE 输出暂时没有可靠的 fixed-K 方法胜者；native 分数差异很大一部分与
是否生成可识别 claim 有关。下一算法必须同时报告 fixed-K claim quality 与 coverage，
否则“少说”会被误判为“少幻觉”。

## 对主线的影响

### 可以复用

1. finding-conditioned 双模型校准是可靠的 Kaggle 式 +3.23pp 上限，可作强 baseline。
2. operating-point matched placebo 应进入所有 CE mitigation 审计。
3. fixed-K + coverage 应进入所有 OE 新方法早期门控。
4. 小病灶共同遗漏支持继续检验 patch search，但它也显示普通 ensemble 的上限有限。

### 不应升格

1. finding-conditioned stack 是标准校准/ensemble，不是 ICLR 核心创新。
2. 六方法 oracle 使用标签，不可实现。
3. bbox size oracle 使用真值框，不能当 inference 方法。
4. 14-concept OE extractor 只可作快速 proxy，没有医生时不能声称临床幻觉率。

## 可复现产物

- 代码：`anchor/corrected_sgta/analyze_competition_fastpaths_v2.py`
- CE：`corrected_runs/daylong_idea_search_v1/competition_ce_operating_point_specificity_v3.json`
- VinDr：`corrected_runs/daylong_idea_search_v1/competition_vindr_model_complementarity_v3.json`
- OE：`corrected_runs/daylong_idea_search_v1/competition_visual_mimic_fixed_k_v3.json`

每个 JSON 均保存 dataset、model、method、seed、完整命令、输入 SHA256 与源码 SHA256。
旧 `v2` 文件由全方法 fixed-K 交集塌缩（K=1 仅2例）所取代，正式引用只使用 `v3`
的 pairwise fixed-K 结果。

独立复跑审计：主 agent 在新的临时输出目录完整重跑三项分析；除输出目录导致的
`provenance.command`字符串不同外，三个JSON均与正式`v3`逐字段完全一致。

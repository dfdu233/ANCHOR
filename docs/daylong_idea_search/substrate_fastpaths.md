# 两小时内可完成的研究致死实验：真实 substrate 审计

日期：2026-08-12

## 结论先行

当前最值得做的不是再造一个解码公式，而是依次回答两个尚未被仓库关闭的问题：

1. **明确的病灶局部信号能否被模型因果地读取？** 用真实病灶的删除—搬运，而不是普通模糊掩码。
2. **同一观测的变换不增加信息，真实第二次观测是否增加信息？** 用 IU-Xray 同一 study 的第二张片，与风格变换和错患者第二张片对照。

“评测标准变化造成假提升”已经有实质正信号，可立即作为方法筛选的审计规则，但它本身不是缓解算法。

本次没有占用 GPU、没有暂停或修改 baseline。审计时 GPU 上仍在运行 `baseline_llava_methods_v2` 的 CXR-VisHal VISTA chunk；所有 baseline tmux 会话仍在。

## 已核实的 substrate

| 资源 | 实际规模 | 能否直接复用 |
|---|---:|---|
| VinDr reader manifest | 3,200 claims，2,341 images，8 findings×400 | 可复用标签、split、DICOM 路径 |
| VinDr bbox manifest | 2,387 image-finding rows，1,773 images，8 findings 各约 297–300 | 可复用真实病灶框 |
| focal erasure | 128 claims / 126 images，Huatuo，original/lesion-erased/mirror-erased | 可直接 CPU 重分析 |
| lesion transplant | 旧版 2 例 smoke；当前 v2 可稳定选出 128 个独立 Nodule/Mass cases | DICOM 与选择可复用；v2 logits 需重跑 |
| Evidence Addressability | Huatuo/Hulu 各 532 confirmation images | 只保存 `pre/post mean/std`，没有逐 patch token |
| IU-Xray report test | 590 studies，每例恰好 2 张图，共 1,180 张，0 对字节级重复 | 可作真实第二观测 |
| IU-Xray fine-grained CE | 2,017 questions；其中 1,484 binary，覆盖 289 paired studies | 可抽 256 个 balanced、study-disjoint claims |
| MIMIC SISC | 653 studies 中仅 44 个多图 study，且只有 1 条独立 view-local truth | 不适合作为第一轮多视图实验 |
| CXR-VisHal completed scores | LLaVA 的 DoLa/VCD/OPERA/PAI/VISTA，各 5,587 questions / 475 image clusters | 可直接做 criterion audit |

重要边界：现有 532×2 模型的 visual cache 只有全局均值和标准差；16 例 domain-orbit JSON 只记录 token shape 与汇总量，不保存 `576×d` token tensor。因此，**“不跑 GPU、直接用现有缓存完成大规模 bbox patch probe”并不存在**。不能把 global summary 换名为 sparse patch evidence。

## 实验 0：病灶模糊掩码的现成致死结果（已完成，零 GPU）

### 在问什么

如果模型确实使用框内病灶，擦除 radiologist box 后，阳性 margin 应比擦除同形状镜像区域下降更多；这里 margin 是“回答 Yes 的分数减去回答 No 的分数”。

### 入口与数据

```bash
PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.corrected_sgta.analyze_vindr_focal_evidence_erasure_v1 \
  --raw corrected_runs/c3_guard/vindr_focal_erasure_n128_v2/raw.jsonl \
  --output /tmp/vindr_focal_erasure_reanalysis.json \
  --bootstrap-draws 10000
```

- `n=128` claims，`126` images；Calcification 64、Nodule/Mass 64。
- 原图、病灶擦除、水平镜像同形状擦除已经全部缓存。
- 预计 CPU 时间：小于 1 分钟。

### 已有结果

- 总体病灶擦除下降：`+0.029`，95% CI `[-0.049, +0.109]`，不成立。
- Calcification：`+0.105`，CI `[+0.035, +0.189]`。
- Nodule/Mass：`-0.047`，CI `[-0.176, +0.086]`，均值甚至反向。
- 病灶框相对镜像框的差值虽为正，但镜像擦除本身会提高 Yes 分数，说明掩码响应混有非临床的全局工作点变化。

### 判定

**已 NO-GO：普通“只遮病灶”不能作为通用方法。** 它没有关闭所有 sparse patch 方法，只把“病灶模糊本身就是证据干预”关闭了。

最大风险：blur 不是临床上真实的无病灶反事实，且只有 Huatuo、两个 finding。

## 实验 1：病灶删除—搬运（Sparse Local Evidence 最后一门）

### 在问什么

不再问“遮挡有没有反应”，而问一个更强的守恒问题：同一个小结节外观被删除后，Yes margin 是否下降；再把这块外观搬到同图另一侧，margin 是否恢复。

定义：

- `deletion_drop = margin(original) - margin(deletion)`；
- `relocation_recovery = margin(relocation) - margin(deletion)`。

只有两者同时为正，才说明分数跟随局部病灶内容，而不是跟随遮挡面积或全图扰动。

### 入口、规模与缓存

```bash
PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.corrected_sgta.run_vindr_lesion_transplant_v1 \
  --csv /workspace/vinbigdata/train.csv \
  --dicom-root /workspace/vinbigdata/train \
  --output-dir corrected_runs/daylong_idea_search/lesion_transplant_n128_v2 \
  --per-finding 128 \
  --seed 20260806
```

- CPU 预检已确认当前 v2 能选出 `128/128` 个独立 Nodule/Mass cases：86 个有 2 reader boxes，42 个有 3 reader boxes。
- 每例 3 次前向，共 384 次；本机同类 focal-erasure 384 次前向实测约 2 分 44 秒。
- 保守预计：模型加载加推理共 5–12 分钟。
- 可复用本地 DICOM、bbox、Huatuo scorer；旧 n=2 是 v1 insertion 语义，不能冒充 v2 结果。

### 冻结门槛

- `deletion_drop` 与 `relocation_recovery` 的 image-bootstrap 95% CI 下界均大于 0；
- 两个方向各自至少 60% 病例为正；
- 原图本来预测阳性的 admitted subset 也同向；
- 任一失败即关闭“局部病灶可安全搬运”的路线，不调阈值。

最大风险：水平反射组织仍可能产生不自然边界或解剖冲突。即使通过，也只授权“局部内容可被读取”，不等于能降低开放报告幻觉。

## 实验 2：同一观测变换 vs 真实第二观测

### 在问什么

仓库已经看到：对同一张图做 style 变换，决策翻转很少，平均融合还会变差。这可能不是模型太差，而是一个更基本的事实：**同一观测的可逆/轻微变换没有增加新的临床信息；第二次真实拍摄才可能增加信息。**

这是 active sensing / information acquisition 的最小致死实验，不把“不同图片”误叫成域中心。

### 已准备的 CPU 入口

已新增可复现 builder/analyzer：

```bash
PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.corrected_sgta.iuxray_observation_complementarity_v1 build \
  --input data/medheval/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/xray_closed_pairs.json \
  --image-root data/medheval/images/IU-Xray \
  --output-dir corrected_runs/daylong_idea_search/iuxray_paired_claims_v1 \
  --per-label 128 \
  --seed 20260812
```

该入口已 CPU 实测通过，生成 256 claims：128 Yes、128 No、256 个不同 studies；每个 qid 保持相同问题与答案，只将 `view0` 的 `/0.png` 对应到同 study 的 `/1.png`。

分别打分两个真实观测：

```bash
PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.corrected_sgta.run_claim_universe_scoring \
  --model huatuo \
  --questions corrected_runs/daylong_idea_search/iuxray_paired_claims_v1/view0.json \
  --image-root data/medheval/images/IU-Xray \
  --output-dir corrected_runs/daylong_idea_search/iuxray_huatuo_view0_v1 \
  --skip-null

PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.corrected_sgta.run_claim_universe_scoring \
  --model huatuo \
  --questions corrected_runs/daylong_idea_search/iuxray_paired_claims_v1/view1.json \
  --image-root data/medheval/images/IU-Xray \
  --output-dir corrected_runs/daylong_idea_search/iuxray_huatuo_view1_v1 \
  --skip-null
```

分析入口：

```bash
PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.corrected_sgta.iuxray_observation_complementarity_v1 analyze \
  --view0-raw corrected_runs/daylong_idea_search/iuxray_huatuo_view0_v1/raw.jsonl \
  --view1-raw corrected_runs/daylong_idea_search/iuxray_huatuo_view1_v1/raw.jsonl \
  --output corrected_runs/daylong_idea_search/iuxray_huatuo_complementarity_v1.json \
  --bootstrap-draws 5000 \
  --permutations 2000
```

### 对照与门槛

- 固定融合：`(margin_view0 + margin_view1)/2`，不在 test 调权重。
- wrong-study placebo：把第二张图的 margin 在 256 studies 间打乱；不增加 GPU。
- oracle：任一真实 view 答对的上限，只衡量 headroom，不当方法结果。
- 同观测对照：builder 同时生成 `style_view0.jsonl`，可用现有 `run_huatuo_style_phenomenon_confirm.py` 跑 gamma 0.9/1.1 与轻量 FedDG 变换；原 128 例 cache 中四视图均值已从 76.56% 降到 75.00%，oracle 仅 78.13%。

GO 必须同时满足：

- 真实两图平均融合比 view0 至少 `+2pp`，95% CI 下界大于 0；
- 比 wrong-study 融合至少 `+2pp`，permutation `p≤0.05`；
- 两图 oracle 比 view0 至少 `+5pp`。

Huatuo 通过后才用同一 manifest 跑 Hulu；Huatuo 失败则立即回到 baseline，不扩模型。

- 每模型 512 次单步 claim scoring。历史实测：Hulu 1,892 次约 4 分 14 秒；Huatuo 512 次 style scoring 约 50 秒。保守按每模型 5–10 分钟。
- 最大风险：真值来自 study-level 共享报告，并不是医生逐 view 标注。因此通过只能说明“第二次采集对 study-level 判断有互补信息”，不能声称某病灶在哪个 view 可见。

## 实验 3：Criterion-Shift Mirage（CPU，初步已确认）

### 在问什么

同一批自然语言回答，仅改变固定评判口径，方法优劣是否会翻转。如果会，所谓提升可能来自“更容易被某个 parser 接受”，而非更正确的临床判断。

### 可复现入口

已新增：`anchor/corrected_sgta/analyze_criterion_shift_mirage_v1.py`。

```bash
PYTHONPATH=. /opt/miniconda3/envs/huatuo/bin/python \
  -m anchor.corrected_sgta.analyze_criterion_shift_mirage_v1 \
  --input DoLa=corrected_runs/paper_baselines_v1/full_matrix_v1/derived_scores/llava/DoLa/cxr_vishal/evaluation_ce_v7.json \
  --input VCD=corrected_runs/paper_baselines_v1/full_matrix_v1/derived_scores/llava/VCD/cxr_vishal/evaluation_ce_v7.json \
  --input OPERA=corrected_runs/paper_baselines_v1/full_matrix_v1/derived_scores/llava/OPERA/cxr_vishal/evaluation_ce_v7.json \
  --input PAI=corrected_runs/paper_baselines_v1/full_matrix_v1/derived_scores/llava/PAI/cxr_vishal/evaluation_ce_v7.json \
  --input VISTA=corrected_runs/paper_baselines_v1/full_matrix_v1/derived_scores/llava/VISTA/cxr_vishal/evaluation_ce_v7.json \
  --output corrected_runs/daylong_idea_search/criterion_shift_llava_cxr_vishal_v1.json \
  --bootstrap-draws 5000
```

- `n=5,587` questions，`475` image clusters，5 个 LLaVA methods。
- 只读已有完整输出；预计 3–8 秒，零 GPU。

### 已观察到的结果

- 严格解析排名：`OPERA > VISTA > PAI > DoLa > VCD`。
- 官方 proxy 排名：`OPERA > VCD > PAI > DoLa > VISTA`。
- 10 对方法中，strict 与 official 有 5 对反转；strict 与 parseable-only 有 6 对反转。
- 最清楚的例子：VCD 相对 VISTA，在 strict 下是 `−4.58pp`，image-bootstrap 95% CI `[-5.62, -3.51]pp`；在 official proxy 下却是 `+4.98pp`，CI `[+3.88, +6.18]pp`。同一输出、同一问题，结论符号完全相反。

### 判定

**Criterion-shift 现象在该模型×数据集上已确认。** 后续新方法必须在预先冻结的 strict、official、balanced/parse 审计中报告，若方法结论随 criterion 反号，不得称为幻觉缓解。

最大风险：这个结果证明“评价依赖”，不告诉我们哪套规则具有临床正确性，也不是缓解算法。它最适合作为论文的测量发现或所有新方法的防伪门。

## 最短执行顺序（不破坏 baseline）

1. Criterion audit 已可在 CPU 完成，不动 GPU。
2. 等当前 baseline 完成一个可恢复 chunk 后，用同一 GPU lock 插入 **lesion transplant n=128**；预计 5–12 分钟。失败则永久停止 sparse-local 扩展。
3. 再插入 **IU Huatuo paired-view n=256**；预计 5–10 分钟。失败则不跑 Hulu，立即让 baseline 继续。
4. 只有 IU Huatuo 过全部门槛，才花约 5–10 分钟做 Hulu confirmation。
5. 整个探索 GPU slice 保守小于 35 分钟；任何进程都用独立输出目录，不修改 baseline manifest、answers、state 或 tmux 会话。

## 最可能得到的研究分叉

- **局部搬运 FAIL、真实第二观测 PASS**：最有价值。说明问题不是“找到一个内部 patch/head”，而是“单次观测的信息缺失”；应转向 active acquisition / value of information，而不是继续做同图解码扰动。
- **局部搬运 PASS、真实第二观测 PASS**：局部证据存在且新观测互补，可研究最小额外采集或区域请求，但仍需开放式 claim 验证。
- **两者都 FAIL**：内部操控与额外视图都没有可靠 headroom，停止方法包装，回到 baseline 与不可解码负结果。
- **局部搬运 PASS、真实第二观测 FAIL**：仅能做狭窄的 focal-finding 机制论文，不足以支持通用医学 VLM 幻觉缓解。


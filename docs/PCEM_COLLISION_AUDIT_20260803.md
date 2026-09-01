# PCEM Fatal Collision Audit（2026-08-03）

## 0. 结论先行

**裁决：机制新颖性仍有一个狭窄但真实的开放窗口；当前实验状态仍为 `ACCESS-BLOCKED / GPU NO-GO`。**

以下命题都已经被既有工作充分占据，不能构成论文贡献：

1. AP 投照会放大心影，PA/AP 应使用不同的心胸比阈值；
2. 网络能够从像素中高精度识别 AP/PA；
3. CXR 模型会利用 view position、portable equipment 等采集捷径；
4. view-specific threshold、显式 view token、domain adaptation 可以改善性能或校准；
5. VLM 的视觉信息可能随 decoder 深度丢失，并可用 steering 缓解幻觉。

PCEM 唯一可能留下的机制增量是：

> 在独立超声真值稳定的同患者 AP/PA episode 中，医学 VLM 的中间表征同时保留 projection state 与 apparent cardiac geometry，但二者形成临床判断所需的条件交互在 support-to-language transition 附近被削弱或不再被因果使用，导致模型把 projection-limited apparent enlargement 过度承诺为 definite intrinsic cardiomegaly。

这个命题比“view 可解码但最终答案没用好”严格得多。仅证明 view 和 morphology 分别可线性解码，**不能**证明 conditional interaction 曾经存在，更不能证明它后来被 erased。只有早层的条件交互对独立临床目标具有额外预测力，而后层该预测力显著下降，并且选择性恢复该交互能改变 certainty、但不改变 claim identity、覆盖率及清晰病例性能，才能使用 `erasure` 一词。

当前本地审计已经证明 MIMIC-CXR 有足够的 AP/PA 候选患者，但 MIMIC-IV-ECHO 结构化测量仍未取得，无法确认有效 echo join、样本数和 construct validity。因此现在启动 GPU 只会把一个临床构念未闭合的问题包装成昂贵的 representation probe。

---

## 1. 冻结后的研究问题

### 1.1 不是一般的 projection bias

研究对象不是：

```text
AP image -> more cardiomegaly predictions
```

而是一个条件证据律是否在生成阶段失效：

```text
independent intrinsic-heart truth T
        +
apparent silhouette geometry G
        +
projection/acquisition state V
        |
        v
clinically valid interpretation P(T | G, V, quality, episode)
        |
        v
language commitment C
```

必须区分三种临床输出：

1. `apparent silhouette enlarged`：影像表象陈述；
2. `projection-limited / intrinsic size uncertain`：证据受投照限制；
3. `definite intrinsic cardiac enlargement`：对真实心脏增大的确定承诺。

如果 AP 片上确实有放大的心影，模型说“cardiomediastinal silhouette appears enlarged”并不是幻觉。只有在缺乏独立支持时，把表象升级为第三种确定性 claim，才是本文所指的 overcommitment。

### 1.2 目标 estimand

行为层主模型应预注册为：

\[
C \sim T + G + V + G\!\times\!V + Q + A + P + (1\mid patient/episode)
\]

其中：

- \(C\)：三态语言承诺，而非仅有 cardiomegaly 关键词；
- \(T\)：独立 echo/CT truth；
- \(G\)：apparent CTR 或分割得到的 apparent geometry；
- \(V\)：AP/PA，并尽可能细分 portable、supine、source distance；
- \(Q\)：吸气程度、旋转、遮挡等图像质量；
- \(A\)：acuity/ICU/ED 等病情与场景；
- \(P\)：prior study、治疗、容量状态变化等 episode covariates。

关键量不是 projection 的主效应，而是加入 \(G\times V\) 后，对经临床复核的 overcommitment 是否带来 patient-disjoint 的增量预测能力。

---

## 2. 最近十篇最相邻工作及碰撞程度

| 工作 | 年份 / venue | 已经证明什么 | 与 PCEM 的核心重叠 | 官方代码 / 数据 | 尚未覆盖的机制 delta |
|---|---|---|---|---|---|
| [Bhave et al., CheXchoNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC11156488/) | 2024, *European Heart Journal* | 用 echo 衍生的 IVSd、LVIDd、LVPWd 训练 CXR 模型，预测左室结构异常，并进行外部验证和放射科医师比较 | 独立 echo truth + CXR 已经不是新意 | [官方代码](https://github.com/sbhave77/CheXchoNet)，[PhysioNet 数据](https://physionet.org/content/chexchonet/1.0.0/) | CheXchoNet 明确只保留 PA、排除 portable AP，不能检验 projection-conditioned evidence law 或 VLM commitment |
| [Dávila-García et al.](https://pubmed.ncbi.nlm.nih.gov/42390349/) | 2026, *Radiology: Cardiothoracic Imaging* | 在 AP CXR 与 24h 内 TTE 配对数据上预测 cardiac chamber enlargement，优于 CTR 和读片者 | AP 片并非“无真值信息”；AP-specific echo-grounded modeling 已被占据 | 未找到官方训练代码 | 仅 AP，无同患者 AP/PA 条件交互、无 VLM language commitment 或层级机制 |
| [Hosch et al.](https://pubmed.ncbi.nlm.nih.gov/32615636/) | 2021, *RöFo* | CNN 可从像素以 >.99 AUROC 区分 AP/PA，并在外部数据复现 | “模型编码 projection state”本身毫无新颖性 | 未找到官方代码 | 未研究 morphology × view 的临床组合及其进入语言的过程 |
| [Lotter et al.](https://www.nature.com/articles/s41467-024-52003-3) | 2024, *Nature Communications* | view、FOV、window、portable equipment 等 acquisition factors 会影响 CXR 模型；per-view threshold 可缓解偏差 | projection-specific calibration/threshold 已有强碰撞 | 官方页面提供 source data；未发现官方训练仓库 | 未研究 cardiomegaly 的独立 echo truth、VLM 层级 conditional binding 或 certainty-only causal repair |
| [Thiam et al.](https://pubmed.ncbi.nlm.nih.gov/36844424/) | 2023, *Frontiers in AI* | cardiomegaly 检测存在跨数据域偏移，可学习 domain-invariant representation | “cardiomegaly + domain shift correction”已被占据 | 未找到官方仓库 | 使用 PA-only 数据；未触及 AP/PA evidence law、echo truth 或生成承诺 |
| [Jabbour et al.](https://proceedings.mlr.press/v126/jabbour20a.html) | 2020, MLHC | CXR 模型会编码并利用 age/sex 等非病理属性捷径，并讨论迁移缓解 | “可解码元数据导致 shortcut”是成熟范式 | PMLR 页面未列官方代码 | 未区分属性 availability、临床条件组合和 support-to-language use |
| [Brown et al., ShorT](https://www.nature.com/articles/s41467-023-39902-7) | 2023, *Nature Communications* | 测试敏感属性是否被编码，并通过训练干预改变其对输出与公平性的影响 | “属性被编码 + 干预属性方向 + 观察输出”已存在 | [官方 shortcut-testing demo](https://github.com/google-research/google-research/tree/master/shortcut_testing)；临床模型细节未完整开源 | 未研究临床上应当使用、但必须条件使用的 projection 信息，也无语言承诺边界 |
| [Boland et al.](https://proceedings.mlr.press/v250/boland24a.html) | 2024, MIDL | 用 Prediction Depth 与 KL 定位医学影像模型学习 shortcut 的层级 | 单纯定位“projection shortcut 在哪一层出现”已不新 | PMLR 页面未列软件链接 | PCEM 若成立，projection 不是应删除的 shortcut，而是需要与 geometry 正确组合的 lawful context |
| [Lee et al., ViewXGen / UniXGen](https://proceedings.mlr.press/v248/lee24a.html) | 2024, CHIL | 用 view-specific token 和多视图结构进行 CXR 生成/报告建模 | 显式把 view condition 注入生成器已被直接覆盖 | [官方代码](https://github.com/ttumyche/UniXGen) | 未检验自然表征中 view 已存在却未与 claim evidence 正确绑定，也无 echo-grounded因果审计 |
| [VISTA](https://arxiv.org/abs/2502.03628) | 2025, ICML | LVLM 随深度出现视觉信息损失，早层激活和 visual steering 可缓解幻觉 | “早层有视觉信息、后层丢失、steering 修复”是最危险的通用机制碰撞 | [官方代码](https://github.com/LzVv123456/VISTA) | 未定义 projection-conditioned clinical law、独立 reader/echo truth、certainty-only repair 与 fixed-content controls |

另一个必须纳入强基线的机制工作是 [Jiang et al., Interpreting and Editing VLM Representations](https://openreview.net/forum?id=94kQgWXojH)（ICLR 2025；[官方代码](https://github.com/nickjiang2378/vl-interp)）。它已经说明 VLM 表征编辑可减少 hallucination；因此“找到一个方向并 steering”不能成为 PCEM 的贡献。贡献只能来自预先定义的临床条件律、独立真值以及对 conditional use 的选择性因果验证。

### 2.1 更早但直接封死弱版本的临床事实

- [Sahin et al.](https://pubmed.ncbi.nlm.nih.gov/30143921/) 已直接比较 AP 心胸比与 echo/CT，说明传统 PA 阈值不能机械用于 AP，较高 AP 阈值具有更合理的特异度。
- [Portable AP corrected CTR](https://pmc.ncbi.nlm.nih.gov/articles/PMC3207047/) 甚至已经提出基于成像几何的显式校正。
- [临床综述](https://pmc.ncbi.nlm.nih.gov/articles/PMC8139021/) 已系统说明 AP magnification、supine 和 poor inspiration 会模拟心影增大。

所以“发现 AP 造成 cardiomegaly bias，并设计 view correction”至多是已知临床规则的现代实现，不足以支持顶会机制论文。

### 2.2 2026 grounding update：弱版 PCEM 的行为 claim 已被进一步占据

本次复核加入三个在旧十篇表之外、但对当前 claim ceiling 更直接的
2026 邻居。机器可审计的 source/commit/license/entry hashes 冻结于
`results_reference/pcem_2026_collision_update_v1.json`。

1. [Lotfinia et al., *Vision-language models for chest radiography do not
   always need the image*](https://arxiv.org/abs/2606.17710v2) 已经对九个系统、
   2,575 个 CXR 判断运行 original、same-label different-patient swap、
   target mask 和 irrelevant mask 四条件，并联合使用 CGR/UAR/IS 区分
   `uses-image / ignores-image / unstable`。更直接地，所有 image-user 的
   causal grounding 都是 PA 高于 AP，三个系统经校正达到显著；
   cardiomegaly 在每个可评估 image-user 上都有 CI 下界大于零的局部
   grounding。其 [官方代码](https://github.com/mahshadlotfinia/causal)
   固定于本次审计 HEAD
   `6acd5639f06c7ac89c890f67a7e1eef335726d47`，MIT license，包含 view
   subgroup permutation/FDR 实现。
2. [Xiong et al., MedFocus](https://arxiv.org/abs/2605.20158) 已经构造经
   causal editing 筛选的胸片 attribution benchmark，并比较 11 种归因
   方法；其 concept/anatomy attribution 再用 targeted intervention 衡量
   对输出的因果作用。[官方代码](https://github.com/gzxiong/medfocus)
   固定于 `4f11fafcd6d53e8338a32c7b5a4c14f7f26db73d`，MIT license。故
   “新的 attention/heatmap/direction 更像临床证据”也已不能作为 PCEM
   机制贡献。
3. [HalluCXR](https://arxiv.org/abs/2605.20469) 已经把 response length
   作为胸片 VLM 幻觉 risk signal，并清楚展示 ensemble 降 fabrication
   会增加 omission。截至本次审计未找到作者官方代码，故 paper-native
   实现仍为 `not_admissible`；但 length/omission 交换必须作为解释控制，
   不能因无代码而忽略其论文结论。

这批结果不完全杀死强 PCEM，因为它们没有 independent echo truth、
没有 apparent geometry × projection 的临床条件交互、没有三态语言承诺，
也没有区分 `failure to compose` 与 layerwise erasure。但它们永久杀死以下
弱主张：

```text
image use is view-dependent
AP receives weaker visual grounding than PA
cardiomegaly answers can causally depend on image regions
masking/attribution alone identifies a new medical-VLM mechanism
```

因此 PCEM 在任何 representation capture 前新增 **G-image-use**：目标模型
与 finding 必须先通过同构的 original/swap/target-mask/irrelevant-mask
triad，且属于 stable `uses-image`。`ignores-image` 只能说明语言 prior，
`unstable` 只能说明扰动敏感；两者都不能并入 conditional-interaction
estimand。旧论文对 LLaVA-Med 的 always-positive 结果是强负先验，但不能
代替本仓库 exact checkpoint/template 的重新认证。

该门现已冻结为后端无关的
[Causal image-use common protocol v1](./CAUSAL_IMAGE_USE_COMMON_PROTOCOL_V1.md)，
可执行入口为 `anchor/medeval/evaluate_causal_image_use.py`。协议强制四条件
同一 reference/swap manifest、patient-cluster bootstrap、可信 truth/region
provenance、逐条件解析率和跨模型 closure；即使通过也固定
`representation_capture_authorized=false` 与 `gpu_authorized=false`，必须再由
独立 echo construct 和 geometry-by-view 行为门授权。当前没有真实模型结果，
不得把工程实现写成 G-image-use 已通过。

---

## 3. 已知结论与真正开放问题的分界

### 3.1 已经被占据的五层命题

| 层次 | 命题 | 状态 |
|---|---|---|
| 临床物理 | AP 改变 apparent heart size | 已知 |
| 表征可用性 | 模型可以识别 AP/PA | 已知，且非常容易 |
| 行为偏差 | AP/portable 与预测偏差相关 | 已知 |
| 工程修复 | view-specific threshold、metadata token、domain adaptation | 已知 |
| 通用 VLM 机制 | 深层视觉信息衰减，steering 可减少 hallucination | 已知 |

### 3.2 仍可能开放的第六层

真正的研究对象是 `conditional binding`：

\[
I_l = \text{incremental held-out information about }T\text{ or calibrated commitment from }G_l\times V_l
\]

应分别测量：

- \(A_l\)：projection state 的 decodability；
- \(G_l\)：apparent cardiac geometry 的 decodability；
- \(I_l\)：控制 \(A_l,G_l\) 主效应和混杂后，二者交互的额外临床预测力；
- \(C_l\)：该交互对最终 claim commitment 的因果使用程度。

关键逻辑：

```text
A_l high and G_l high
        does not imply
I_l exists
        does not imply
the model causally uses I_l
```

因此存在两种必须区分的失败：

1. **failure to compose**：任何层都没有临床有效的 \(G\times V\) 表征；
2. **conditional erasure**：中间层已经形成有效交互，但在 support-to-language transition 明显衰减或不再被使用。

只有第二种能支持 PCEM 的 `erasure` 主张。第一种仍可能是可信负结果，但不能包装成后层抹除。

---

## 4. 数据 substrate 与当前访问结论

现有详细审计见 [PCEM_CHEXCHONET_SUBSTRATE_GATE_20260803.md](./PCEM_CHEXCHONET_SUBSTRATE_GATE_20260803.md)。关键事实如下：

### 4.1 CheXchoNet 不适合作为主 substrate

- 71,589 张 CXR、24,689 名患者；
- 有独立 echo 测量，适合结构异常预测；
- 但官方 cohort 明确只使用 PA，并排除 portable AP；
- Columbia patient ID 无法与 MIMIC-CXR patient ID 直接连接；
- echo 允许最长约 ±12 个月，并取窗口内最大测量，不适合建立短时 AP/PA 稳定真值。

结论：CheXchoNet 可作为 echo-grounded PA 外部边界实验，不能验证 PCEM 的 projection interaction。

### 4.2 MIMIC-CXR 提供 projection 对比，但 echo join 尚未闭合

本地 metadata 审计：

- 377,110 images / 65,379 patients；
- AP 147,173，PA 96,161；
- 15,185 名患者同时拥有 AP 与 PA；
- 最近 AP→PA 在 6h 内：1,491 links / 1,158 patients；
- 24h 内：5,246 / 3,280；
- 72h 内：13,085 / 5,982；
- same-study 最近配对：345。

这些只是候选 image links，不代表 echo-stable clinical episodes。MIMIC-IV-ECHO 官方数据包含大量结构化 echo studies，并可通过 `subject_id` 与时间连接，但当前本地请求仍返回 HTTP 403，尚无可审计的测量文件、hash、schema 和实际 join 数量。

因此当前状态必须写作：

```text
scientific delta: conditionally open
data execution: ACCESS-BLOCKED
GPU experiments: NO-GO
```

不能根据 CXR metadata 候选数外推“数据已经足够”。

---

## 5. 最危险的真值与因果混杂

### 5.1 Echo 不等于 radiographic cardiomegaly

LVIDd、室壁厚度、左室质量或 chamber enlargement 与 total cardiomediastinal silhouette 不是同一构念。右心、心包积液、体型、胸廓、旋转和呼吸相位都可能改变 CXR 表象。

因此真值合同必须是分层的：

1. CXR apparent silhouette / CTR：由图像本身定义；
2. projection limitation：由 acquisition + 质量定义；
3. intrinsic chamber/heart enlargement：由 echo/CT + 临床复核约束。

Echo 只能约束第三层，不能把真实的第一层表象判成“幻觉”。

### 5.2 AP/PA 不是随机干预

AP 与以下变量高度耦合：

- portable equipment、supine、ICU/ED；
- 更差吸气、旋转和遮挡；
- 更严重病情、容量负荷和通气状态；
- 利尿、输液、机械通气和手术前后；
- source-to-image distance 与 detector geometry。

即便同患者 24–72h 配对，也可能跨越治疗和真实生理变化。必须优先：

1. same-study 或极短窗口；
2. episode-level 临床审计；
3. patient/episode grouped split；
4. 对 treatment、ventilation、ICU status、inspiration、rotation 等显式控制；
5. 对不同时间窗口做 sensitivity analysis。

### 5.3 Echo selection bias

接受 echo 的患者本身具有更高的心脏病先验，MIMIC 又偏 ICU/ED 场景。即使做 inverse propensity weighting，也无法消除所有未观测选择。论文必须把结论限定为有临床指征并获得 echo 的医院人群，不能泛化到普通筛查 CXR。

---

## 6. 最小但有判别力的机制设计

### 6.1 Stage A：先证明行为现象不是阈值问题

在拟合下述行为模型前，先运行 G-image-use：

- original；
- same-label、different-patient image swap；
- target cardiac-region mask；
- same-size irrelevant-region mask。

必须联合报告 causal grounding rate、unrelated-image answer rate 和
irrelevant-mask stability。只保留 stable image-user cells；不能用总体
accuracy、单一 flip rate 或正确原图答案冒充视觉依赖。这个 gate 是对
Lotfinia et al. 官方 causal triad 的任务内适配，不作为本文新方法。

在 patient-disjoint test 上比较：

1. polarity-only / truth-only；
2. `truth + apparent geometry + projection` 主效应；
3. 加入 `geometry × projection`；
4. 再加入 image quality、acuity、portable/supine、episode covariates。

只有模型 3/4 对**经临床复核的三态 overcommitment**产生稳定增量，PCEM 才进入 representation audit。

建议最低行为 gate：

- held-out ΔAUROC 或相应 ordinal likelihood gain 等价于 ΔAUROC ≥ 0.03；
- patient-cluster bootstrap 95% CI 排除 0；
- 至少两个开放医学 VLM；
- matched claim count、answer length、negative rate、refusal rate；
- geometry-insensitive findings 不出现同样模式。

### 6.2 Stage B：区分 availability 与 conditional binding

在 vision encoder、projector 前后、decoder 1/4、1/2、3/4、final 层记录：

- view probe \(A_l\)；
- apparent geometry probe \(G_l\)；
- truth probe \(T_l\)；
- residualized conditional-interaction probe \(I_l\)。

所有 probe 只在 dev 拟合，test 只执行一次；按 patient cluster bootstrap。交互 probe 必须与以下基线比较：

- 相同参数量的 main-effect MLP；
- shuffled-view interaction；
- same-support / same-CTR image swap；
- norm-matched random direction；
- projection label 从 DICOM metadata 与 pixel classifier 两种来源；
- prompt paraphrase 与生成温度。

最关键的 preregistered test：

\[
\max_{l<final}\mathrm{AUROC}(I_l)-\mathrm{AUROC}(I_{final}) \ge 0.05
\]

并且 image-cluster bootstrap 95% CI 排除 0。若 \(A_l\) 与 \(G_l\) 高、但任何层的 \(I_l\) 都不成立，结论只能是 failure to compose。

### 6.3 Stage C：因果干预必须只修复条件使用

候选干预不是注入一个 AP token，而是：

1. 将 view component 对 polarity / claim-identity subspace 正交化；
2. 只 patch 或 restore 与 valid \(G\times V\) interaction 对齐的残差分量；
3. 恢复原 activation norm；
4. 固定 claim set、positive K、answer length 与 ontology coverage；
5. 只允许 certainty 或 projection qualification 改变。

成功干预应把：

```text
definite intrinsic cardiomegaly
```

修正为：

```text
apparent enlargement; projection limits assessment of intrinsic size
```

而不是删除 cardiomegaly claim、统一输出 negative、拒答或缩短报告。

---

## 7. 必须击败的最强基线

任何 PCEM 方法必须同时超过以下组，否则机制解释不成立：

### 7.1 简单临床/校准基线

- PA/AP 独立 threshold；
- DICOM view metadata-conditioned logistic/ordinal calibration；
- apparent CTR + projection 的显式规则；
- temperature scaling、isotonic、Platt scaling；
- prompt 中直接告知 `portable AP`；
- output post-hoc hedge rule。

若这些基线达到同等收益，合理解释是普通 calibration，不是 latent conditional erasure。

### 7.2 表征与 domain 基线

- view token / metadata fusion；
- domain-adversarial or invariant representation；
- per-view classifier / mixture-of-experts；
- VISTA-style visual information steering；
- ICLR 2025 VLM representation editing；
- random/norm-matched steering；
- final-layer linear steering。

另外，任何 attribution/patch 解释都必须比较 MedFocus/released attribution
baselines；任何行为 grounding 解释都必须报告 original/swap/target-mask/
irrelevant-mask 的 CGR/UAR/IS triad。二者是 collision controls，不是可选
附录。

### 7.3 内容保持基线

- 固定 claim 数 K；
- matched answer length；
- matched negative rate；
- matched hedge rate；
- matched refusal rate；
- matched clear-case accuracy；
- same intervention applied to geometry-insensitive findings。

主方法必须证明收益来自对临床条件律的恢复，而不是更少说、更模糊地说或更依赖 metadata。

---

## 8. 硬 GO 条件

只有以下四组条件全部满足，PCEM 才能进入论文主线。

### 8.1 数据与真值 GO

- 已实际取得 MIMIC-IV-ECHO 文件，记录版本、hash、schema；
- 至少 300 个 unique-patient、短窗口 AP/PA + TTE 有效 episodes；
- intrinsic enlargement positive / negative 各 ≥100，borderline ≥100；
- 至少 60 个 episode 完成治疗、通气、容量状态与时间稳定性审计；
- 至少 100 个样本做 blinded 三态临床复核；
- inter-reader agreement 达到预注册门槛，或把 reader distribution 作为软真值；
- train/dev/test 按 patient 和 episode 双重隔离。

### 8.2 行为 GO

- 每个进入主估计的模型/finding cell 先被 causal triad 判为 stable
  image-user；`ignores-image` 与 `unstable` 分开报告并排除；
- 至少两个医学 VLM 的 pixel-only view AUROC ≥0.90；
- apparent geometry score 在 holdout 有可靠校准；
- `geometry × projection` 在主效应和混杂之上带来 ≥0.03 held-out 增量，cluster CI 排除 0；
- effect 在 matched length / K / prompt 后仍在；
- geometry-insensitive controls 无同类效应。

### 8.3 机制 GO

- 早/中层的临床有效 conditional interaction 比 final layer 高 ≥0.05 AUROC；
- 差值的 cluster-bootstrap 95% CI 排除 0；
- 不是 view 或 morphology 单独可解码造成的假象；
- selective patch/scrub 对目标 overcommitment 有方向一致的因果效应；
- random direction、norm、temperature、prompt 和 image-swap controls 均不能解释。

### 8.4 方法 GO

- overcommitment 相对下降 ≥20%；
- clear-case accuracy 下降 ≤1pp；
- reader/echo distribution Brier 相对改善 ≥5%；
- claim identity、positive K、coverage、长度和拒答率基本不变；
- 优于 view-specific threshold、metadata prompt、ordinary calibration 与 generic visual steering；
- 至少两个模型，并在 CE 与原生 OE/report 中复现。

---

## 9. 硬 NO-GO 条件

任一项成立即停止 PCEM 主线，不继续修阈值：

1. 无法取得 echo join，或有效 episode / 各 truth bin 数量不足；
2. echo construct 与 clinically adjudicated intrinsic enlargement 不一致；
3. AP/PA effect 在控制 acuity、portable、supine、inspiration、rotation、treatment 后消失；
4. 模型连 projection state 都不能稳健识别；
5. 行为增益可被 per-view threshold、metadata prompt 或普通 calibration 完全解释；
6. view 与 geometry 虽分别可解码，但任何早层都没有额外 conditional interaction；
7. conditional interaction 在层间不下降，或下降 <0.05 / CI 跨 0；
8. patch 只通过删除 claim、增加 hedge、缩短回答或提高拒答获益；
9. patch 同样影响非 geometry-sensitive findings，说明是通用生成扰动；
10. clear cases 损失 >1pp，或 claim identity/coverage 改变；
11. 仅一个模型、单一 prompt 或结构化 Yes/No 成立；
12. 最终“方法”只是把 DICOM AP/PA 写进 prompt 或应用手工阈值。
13. 目标模型/finding 在 original/swap/target-mask/irrelevant-mask triad 上是
    `ignores-image` 或 `unstable`，却仍被用于声称 projection-conditioned
    visual binding。

若第 6 项成立，允许形成一个更诚实的负结果：医学 VLM 能识别 acquisition context，却没有把它组合成临床有效的证据律。此时应称 `failure to compose`，不能称 `erasure`。

---

## 10. 当前研究决策与下一步

### 当前裁决

| 维度 | 状态 | 原因 |
|---|---|---|
| 临床问题重要性 | 高 | AP/portable 在真实部署常见，过度诊断具有直接临床意义 |
| 弱版本新颖性 | 已碰撞 | view correction、AP threshold、domain shift 均已有成熟证据 |
| 强机制版本新颖性 | 条件开放 | 尚未发现工作同时证明 independent truth + conditional interaction erasure + certainty-only causal repair |
| 数据可执行性 | ACCESS-BLOCKED | MIMIC-IV-ECHO 未实际取得，join 与构念样本数未知 |
| GPU 决策 | NO-GO | 真值和 episode 尚未闭合，先跑 probe 没有可解释性 |

### 立即执行顺序

1. 取得并审计 MIMIC-IV-ECHO，冻结 measurement schema 和 truth contract；
2. 只做 CPU temporal join、episode stability 与 sample-size gate；
3. 完成小规模 clinician 三态 construct audit；
4. 先跑廉价行为 interaction test，击败 per-view threshold 和 metadata prompt；
5. 只有行为 gate 通过，才启动多层 activation capture；
6. 只有早层 conditional interaction 确实存在且后层衰减，才开发 causal projection。

### 最终定位

PCEM 不能作为“新的 AP/PA 校正方法”投稿。它只有在下述一句话被完整证实时才具备 ICLR 级机制潜力：

> The model knows the acquisition geometry and represents apparent morphology, but loses the clinically lawful interaction between them exactly when visual support is converted into linguistic commitment; restoring that interaction changes certainty rather than content.

在 echo 访问与 construct gate 通过前，保留它为**高价值、访问受阻的竞争方向**，但不消耗 GPU，也不让它挤占已经通过数据门槛的研究分支。

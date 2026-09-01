# VinDr-CXR 作为 CECD 多-claim listing substrate：严格边界与最小协议

**Freeze:** 2026-08-03
**Verdict:** **CONDITIONAL GO for ontology-constrained, open-cardinality
listing; STRICT NO-GO for unrestricted native OE truth.**
**Execution:** outcome-blind CPU audit only; no model outputs, model scores, or
GPU were read or used.

## 1. 结论先行

VinDr-CXR 确实能解决 MIMIC-only 路线最致命的 atomic-truth 问题，但只在
一个窄而诚实的任务定义下：给模型明确的 finding ontology，让它输出任意
数量的 ontology members。该任务是一个真正的多-claim content-selection
问题，因为输出 cardinality 不固定且一个回答可同时包含多个 finding；但它
不是 unrestricted free-form OE，也不是 report generation。

推荐采用两层 substrate，而不是在 8 类和 14 类间虚假二选一：

| Track | 真实用途 | 当前规模 | 强项 | Claim ceiling |
|---|---|---:|---|---|
| **8-finding balanced mechanism track（主轨）** | 首轮 Huatuo/Hulu CECD 行为机制和 fixed-`K` | 2,341 images × 8 = 18,728 claim rows；609 images 含至少两个 3/3 required findings | 每个 finding 的 0/3--3/3 都有充分 dev/confirmation cells；现有 manifest 已 image-disjoint | 八类 closed-ontology listing 的 aggregate 与 per-finding 结论 |
| **14-finding natural breadth track（广度轨）** | 证明结论不依赖精挑 8 类，并测试更自然的 multi-claim set composition | 固定 panel 5,501 images 中 1,360 个真实 multi-claim cases；冻结 sample 为 420 images × 14 = 5,880 claims，其中140个 selected multi-claim cases | 与 VinBigData detection ontology 一致；有明确正常/单-claim/多-claim分层 | 14类 aggregate 结论；稀有 finding 只作描述性结果 |

主轨不是“8个 Yes/No 问题拼接”。每张影像只接受一个 listing prompt，模型
自由选择 0--8 个 finding；完整输出随后确定性展开为固定 claim membership
vector。CE 只在必要的辅助 verifier 分支出现。

## 2. 数据事实与不能越过的语义边界

官方 VinDr 宽表提供每张训练影像三位独立 radiologist 的完整 28-label
向量；15,000 张训练影像中每张恰有三位 reader。官方说明同时确认训练集每
例由三位 radiologist 独立标注，并区分 22 个 local labels、5 个 global
diagnoses 与 `No finding`。见 [PhysioNet VinDr-CXR v1.0.0](https://physionet.org/content/vindr-cxr/1.0.0/)
及 [Scientific Data dataset paper](https://doi.org/10.1038/s41597-022-01498-w)。

本地审计得到：

- exact `R8/R9/R10` panel 覆盖 5,501 张影像；
- 14 类中有 1,360 张影像至少包含两个 3/3 unanimous findings；
- 4,191/5,501 张影像在 14 类之外仍有至少一个 reader-positive 官方标签；
- 3,620/5,501 张影像在 14 类之外有至少一个 3/3 unanimous 官方标签；
- 因而 14 类绝不能被称为完整 radiographic truth，只能在 prompt 显式限定
  ontology 时使用 closed-world scoring；
- DICOM `PatientID`、`StudyInstanceUID` 与 `AccessionNumber` 在冻结样本中均
  不可用；`image_id` 是匿名 SOP identity，不是 patient identity。只能证明
  image-disjoint，不能证明 patient-disjoint。

`No finding` 也不是第15个与14种疾病并列的独立 claim。对限定 ontology 的
任务，空输出被序列化为 `None of the listed findings`；这不会把“目标14类
为空”偷换为“整张影像完全正常”。

14类冻结样本采用在任何采样前完成的 image-SHA256 `20/20/60`
pilot/dev/confirmation split，然后在每个 split 内对以下三类各取
`20/40/80` 张：

1. 三位 reader 均为全局 `No finding`；
2. 目标 ontology 中恰好一个 3/3 required finding；
3. 目标 ontology 中至少两个 3/3 required findings。

分层采样改变 prevalence，因此 population metrics 必须使用 manifest 中的
inverse sampling weights；不得把平衡样本的 raw precision 当自然 prevalence。

## 3. CECD 在 listing 中的两个不同读数

### 3.1 Primary：直接列表生成后的 fixed-membership derivative

每个 `render × prompt` cell 都执行一次 greedy list generation。严格 parser
只接受用逗号分隔的 exact ontology labels 或 exact empty-set token；解释、
重复 label、ontology 外内容和混合 empty token 都被保留为
`format_violation`，不得静默丢弃。

解析后，对每个冻结 finding (c) 得到同一身份的 membership：

\[
g_c(r,p)=\mathbb 1[c\in L(r,p)].
\]

因此虽然每个 cell 重新生成完整列表，claim identity 并未漂移；漂移的正是
OE 要研究的 content selection。对完整 `5 × 3` science orbit 计算：

\[
I_c(r,p)=g_c(r,p)-\bar g_c(r,\cdot)-\bar g_c(\cdot,p)
+\bar g_c(\cdot,\cdot).
\]

这直接回答：两个各自 admitted-equivalent nuisance 联合后，是否额外加入
0/3 fabricated claim、把 1/3--2/3 disagreement 当确定异常，或删去 3/3
required claim。它是本 track 的正式 OE behavioral estimand。

完整 orbit 要求 15 个 science cells、3 个 identity-render controls 和1个
exact-duplicate-prompt control，共19 cells。任一 cell 不可解析或缺失时，
该 orbit 不能偷偷进入 complete-case mechanism analysis；同时必须报告全样本
format-failure risk，防止通过筛选“听话回答”制造结果。

### 3.2 Secondary：atomic teacher forcing 不是同一个 OE estimand

在 listing prompt 下逐个 teacher-force 单一 finding 名称，会被输出顺序与
“这个 finding 是否应该排第一”混淆；teacher-force 完整 gold list 又把
ordering、标点、其他 claims 与目标 finding 耦合。因此这两种做法都不能
替代上面的 membership derivative。

若另加 `Should the list include <finding>? Yes/No/Maybe` verifier，则可获得
固定原子 claim 的 teacher-forced support score，但该分支本质上是 CE
verification over an OE draft。它必须：

- 单独定义并通过 prompt admission；
- 只作为 reranking/verifier 辅助读数；
- 不被写成“native OE teacher forcing”；
- 与 direct-list membership、full-orbit membership mean 和不使用 verifier
  的 controls 同时比较。

因此近期最省且语义最干净的路线是先跑 direct generation membership；只有
direct behavioral interaction 通过后，才值得为 causal localization 增加
atomic teacher forcing。

## 4. Fixed-`K`：不以少说换低 hallucination

对每个 image/model，以 canonical cell 的合法 ontology list 大小定义
(K=|L_{00}|)。所有 content correction 均必须输出恰好 (K) 个 ontology
claims；`K=0` 是显式 no-op，不能借机补 claim。canonical output 的
out-of-ontology 内容、拒答或 format violation 必须原样留在审计表中，不能
删除后声称结构守恒。

直接生成 surface 提供以下同预算候选分数：

- render marginal；
- prompt marginal；
- full-orbit membership mean；
- canonical-cell additive projection
  \(s_c^{add}=\bar g_c(r_0,\cdot)+\bar g_c(\cdot,p_0)-\bar g_c(\cdot,\cdot)\)，
  即从 canonical membership 去掉 centered interaction residual；
- deterministic random/tie-matched controls。

按每个分数选 top-`K`，tie-breaker 在看模型结果前冻结为 ontology ID 顺序并
同时运行随机 tie control。additive projection 是 classical factorial
projection 和 causal probe，不是独立算法 novelty；它只有在击败 full-orbit
mean 与两个 marginal 后才支持 CECD-specific content correction。

每张影像的 reader truth 定义为：

- 3/3：`supported` 且 listing 中 `required`；
- 1/3 或 2/3：`undetermined` 且 `optional`；
- 0/3：`refuted` 且 `out_of_scope`。

主指标必须同时报告：3/3 required recall、0/3 fabricated inclusion、
1/3--2/3 definite inclusion、3/3-support precision、reader-distribution Brier、
fixed-`K` identity、回答长度、拒答、format violation、以及 weighted 和
unweighted 版本。fixed-`K` 只保证 claim 数不变，不保证 claim identity
不变；所以这是 content exchange，不应与只改变 certainty 的 intervention
混为一谈。

## 5. Admission 与执行顺序

当前 pack **没有授权模型或 GPU**。原 binary CE 的 prompt admission 不会
自动转移到多-label listing speech act。即使 render 实现代码可以复用，也需
新 pack 在至少60张分层 multi-claim/normal影像上确认：

1. 五种 science render 不改变14类任一 claim 的 reader-grounded support；
2. 三种 prompt 保持同一 ontology、inclusion obligation、certainty demand、
   answer space 和 exact-output grammar；
3. 两位独立 clinical reviewers，加独立 language/template review 与冻结的
   adjudication；
4. admission 在任何 Huatuo/Hulu listing output 生成前冻结。

正式执行顺序仍受总 CECD gate 控制：

1. Huatuo + Hulu binary CE Stage-1 和 closest-work envelope 先通过；
2. 新 listing equivalence admission 通过；
3. 先运行8-finding balanced mechanism track 的 pilot/dev；
4. 只有两模型 direct membership interaction 均通过，才打开 confirmation；
5. 再运行14-finding breadth track 的 aggregate confirmation；
6. direct fixed-`K` 成功后才考虑 atomic verifier 与 hidden intervention。

## 6. 淘汰标准

任一项发生即停止将此轨道写成 OE mitigation：

- 任一 listing prompt/render admission 失败；
- Huatuo 或 Hulu 的 strict parseable complete-orbit rate不足，或只在筛掉
  invalid outputs 后成立；
- centered interaction 在 clean membership、两个 marginals、full-orbit mean、
  finding、reader support 和回答长度后没有独立信息；
- additive projection 不优于 full-orbit/marginal controls；
- fixed-`K` 下 fabricated/overcommitted inclusion 相对下降不足20%，bootstrap
  95% CI触零，或 required recall/coverage下降；
- 只在一个模型成立；
- 14类 aggregate 结果被少数高频 findings 完全驱动。

即使所有门通过，论文主张仍是 **reader-grounded closed-ontology clinical
claim listing**。要写 unrestricted native OE，必须对 ontology 外、anatomy、
attributes、differential diagnosis、history 与 unobservable claims 增加独立
医生 truth；VinDr 14/8-vector 不能替代这一步。

## 7. 可复核 artifacts

### Existing balanced 8-finding core

- `manifests_v2/summary_v2.json`: SHA-256
  `d3fe0c7b49b89f43feb73a1183187761a1181e9753e28a53d934f011e5412971`
- `manifests_v2/oe_listing_reference_v2.jsonl`: SHA-256
  `06a9d065f668d7d658a43522da4a891625577c30b66844be52d9133b57ef62a0`

### New 14-finding breadth pack

- `corrected_runs/vindr_v2/cecd_ontology_listing_substrate_v1/experiment_manifest.json`:
  SHA-256 `79e7469be61ef17ad9ac7764652e077434cad39fc9a806d7118ce659ec97be06`
- `reference_images.jsonl`: SHA-256
  `0c229f63b2c3427bdc49eebb5072abacf650471bc1ba98b1b3552e603f950213`
- `structural_validation.json`: SHA-256
  `3d8a72937fa1998aa296389b7ac420fe2a751f3cd52ad02416dbfcc11b95811a`
- `two_track_audit.json`: SHA-256
  `2525cdb6def9cac4990b6d17dfa4cddda4dd7b04c0c771bcab0e631cd936d745`

Builder、validator 和 parser tests：

- `anchor/corrected_sgta/prepare_vindr_cecd_ontology_listing_v1.py`
- `anchor/corrected_sgta/validate_vindr_cecd_ontology_listing_v1.py`
- `anchor/corrected_sgta/audit_vindr_cecd_ontology_listing_tracks_v1.py`
- `tests/test_vindr_cecd_ontology_listing_v1.py`

验证命令：

```bash
cd /home/dbw/ANCHOR

PYTHONPATH=.:anchor .venv-full/bin/pytest -q \
  tests/test_vindr_cecd_ontology_listing_v1.py

PYTHONPATH=.:anchor .venv-full/bin/python -m \
  corrected_sgta.validate_vindr_cecd_ontology_listing_v1 \
  --manifest corrected_runs/vindr_v2/cecd_ontology_listing_substrate_v1/experiment_manifest.json \
  --output /new/write_once/structural_validation.json
```

Current focused test result: `7 passed`; combined VinDr-v2/OE-transfer regression:
`16 passed`. Structural verdict:
`structurally_valid_conditional_go_closed_ontology_only`.

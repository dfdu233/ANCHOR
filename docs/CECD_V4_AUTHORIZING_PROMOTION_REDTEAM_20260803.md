# CECD v4 prototype → authorizing promotion：outcome-blind red-team

**日期：** 2026-08-03  
**范围：** 只审阅 `docs/CECD_PRODUCT_ATTRIBUTABLE_RISK_V4_DRAFT.md`、
`anchor/corrected_sgta/analyze_cecd_product_attributable_risk_v4_prototype.py`、
对应合成测试及既有 outcome-blind 设计审计。未读取 human return、真实模型结果、
sealed confirmation output 或 GPU 状态；未修改任何分析代码。尝试运行该单测文件时，
当前 shell 无 `pytest` 可执行文件，因此本报告的测试判断来自逐行 source/test audit，
不把“测试已通过”作为证据。

## 结论先行

**当前 verdict：CONDITIONAL NO-GO FOR AUTHORIZING PROMOTION。**

v4 prototype 已经保留了一个可信的统计核心：

- reader target `q=votes/3` 与模型分数独立；
- actual 与 additive counterfactual 使用同一 dev-only calibrator；
- interaction 在 Haar centered subspace 中保持 row/column sums 与 singular spectrum；
- primary PAEL 先按 orbit 平均 cell，再按 finding × vote macro；
- bootstrap 代码确实创建了一套跨模型共享的 cluster multiplier matrix，并在每个 draw
  重新计算 macro numerator 与 `B0` denominator；
- artifact seal、source hash、dev/confirmation overlap 与 apply-only 方向是正确的。

但 prototype 目前只能是 **non-authorizing diagnostic**，不能靠把
`authorized=False` 改成一个布尔门来升级。最致命的缺口是：

1. “16 strata”只是输出中的布尔描述，不是输入合同；单模型、单 finding、3×3 grid
   也能完整跑完；
2. draft 要求的每 orbit 至少 4,096 个 antithetic Haar draws、独立 MC-SE 审计和冻结
   doubling rule 均未实现；当前最低只要求 19，CLI 默认 499；
3. shared bootstrap 的 multiplier 是共享的，但全局 image→patient 映射、两模型 exact
   orbit pairing 与固定 finding closure 没有核验；
4. isotonic 的 `increasing=True` 不能证明原始 score→reader support 是单调的，且所有
   transformed/additive/Haar scores 超出 canonical-dev support 后都会静默 clip；
5. `B0>0` 不是 ratio 稳定性条件，任意接近零的 denominator 仍可授权一个巨大的 `R`；
6. LORO、identity/duplicate noise gate、per-finding direction、clear-case per-model net harm、
   human product control、MetaRA-style 与 semantic-boundary controls 均未进入 authorizing
   输入或计算图。

因此最小正确动作是：**保留 prototype 和现有结果格式作为 diagnostic；另建严格版本的
authorizing analyzer/config，不原地放宽或解释现有布尔字段。**

## P0：不修就不得授权

### P0-1 — exact 16-stratum closure 必须 fail closed

**发现。** `fit_dev_calibration` 动态遍历 payload 中“实际出现”的 models/findings，只要求
每个已出现 finding 有四个 vote bins（source `:271-282`）。`_macro_point` 在未传
`required_strata` 时把 present strata 当 target（`:528-545`）；
`complete_expected_16_strata` 只记录 `len(strata)==16`，不影响流程（`:607-614`）。
测试 fixture 甚至只有一个模型、一个 finding 和 3×3 grid，却被当作正常输入
（test `:17-18, :38-114, :165-198`）。

这会让 whole-orbit exclusion、缺模型、缺 finding 或错误 grid 静默改变 estimand。
“平均仍可计算”不等于原先冻结的四 finding × 四 vote-bin target 仍存在。

**authorizing contract 必须 exact-set 检查：**

- models 恰为 `{Huatuo, Hulu}`；
- findings 恰为冻结的四类；每模型恰有 16 个 finding × vote strata；
- science grid 恰为冻结 5 renders × 3 prompts，control IDs 也与 admission artifact
  exact-match；
- dev 每 stratum 恰有 20、confirmation 每 stratum 恰有 60 个完整 orbit，或使用事前
  写明的最低合格 quota；任何 invalid-cell 导致的 whole-orbit exclusion 后重新计数，
  不足即失败；
- 两模型的 `(image_id, finding, reader_votes)` orbit key 集完全相同；
- dev 与 confirmation 的 exact model/finding/grid/reader-panel contract 由一个 hash-bound
  config 提供，而不是从 payload 推断。

不存在“缺一个 stratum 仍按 15 个 macro”的可接受降级；这正是必须 fail closed 的地方。

### P0-2 — Haar 4,096 + antithetic + MC-SE rule 目前均未兑现

**发现。** draft `:126-128` 要求至少 4,096 antithetic draws/orbit，数值 MC-SE 低于
image-cluster sampling SE 的 10%，否则按冻结规则 doubling。实现只要求
`null_draws>=19`（source `:258-265, :321-326`），CLI 默认 499（`:846-855`）；循环对每
draw 独立采样（`:697-729`），没有 antithetic pair、独立-stream audit、MC-SE、doubling
或未收敛 fail-closed。

**最小修复：**

1. authorizing config 固定 `haar_total_draws_min=4096` 且必须为偶数；一个 base Haar
   orientation 与其 `-J'` 形成显式 antithetic pair，4096 指总 rotations 而非 4096 pairs；
2. 使用 keyed counter seed
   `(master_seed, payload_hash, model, image_id, finding, base_draw, left/right)`，避免一个
   orbit 的增删改变后续所有 orbit 的有限 MC realization；
3. 用至少两个独立 keyed streams 或预冻结 batch-means 估计 **最终 16-stratum model-level
   PAEL integral** 的 MC-SE；不能把 Haar draws 当 clinical samples；
4. 对两模型及四 finding 中最差者要求
   `SE_MC <= 0.10 * SE_cluster`；若不满足，按 4096→8192→16384 的冻结 doubling rule
   继续，只增加数值积分，不改变数据、threshold 或 seed family；达到预冻结 cap 仍不满足
   则 fail closed；
5. cluster bootstrap 条件于已收敛 Haar expectation；另报告独立 stream 差异，不能让
   cluster CI 隐藏 MC noise。

当前 `spectral_haar_interaction` 的几何实现本身是合理的（source `:341-368`）；问题是
积分合同，而不是旋转公式。

### P0-3 — “shared whole-image bootstrap”需要先闭合 cluster identity

**保留项。** `_shared_cluster_multiplier_plan` 生成一套 global weights，两个模型重复使用
（source `:618-659, :731-761`）；`_macro_cluster_bootstrap` 在每 draw 重新计算各 stratum
mean、`B0` 与 ratio（`:548-615`）。这一部分应保留。

**致命缺口。** `_orbit_cluster_map` 只在单个 `(model,image,finding)` orbit 内检查
patient ID 一致；同一 image 跨 finding/模型可拥有不同 patient ID，或一处有 patient ID、
另一处回退 image ID，而不报错（`:79-99`）。此外代码不要求两模型具有同一 orbit 集。
共享 multiplier 在这些情况下只是“同一随机矩阵”，不是同一临床 cluster 的配对重采样。

**最小修复：**

- cluster mode 必须全局选择：只有 patient mapping 完整且一对多映射一致时才全体使用
  `patient:<id>`；否则全体使用 `image:<id>`，不得按 orbit 混用 fallback；
- 校验同一 image 在所有 finding、cells、models 上映射到唯一 cluster；校验一个 image
  不跨 dev/confirmation；
- 两模型 exact orbit pairing 后才生成 shared weights；一个 cluster 的全部 cells、findings、
  models 一起带权；
- 输出每 cluster 最大 orbit contribution、effective cluster count、每模型/stratum unique
  cluster count；
- 当前 bootstrap 会拒绝任何丢失 stratum 的 multinomial draw（`:635-654`），这是条件化
  bootstrap。authorizing 版应优先使用 strictly-positive shared multiplier（如冻结的
  exponential/Bayesian multiplier）或预先冻结最大可接受 rejection rate；超过即失败，
  不能静默用重抽后的分布。

### P0-4 — directional admission 与 calibration support 不能由 isotonic 自证

**发现。** `_fit_isotonic` 只检查 pooled covariance 为正，然后强制
`IsotonicRegression(increasing=True)`（source `:118-136`）。因此原始四 vote-bin mean
可以有严重 reversal，fit 后仍必然“单调”。这不满足 draft `:158-160` 的
non-positive **or non-monotone** admission rule。

同时 calibrator 仅由 canonical dev cell 拟合，但 actual 15 cells、additive scores 和 Haar
rotations 全部经 `np.interp(... left=y[0], right=y[-1])` 静默 clip（source `:139-150,
:439-447`）。如果 Haar 比 observed 更常出 dev score range，PAEL 可以测到 clipping/saturation
差异，而不是 clinical orientation。

**最小修复：**

- dev-only admission 在 fit 前检查四 bin 的 raw-score location 按 `0<1<2<3` 正向；冻结
  adjacent-bin contrast、rank correlation/ordinal slope 与 image-cluster bootstrap rule；
  isotonic fit 的形状不能作为 admission 证据；
- 保留 cross-fitted dev diagnostic，但增加每 fold 的 bin support、OOF Brier/NLL、slope/
  intercept 和 reliability；
- 对 confirmation actual、additive、每类 single-axis 与 Haar score 分别报告 calibration
  support outside-rate、tail distance 和 clip mass，按 model/finding/stratum macro；
- 预先冻结 overlap gate。若 observed 与 Haar 的 out-of-support rate/距离不在允许范围，
  primary PAEL 不授权；可报告 raw-margin sensitivity，但不能用其替代 proper-loss gate；
- calibrator family、tie handling、probability floor 与 folds 在 config 中固定并 hash-bind。

### P0-5 — `B0>0` 不足以保护 `R=PAEL/B0`

**发现。** 代码只要求 macro `baseline_point>0`，bootstrap denominator 只要不是零就计算
ratio（source `:565-605`）。一个极小、强偏斜或 bootstrap 下接近零的 `B0` 会让 `R` 极不
稳定，却仍可能越过 5% operational threshold。

**最小修复：**

- 始终同时报告 absolute `theta_PAEL`、`B0`、`R=theta/B0`，坚持 ratio-of-macro-means；
- 输出每 bootstrap draw 的 macro `B0` 分布、最低分位数、SE/CV 与各 stratum denominator；
- 在看 outcome 前通过设计模拟冻结 denominator regularity rule（例如预冻结的
  `B0_min` 与 relative-SE/low-quantile guard）；任何 draw 触及 guard 时 ratio
  **non-authorizing**，只保留 absolute theta；
- 每 draw 重新计算 denominator 的现有做法保留；不要改成 mean of orbit ratios；
- 若无法为 5% ratio 给出 outcome-blind 稳定性界限，则应把 absolute theta 设为 primary，
  `R` 降为描述量，而不是事后从结果选择较好看的尺度。

### P0-6 — authorizer 所需临床与工程 gates 尚不在计算图

prototype 明确永远输出 `authorized=False`，这是正确的。真正 promotion 还缺：

- per-model clear-case introduced/repaired/net harm；当前 `_summarize` 把所有模型、findings、
  strata cell-pooled（source `:471-482, :701-703`）；
- 四个 finding 各自的 PAEL/B0/R 与 3/4 direction、`-0.05` meaningful-opposite guard；
- identity-render 与 duplicate-wording noise ceiling；control rows被 validator要求，但 v4
  analyzer 没有计算它们；
- exact both-model conjunction；模型集合目前由 payload 动态生成（`:730-753`）；
- clinician axis admission 与 independently randomized human product control 的 hash-bound
  pass artifact；
- authorizer 从 raw bound inputs 独立重算所有 gate，不信任上游传入的 `passed=True` 或
  metric booleans。

上述任一缺失时，结果只能是 diagnostic PAEL。human product control 缺失时，最大 claim
仍应严格限于“两个分别 admitted axes 下的 model-score nonseparability”，不能写成人类
临床等价 product defect。

### P0-7 — LORO 在当前 v4 schema 中不可实现

draft `:154-156, :178-180` 要求 reader identity 可用处报告 LORO。v4 payload validator
只保留 `reader_votes` count；prototype 没有 named reader 字段。由 aggregate
`q∈{0,1/3,2/3,1}` 无法恢复 1/3、2/3 中是哪位 reader 为正。

**可实现路径已经存在，但尚未接入 v4：**
`configs/cecd_reader_threshold_alias_sensitivity_v1.json` 冻结了 panel
`R8/R9/R10`、manifest hash、join key `(image_id,finding)`、拒绝 positional list、每次遗漏
一 reader 后在 dev 重拟合 calibrator并 apply confirmation。authorizing 输入必须复用这个
named mapping；不得从 `[0,1,0]` 位置猜 reader identity。

最小执行合同：

1. 从 hash-bound reader manifest 按 `(image_id,finding)` join named votes，并核对 sum 等于
   payload `reader_votes`；缺失、重复、身份集合不等均 fail closed；
2. 对 omitted R8/R9/R10 分别生成 `q_-j∈{0,0.5,1}`，每个 variant 在 dev 重新拟合同一家族
   calibrator，confirmation apply-only；
3. 三个 variant 使用同一 whole-image shared bootstrap，并保持两模型/findings 同 draw；
4. 明确它是固定三 reader panel robustness，不是 reader-population inference；reader
   不 bootstrap；
5. LORO 是 mandatory sensitivity，不应事后变成额外显著性 gate；但稳定反号必须阻止
   reader-grounded mechanism 升级。

### P0-8 — 两个 collision control 目前只有文字，没有可执行输入合同

draft `:136-146` 明确规定：MetaRA/composite-MR 与 semantic-boundary-proximity control
吸收 PAEL 时，不授权 fusion-orientation claim。prototype 无二者输入、feature 或 gate。

**MetaRA-style CE adaptation 的最小输入：**

- 对每个 `(model,image,finding,r,p)` 绑定四 cell：clean `(r0,p0)`、render-only `(r,p0)`、
  wording-only `(r0,p)`、joint `(r,p)`；
- 保存 exact prompt text/hash、source/transformed image hash、transform family/parameters、
  proposition mapping、admission artifact hash 与三态 logits；仅有 `render_id/prompt_id` 不足以
  证明 paper-faithful input；
- 冻结 reader-label-free features：single-axis argmax flip、joint flip、
  “singles stable but joint flips”、三态 JS/KL divergence、joint excess beyond the two single
  axes；这些从现有 logits 可算；
- 若用 reader q 定义 correctness，必须另标为 reader-grounded variant，不能冒充“完全不知道
  reader votes”的 generic collision control；
- 因当前是 one-token atomic CE，只能称 **MetaRA-style CE adaptation**，除非另有官方
  conformance test；不能声称 faithful MetaRA reproduction。

**boundary control 分两级：**

1. 当前 payload 可立即实现的只是 dev-frozen logit-margin control：clean margin、两个
   single-axis minimum absolute margin、到 tristate top-two boundary 的最小 gap、full-orbit
   boundary crossing count；它应作为最小 P0 generic alternative；
2. 真正 Yang-style semantic-plane/boundary proximity 需要当前 payload 没有的 frozen
   representations：model/checkpoint/tokenizer hash、选定层的 normalized visual/fused embedding、
   text proxy embeddings、proxy set、plane basis、norm、projection/distance formula、raw input
   hashes。没有这些只能叫 margin proxy，不能叫 semantic certification baseline。

二者的 feature construction 不得使用 reader votes；若用 dev reader loss拟合“absorption”
模型，必须将 feature extractor 和 regression family 在 dev 冻结，confirmation apply-only。
任一 generic control 吸收 held-out PAEL，或 boundary residual 不再成立，就停止
fusion-induced clinical orientation 解释。

## P1：不一定改变主点估计，但会使推断或解释不可信

### P1-1 — calibration-fit uncertainty 只能“条件化”，不能被忽略

主 bootstrap 固定 dev map，估计的是“条件于这一次 dev calibration”的 confirmation
sampling uncertainty；这是可接受的 primary conditional estimand，但必须如此命名。每
model/finding 只有 80 个 canonical dev orbits，isotonic step locations 的不确定性可能明显。

沿既有 alias sensitivity contract 增加 1,000-draw two-sample nested bootstrap：分别按 whole
dev image 重采样并重拟合 calibrator，再按 whole confirmation image共享重采样。nested CI
不必成为第二显著性 gate，但若方向反转、interval 极度膨胀或 B0 stability失效，必须作为
construct warning，并阻止强机制表述。禁止在 confirmation 中选择 logistic/isotonic 中
较好的一个；family/primary-secondary 次序必须 dev 前冻结。

### P1-2 — alternative null summaries 未使用同一个 estimand

primary Haar PAEL 是 per-model、orbit-first、16-stratum macro；但
`_summarize` 对 matched/cell/sign null 把所有 cell、findings、models 直接 pooled
（source `:471-482, :701-725, :765-794`）。这些 sensitivity percentile 与 primary PAEL
不在同一 weighting target 上，可能由样本较多的模型/stratum 主导。

所有 null 和 clear-case sensitivity 都应统一成：orbit-first → fixed 16-stratum macro →
per-model → shared whole-cluster inference；再报告 pooled 只能作附加描述。

### P1-3 — matched-orbit “随机化”实际上只是 sorted-list cyclic shift

`_matched_donors` 对每个 stratum 只随机一个 offset，然后在 sorted indices 上循环移位
（source `:400-424`）。它不是 uniform permutation/derangement，只有 `n-1` 种 donor maps；
若 image IDs 的排序携带 site/time/source 结构，会形成伪随机化。当前输出正确地称其为
descriptive sensitivity、不是 exact p-value（`:785-793`），这条边界必须保留。

修复时以 whole-image bundle 为单位，在 dev-frozen overlap blocks 内做 keyed uniform
permutation/derangement；保留 fixed clinical cell coordinates。若无法满足 cluster-level
exchangeability，仍只报告 reference percentile，任何
`approximate_reference_tail_fraction` 不进入 authorizer。

### P1-4 — finite-MC seed 不应依赖全局 orbit 顺序

当前每 draw 建一个 RNG，然后顺序遍历全部 orbits（source `:707-725`）。虽然排序通常稳定，
但删除或新增前序 orbit 会改变后续全部有限 rotations。用 P0-2 的 orbit-keyed RNG 可让每个
orbit 的 reference 独立可复核，并把 manifest/stream hash 写进输出。还应绑定 NumPy/BLAS
environment或直接保存 center bases/rotation-plan digest，避免不同数值环境悄悄改变结果。

### P1-5 — confirmation calibration diagnostics 存在 cell pseudo-replication

prototype 对每个 orbit 的全部 transformed cells 重复同一 q 后汇总 calibration
（source `:796-821`）。作为“所有条件下输出分布”描述可以，但其 `n` 不能被解释为独立
clinical sample count。应同时给 canonical-cell calibration、orbit-first transformed
calibration 和 image-cluster CI；论文 calibration claim以 canonical/apply-only为主。

## P2：工程与审稿防御完善

### P2-1 — 当前合成测试没有覆盖 authorizing failure surface

现有测试覆盖 Haar 几何、纯 additive 零效应、generic/random 与 target-localized fixture、
split overlap、artifact tamper、cell deletion、matched-null unavailable 和共享 multiplier。
但正例 fixture 在 clear bins 内直接搜索“最大化目标 Brier harm”的 interaction
（test `:63-89`）；它适合测试 statistic sensitivity，不足以证明识别不依赖目标构造。

promotion 前至少增加这些纯合成 tests：

1. 15/16 strata、错误 finding、单模型、3×3 grid、quota 少一例全部失败；
2. 两模型少一个 orbit、同 image 不同 patient、partial patient mapping、跨 split patient
   全部失败；
3. `null_draws=4095`、无 antithetic metadata、MC-SE 不收敛、改变 doubling rule 全部失败；
4. large interaction 但 Haar与observed calibration clipping rates不匹配时失败；
5. pooled covariance为正但 adjacent vote-bin reversal 的 calibrator admission失败；
6. `B0=0`、near-zero、bootstrap low-tail near-zero 的 ratio不授权；
7. multi-finding同 image 的全部 rows在每一 bootstrap draw共享同 multiplier；两模型也完全
   相同；
8. unnamed reader list、manifest vote-sum mismatch、任一 LORO refit缺 bin失败；
9. alternative null 的 per-model 16-stratum macro 与手算一致；
10. authorizer拒绝 missing human/MetaRA/boundary/identity artifacts，且拒绝任何 stress-null
    pseudo-p-value作为 primary证据；
11. 一个 interaction orientation 由独立 latent causal variable 生成、而非直接优化 reader
    loss 的正例；其 same-spectrum random-orientation twin失败。

### P2-2 — 结果 artifact 应提供完整审计链

除已有 source/payload/bundle hash 外，增加 authorizing config hash、reader manifest hash、
admission/human-control hashes、raw prompt/image hashes、cluster-map hash、exact orbit-set hash、
Haar stream/doubling trace、bootstrap plan/rejection rate、environment lock和所有输入的
canonical schema version。任何 source/config/hash mismatch 都应阻止 recomputation-equivalent
authorization。

## 目标耦合与伪随机化总审计

| 项目 | 当前判断 | authorizing 边界 |
|---|---|---|
| `q=votes/3` 对模型 score | **独立，合格** | 必须来自 hash-bound reader manifest，不能从模型答案派生 |
| `I=M-main effects` 与 cell score | 数学相关但不再像 v3 那样直接重建分类 target；paired proper-loss contrast可保留 | 只能称 algorithmic additive counterfactual / isospectral excess，不称 randomized causal effect |
| dev isotonic 用 q | 合格的 dev supervision | confirmation不可 refit；family/threshold/overlap gate预冻结 |
| Haar orientation | 合格的 deterministic geometry reference | 不是 randomization law，不报告/使用 exact p-value；MC convergence另行证明 |
| matched orbit | 条件在 reader-vote上且仅 cyclic shift | descriptive sensitivity；没有 whole-image exchangeability就不得叫 CRT |
| bootstrap rejected draws | 不依赖 response value，但条件在 observed stratum support | 报告 rejection；高 rejection fail或改 strictly-positive multiplier |
| synthetic positive fixture | 直接用 target 搜索 harmful `J` | 只测试 sensitivity；不能作为无 target coupling 的 identification test |
| clinical admission/transform selection | prototype不绑定其来源 | 必须证明在任何 model output 前冻结并 hash-bind；否则存在 outcome-guided transform selection |
| MetaRA/boundary predictor | 尚不存在 | features不使用 reader votes；dev-only冻结，confirmation apply-only |

## 最小修复顺序

按依赖关系执行，避免在错误 estimand 上继续加统计复杂度：

1. **冻结 authorizing config/schema：** exact models、4 findings、5×3 grid、20/60 quotas、
   reader panel、cluster mode、exact orbit pairing、admission/control hashes；实现 P0-1/P0-3
   fail-closed tests。
2. **重写 numerical reference contract：** orbit-keyed antithetic Haar，最低 4096，独立
   stream MC-SE 与 frozen doubling；先用合成数据验证，不读 confirmation outcome。
3. **闭合 calibration：** raw directional admission、OOF diagnostics、support-overlap/clipping
   gate；主 inference明确条件于 dev map，并预注册 nested fit-uncertainty sensitivity。
4. **闭合 estimand与 denominator：** 所有 primary/sensitivity统一 orbit-first、per-model、
   exact 16-stratum macro；冻结 `B0` regularity rule；输出 absolute theta、B0、R及 bootstrap
   denominator trace。
5. **闭合 clinical decision：** per-finding方向、clear net harm、identity/duplicate noise、
   both-model conjunction；authorizer必须从 bound raw inputs重算。
6. **接入 named-reader LORO：**复用已冻结 manifest与 alias contract；不得依赖 aggregate votes
   反推身份。
7. **实现 collision controls：**先做当前 logits 可计算的 MetaRA-style CE adaptation 与
   margin-boundary control；若要声称 Yang-style semantic-plane，先新增并冻结 representation/
   proxy inputs。绑定 human product control。
8. **补齐 adversarial synthetic suite 与独立 recomputation verifier。** 全部通过后才创建新
   `authorizing-v4.x` artifact version；旧 prototype artifacts 永远保持 non-authorizing。

## 最终 promotion 条件

只有以下三层同时闭合，v4 才可从 diagnostic 升为 authorizing：

1. **统计闭合：** exact 16 strata、paired cluster map、Haar MC convergence、calibration support、
   stable B0 与 raw-input recomputation；
2. **构念闭合：** reader-distribution Brier方向、clear net harm、per-finding与两模型 guards、
   LORO/nested-calibration无致命 reversal、human product control合格；
3. **碰撞闭合：** MetaRA/composite-MR与boundary proximity不能吸收 held-out PAEL，且这些
   controls有真实可执行输入而非文档占位。

在这之前，最强可信表述仍是：

> v4 prototype 计算一个 outcome-independent-reader-target、additive-counterfactual、
> isospectral-orientation-adjusted的描述性 Brier excess；它尚未证明该 excess在冻结的完整
> 两模型临床设计上数值收敛、校准可支持且超出普通 composite-MR/boundary fragility。

这是一条值得继续的 measurement path，但还不是可授权的 mechanism gate，更不是方法或
metric novelty。

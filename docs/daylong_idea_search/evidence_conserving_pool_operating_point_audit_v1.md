# Evidence-Conserving Pool：工作点独立审计

## 问题

`evidence_conserving_pool_huatuo_v1.json` 的嵌套模型没有通过增量门。本审计进一步拆开单项分数，判断 e-mixture 是提高了病例排序，还是只改变了阳性工作点。

## 冻结设置

- 模型：HuatuoGPT-Vision-7B。
- development：840 claims，746 images；每个 finding 的 vote-0/vote-3 各 60 例。
- confirmation：266 claims，266 images；7 个 findings 的 vote-0/vote-3 各 19 例。
- 分数：`final_margin`、每个 16/64/576 分区的 `raw_max` 与 `e_mix`。
- 每种分数、每个 finding 都只在 development vote-0 上冻结 95% 分位阈值；confirmation 不调阈值。
- 5,000 次 finding×label 分层的配对 bootstrap；同一次抽样用于全部分数。

## Confirmation 结果

| 分数 | Macro AUROC | Macro FPR | Macro Recall |
|---|---:|---:|---:|
| final margin | 0.7099 | 3.76% | 36.84% |
| raw max, 16 regions | 0.6510 | 9.77% | 21.80% |
| e-mix, 16 regions | 0.7178 | 18.80% | 36.84% |
| raw max, 64 regions | 0.6221 | 9.77% | 11.28% |
| e-mix, 64 regions | 0.7202 | 19.55% | 42.11% |
| raw max, 576 regions | 0.6043 | 10.53% | 15.04% |
| e-mix, 576 regions | 0.7040 | 25.56% | 51.13% |

相对 final margin 的配对差值：

| Partition | e-mix AUROC Δ (95% CI) | FPR Δ (95% CI) | Recall Δ (95% CI) |
|---:|---:|---:|---:|
| 16 | +0.0079 [−0.0712, +0.0871] | +15.04pp [+8.27, +21.80] | +0.00pp [−9.02, +9.02] |
| 64 | +0.0103 [−0.0665, +0.0890] | +15.79pp [+8.27, +23.31] | +5.26pp [−3.76, +15.04] |
| 576 | −0.0059 [−0.0851, +0.0740] | +21.80pp [+13.53, +30.08] | +14.29pp [+4.51, +24.06] |

## 严格结论

1. **e-mixture 不是保守规则。** 它在 confirmation 上显著提高 FPR，而且 region 越细，FPR 越高。
2. **相对 final margin 没有确认的排序增益。** 三个 partition 的 AUROC 差值都约为零，且置信区间宽幅跨零。
3. **576-region 的 recall 增益属于更激进的阳性工作点。** recall 增加 14.29pp，但 FPR 同时增加 21.80pp，不能解释为幻觉缓解。
4. **e-mixture 确实优于 raw max，但这只说明它修正了 naive maximum 的 search-size 缺陷。** e-mixture 相对 matched raw max 的 AUROC 增量为 +0.0669/+0.0981/+0.0997，置信区间均排除零；仍未超过模型已有 final margin。
5. development 中所有分数都按 5% FPR 定标，而 e-mixture 在 confirmation 达到 18.8%–25.6% FPR，说明其 `p -> e` 校准没有稳定迁移。单看 null mean 接近 1 会掩盖尾部失配。

因此，当前 e-mixture 的正确定位是：**一个比 raw max 更连贯的局部聚合器，但不是增量临床证据，也不是可用的 hallucination mitigation。** 不应放量或包装为正向方法。

## 产物

- 源码：`anchor/corrected_sgta/audit_evidence_conserving_pool_operating_point_v1.py`
- 完整 JSON：`corrected_runs/daylong_idea_search_v1/evidence_conserving_pool_operating_point_audit_huatuo_v1.json`


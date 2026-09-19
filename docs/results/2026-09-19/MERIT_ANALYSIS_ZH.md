# MERIT 跨模型、跨数据集表现分析

日期：2026-09-19。仅基于当前保存的答案、评分器及执行 trace；未改变方法、重跑生成或按标准答案挑选结果。汇总文件不是最终通过质量验收的排行榜。仍在运行的云端输出不在本地快照覆盖保证内。

## 已确认的同样本结果

采用现有评分器逐题对齐：CE 为严格正确率，OE 为答案 token recall，按样本加权，以下均 ×100。

| 模型／数据集／分组 | N | Greedy | MERIT | 差值 pp |
|---|---:|---:|---:|---:|
| Huatuo VQA-RAD 总体 |451|62.8193|57.1558|−5.6635|
| Huatuo VQA-RAD CE |251|76.4940|68.5259|−7.9681|
| Huatuo VQA-RAD OE |200|45.6576|42.8864|−2.7712|
| Huatuo SLAKE 总体 |2094|54.6920|54.4761|−0.2159|
| Huatuo SLAKE CE |836|70.2153|62.7990|−7.4163|
| Huatuo SLAKE OE |1258|44.375|48.945|+4.570|
| Huatuo SLAKE CT |865|59.306|49.033|−10.273|
| Huatuo SLAKE X-Ray |740|45.985|58.404|+12.419|
| Huatuo SLAKE MRI |489|59.705|58.161|−1.544|
| Huatuo SLAKE English |1061|55.979|55.017|−0.962|
| Huatuo SLAKE Chinese |1033|53.370|53.920|+0.550|

VQA-RAD 改善23题、恶化53题、持平375题；SLAKE 改善278题、恶化271题、持平1545题。OE 有部分分值，因此改善题数更多不保证总分更高。题目共享图像，不能把所有问题视为独立样本推断显著性。

## 机制证据与边界

1. **相似度证据不能用于确认器官存在。** VQA-RAD `vqa-rad-test-0005`：标准答案 No，Greedy No，MERIT Yes。CT 路由；BiomedParse 返回 unknown_anatomy_or_sequence，无证据；UniMed 左／右肾图文相似度约0.535／0.512。这不是存在概率。融合输出与误读该证据相符，但移除证据的对照尚未运行，不能仅凭 trace 宣称严格因果。
2. **正确路由不保证专家正确。** `vqa-rad-test-0054`：纵隔增宽，标准答案及 Greedy 为 Yes；CheXagent 输出 No，融合也为 No。专家错误与基座退化一致。该答案只有很少 token，不是截断或解析失败。
3. **能力覆盖不等于问题覆盖。** `vqa-rad-test-0068`：小肠梗阻，标准答案／Greedy Yes，融合 No。器官目录和分割不提供排除梗阻的充分证据。当前 all_evidence 运行样例显示 admission disabled、vector_gate off、behavior_probe off，缺少可靠的问题级证据筛选；这是当前实验配置，不可推断为所有 MERIT 版本的属性。
4. **不同模态的专家价值不同。** SLAKE CT −10.27 pp 与 X-Ray +12.42 pp 相互抵消，说明总分不能代表统一退化。胸片专用专家提供正收益，而 CT 全图相似度／通用分割可能缺乏判别当前问题的能力。模态分组是相关性证据，不是隔离专家效果的消融。
5. **适配不等同于跨基座效果保证。** Huatuo 使用原生序列化和语义文本证据，training_free_spatial=False；当前并非完整空间分支的效果验证。基座更强、不同指令敏感性可能使错误证据的机会成本更大，但未做匹配消融前只是解释假设。
6. **输出与指标也要分开。** OE recall 奖励参考词覆盖但不惩罚全部额外错误内容，长答案的 recall 提升不自动等于事实质量提升；本次没有据此更改用户指定指标。

## 其他数据集：哪些结论暂不能下

| 结果版本 | N | 分数 ×100 | 限制 |
|---|---:|---:|---|
| LLaVA MERIT historical compact-rows VQA-RAD |451|53.9566|历史版本，不能直接当作当前扩充专家版|
| LLaVA MERIT historical compact-rows SLAKE |2094|39.4856|历史版本，基座／协议差异需匹配|
| LLaVA MERIT common-protocol compact-rows PathVQA |6719|28.1920|旧版本，不能标为最新 MERIT+QUILT|
| LLaVA MERIT common-protocol compact-all PathVQA |6719|27.4899|同上|
| LLaVA MERIT native-expanded-v2 MMMU |10500|20.4286|608空答；官方确定性解析率71.3143%，3012回退计错，待修复|

MMMU 中不可解析不等于语义上全部错误；但也不能把它们默认为正确。应先修复／复核格式与空答，再判断专家方法本身的收益。医学专家面对30学科的能力覆盖限制是合理假设，尚缺分学科的匹配对照证明。

PathVQA capability-pool 的 selection-summary 记录6719题中 QUILT 被选3471次，新增三个 UniMed 病理目录分别3／12／10次。这是调用计数，不是路由正确率、覆盖正确率或最终成绩。该目录与旧 PathVQA compact 分数不是同一版本，不能拼接做因果结论。当前快照未找到该新版本统一评分文件，故不编造最新 MERIT+QUILT 数值。

MIMIC Huatuo MERIT 只有已检查的4例工程 canary：专家调用成功不等于报告事实正确，观察到直接复述专家分值的现象。不能以4例代替全量报告结果或声称效果已验证。

## 建议的最小定位实验（尚未执行）

先固定模型、图像、问题提示、解码及样本；复用专家缓存，仅重新融合：

- 无证据但保留 MERIT 外层提示：隔离提示／流程影响。
- CT 去除 UniMed 目录证据：隔离相似度目录影响。
- 保留专家但增加问题级适用性／冲突检查：这是新方法版本，需单列，不能替换原实验。

这些比较应在预先确定的开发样本执行，不按测试答案选择保留哪一次生成。所有旧结果保留，修复输出单独记账。

## 证据定位

- ANCHOR `anchor/medeval/evaluate_mixed_vqa_table.py`：现用混合评分；`anchor/corrected_sgta/evaluate_medheval_answers.py`：对齐及解析。
- `scored_artifacts.json`：评分文件逻辑路径、文件哈希、答案哈希与指标口径；包含历史及部分集，不是合并排行榜。
- merit-feddg-expert-coverage：`runs/huatuo-merit-full-v1/{vqa_rad,slake}/cases/`、`official_review/`；`merit_feddg/huatuo_generalist.py`；`scripts/run_huatuo_merit_full.py`。
- MMMU：`runs/mmmu-native-expanded-v2/official_review/answers.jsonl` 与官方评分文件。未上传逐题答案、图像或原始临床资料。

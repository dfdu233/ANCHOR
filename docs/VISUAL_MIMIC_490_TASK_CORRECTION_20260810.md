# Visual-MIMIC 490 task-semantics correction

## Verdict

`visual_mimic_oe` 的 490 例不是短答案 OE-VQA，而是 chest-X-ray **report generation**。
当前生成结果可以完整复用，但现有 `evaluate_oe_vqa` 词面分数与 OE 主表分类不能用于论文。

## Direct evidence

- Frozen manifest 每一行的 prompt 都要求 `Generate a medical report summarizing the key findings...`。
- Reference 是多 claim 放射学报告，不是一个 finding/短答案；例如第 0 例同时描述 hyperinflation、
  left apical hematoma、bilateral effusions 和 heart size。
- `configs/unified_eval/baseline_matrix_v1.json` 与
  `prepare_baseline_matrix_inputs.py` 把它硬编码为 `open_vqa`。
- 所有 generation runners 已使用 256-token budget，实际输出也是完整报告，因此**不需要重新生成**；
  错误发生在 task registry、scoring route、coverage ledger 和 paper exporter。

## Non-disruptive repair

正式 baseline 队列运行期间不修改 frozen config 或现有输出路径。建立独立 corrected scoring track：

1. 对每个已完整生成的 cell，用现有 `prepare_report_evaluation_pairs_v1.py` 连接 490 个 reference 与
   prediction，并显式写 `task=report_generation`、`modality=chest_radiograph`。
2. 用现有 `evaluate_oe_reports.py --clinical required` 计算 RadGraph、RaTEScore、CheXbert 与辅助
   BLEU/ROUGE/METEOR/token-F1；patient/study 从 MIMIC image path 解析。
3. 新 coverage ledger 将旧 OE score 标记 `superseded_wrong_task_semantics`，保留审计但不进论文表。
4. `export_baseline_paper_tables_v1.py` 把 `visual_mimic_oe` 路由到 report 表；原生成 provenance/hash
   不变，新 score 独立记录 evaluator/source hash。

## Fail-closed rule

- 不得把 report lexical overlap 称为 OE accuracy 或临床 hallucination rate。
- 临床 metric 环境/权重或 direction sanity 失败时，该 cell 标 N/A/failed scoring，不能回退到旧 OE
  词面分数冒充完成。
- 修复只重评估，不重跑模型；若任何生成 qid/order 不完整，按原 cell 续跑，而不是用缺行分数。

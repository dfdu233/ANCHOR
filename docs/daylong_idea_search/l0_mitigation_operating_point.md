# L0 — Formal LLaVA mitigation operating-point audit

## Question

在当前已完成的正式 CXR-VisHal 二元子集上，training-free 方法是否表现出真实的
阳性/阴性分离改善，还是主要停留在几乎总回答 Yes 的工作点？

## Reused artifacts

- dataset: `cxr_vishal`, binary subset `n=3669`
- model: LLaVA-Med-v1.5-Mistral-7B
- evaluator: corrected source-typed CE v8 output中的
  `primary_multiclass.by_answer_type.binary`
- paths: `corrected_runs/paper_baselines_v1/full_matrix_v1/derived_scores/llava/*/cxr_vishal/evaluation_ce_v7.json`

## Result

| Method | Accuracy | BAcc | Macro-F1 | Yes rate among parsed Yes/No | Parse rate |
|---|---:|---:|---:|---:|---:|
| DoLa | 47.34% | 49.80% | 33.27% | 99.25% | 98.23% |
| OPERA | 48.49% | 50.78% | 38.88% | 96.48% | 90.71% |
| PAI | 47.45% | 49.91% | 33.55% | 99.17% | 97.96% |
| VCD | 42.95% | 44.71% | 40.27% | 91.87% | 79.10% |
| VISTA | 47.53% | 50.00% | 33.33% | 99.23% | 98.86% |

## Interpretation

1. 这五种方法在该模型/数据集的二元任务上均没有形成可用的阴阳性分离；BAcc约为
   随机或更低，并伴随91.9%–99.3%的极端Yes率。
2. VCD较低的Yes率同时伴随更低BAcc和79.1% parse rate，不能解释为可靠的幻觉缓解。
3. 该结果支持继续检验 `criterion-shift / response-bias`，但**尚不能证明**所有方法
   只移动criterion：当前没有同一方法的连续强度/logit扫面，也缺少该正式cell对应的
   合格greedy对照。因此它只是L0现象，不是论文结论。

## Next decisive test

在一个具有合格greedy与多个连续干预强度的同模型CE缓存上，构造完整ROC/coverage
轨迹；若方法轨迹不超greedy阈值轨迹，则判为criterion-shift mirage。该分析可CPU完成。

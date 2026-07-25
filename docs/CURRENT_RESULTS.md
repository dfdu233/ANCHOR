# Current Results

This repository preserves compact reference summaries from prior ANCHOR/RULE/MMed-RAG experiments in `results_reference/`. These files are intended for audit and comparison, not as automatically refreshed claims.

Known reference directories include:

- `results_reference/rule_mimic_source_margin/`
- `results_reference/rule_source_robust_margin_reconfirm_v1/`
- `results_reference/rule_source_robust_margin_scale_v1/`
- `results_reference/rule_mitigation_v1/`
- `results_reference/mmedrag_word_center_final/`

Interpretation rules:

- Only claim results backed by a `metrics.json`, `summary.json`, or documented source cache.
- Registry smoke outputs prove configuration and path compatibility only; they are not experimental results.
- RULE VQA should report strict accuracy, parseable accuracy, and parse rate when available.
- Report generation should keep ROUGE/matching results separate from opencode or clinical judge scores.
- ConfGen claims require coverage, average set size, and non-vacuous guarantee checks.

The default packaged runner is intentionally conservative. It validates datasets and creates reproducible run records; full GPU inference should be launched through the preserved method-specific scripts or newly added bridges that emit the same schema.

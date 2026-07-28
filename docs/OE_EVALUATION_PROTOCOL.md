# OE / Report Evaluation Protocol

This repository separates open-ended VQA from report generation and reports every dataset/modality independently.

## Task definitions

- RULE/MMed-RAG binary VQA: generated sentence, then the official parser; never report metrics.
- Chest-radiograph report generation (MIMIC-CXR, IU-Xray, verified CheXpert): full report with clinical and text metrics.
- Ophthalmology/pathology reports: full report with modality-appropriate text or domain metrics; never RadGraph or CheXbert.
- MedHEval knowledge OE: semantic/LLM judging under its own rubric; never mixed with radiology reports.

## Generation

LLaVA-Med v1.5 uses `mistral_instruct`. The zero-retrieval baseline prompt follows the released RULE/MMed-RAG wording but contains no reference report. `official_rag` is a separate method and requires externally retrieved source reports. References and target labels must not enter prompts or selection.

```bash
PYTHONPATH=anchor python -m corrected_sgta.infer_oe \
  --model llava --dataset DATA.json --output RUN.jsonl \
  --llava-conv-mode mistral_instruct \
  --report-prompt-mode official_zero_shot
```

## Evaluation

RULE/MMed-RAG comparability metrics are BLEU, ROUGE-L, and METEOR. They are secondary because they do not establish clinical factuality. Chest-radiograph primary metrics are RadGraph F1, RaTEScore, and CheXbert. All runs retain raw reports, dataset, modality, prompt variant, study/patient ID, and fingerprints.

```bash
PYTHONPATH=anchor python -m corrected_sgta.evaluate_oe_reports \
  --input RUN.jsonl --output-dir EVAL_DIR --clinical required
```

Before a report baseline is admissible, run real/null/shuffled image sanity checks. A >90% normal-template rate, near-constant output, or negligible real-vs-null difference invalidates the report-generation claim even if ROUGE increases.

The pinned clinical packages passed the registered 100-pair matched-vs-contradiction direction test in `corrected_runs/oe_protocol_fix_v2/metric_direction100/aggregate.json`.

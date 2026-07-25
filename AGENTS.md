# Repository Guidelines

## Project Goal

This repository packages ANCHOR experiments for medical VLM hallucination under domain shift. Keep the default chest-X-ray path runnable first: RULE/MIMIC-CXR binary VQA and CheXpert subset report generation.

## Key Directories

- `anchor/corrected_sgta/`: maintained ANCHOR, SGTA, source-margin, ConfGen, report, and mitigation adapters.
- `third_party/`: vendored RULE, MMed-RAG, MedHEval, and baseline repositories.
- `data/`: Git-LFS tracked datasets and manifests.
- `configs/`: dataset, method, model, and judge registries.
- `results_reference/`: compact verified reference outputs.
- `scripts/`: one-click setup, smoke, run, and summary commands.

## Commands

```bash
bash scripts/bootstrap.sh
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/summarize_results.sh
PYTHONPATH=. python -m compileall -q anchor tests
```

## Rules

Do not commit model checkpoints, credentials, Hugging Face caches, conda environments, or transient failed logs. Large dataset files must be tracked through Git LFS. Every new experiment output must include dataset, model, method, seed, command, and fingerprint.

## Verification

Before pushing, run smoke checks on `mimic_cxr_rule` and `chexpert_subset_report`, compile Python files, inspect `git lfs status`, and verify `git status --short`.

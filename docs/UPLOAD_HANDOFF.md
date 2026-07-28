# Upload Handoff

This document records the repository state after packaging and uploading ANCHOR for server migration. It is a handoff note, not a paper claim file.

## Remote State

- Remote: `git@github.com:dfdu233/ANCHOR.git`
- Branch: `main`
- Uploaded commit: `a5c4b06e Add full ANCHOR reproducibility package`
- Git LFS upload: `14441/14441` objects, approximately `4.8 GB`
- Clean clone verification path on the source server: `/root/autodl-tmp/ANCHOR_REPRO_TEST`

GitHub emitted `GH010` during push because it validates a random sample of large LFS pushes. The push succeeded and the clean clone was able to pull default LFS data.

## Packaged Content

The repository contains the curated ANCHOR migration package:

- Core ANCHOR/SGTA code: `anchor/corrected_sgta/`
- Compatibility import path: `corrected_sgta -> anchor/corrected_sgta`
- Third-party code: `third_party/RULE/`, `third_party/MMed-RAG/`, `third_party/MedHEval/`, and `third_party/baselines/`
- Default datasets: `data/mimic_cxr_rule/` and `data/chexpert_subset_report/`
- Additional dataset entries: `data/medheval/`, `data/rule/`, and `data/mmedrag/`
- Configs: `configs/datasets.yaml`, `configs/methods.yaml`, `configs/models.yaml`, `configs/judges.yaml`, `configs/default_chest_xray.yaml`
- Run scripts: `scripts/bootstrap.sh`, `scripts/prepare_data.sh`, `scripts/run_smoke.sh`, `scripts/run_all.sh`, `scripts/summarize_results.sh`
- Compact references: `results_reference/`

## Default Dataset Verification

Clean clone verification pulled the default LFS data and passed:

```bash
bash scripts/prepare_data.sh --datasets mimic_cxr_rule,chexpert_subset_report --check-only
bash scripts/run_smoke.sh --datasets mimic_cxr_rule,chexpert_subset_report
bash scripts/summarize_results.sh
PYTHONPATH=. python -m compileall -q anchor tests
PYTHONPATH=. python -m pytest tests -q
```

Observed default dataset checks:

- `mimic_cxr_rule`: 3470 records, 694 image-manifest rows, image bytes `1139256082`
- `chexpert_subset_report`: 5000 image-manifest rows, image bytes `170790287`, marked `chexpert_subset_unverified`

## Fresh Server Commands

```bash
git clone git@github.com:dfdu233/ANCHOR.git
cd ANCHOR
git lfs pull
bash scripts/bootstrap.sh
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/summarize_results.sh
```

For faster first checks, pull only default chest-Xray assets:

```bash
git lfs pull --include="data/mimic_cxr_rule/**,data/medheval/images/**,data/chexpert_subset_report/**"
```

Use `HF_ENDPOINT=https://hf-mirror.com` for Hugging Face downloads when needed.

## Known Exclusions And Limits

- `third_party/baselines/VISTA/MMHal-Bench/` is excluded because the local copy contained unresolved Git LFS pointer files, not recoverable image objects. VISTA code remains packaged. Reacquire MMHal-Bench assets from upstream if needed.
- Hugging Face model checkpoints and cache directories are not committed. Models should be downloaded on demand.
- Credentials such as `/root/.netrc`, API keys, and service tokens are not committed.
- The current top-level runner is a reproducible registry/smoke/dispatch entrypoint. Heavy GPU inference code is preserved, but each full method still needs to be bridged to the unified run schema before claiming one-command full inference for every method.
- Registry-only outputs prove path/config compatibility and run-schema generation; they are not model accuracy results.

## Local Source Tree Notes

The source working tree may contain untracked website/project-page files such as `.openai/`, `app/`, `public/`, `package.json`, `package-lock.json`, and `vite.config.ts`. These were not part of commit `a5c4b06e` and were not uploaded as part of the reproducibility package. Review their claims before publishing them.

## 2026-07-28 Method-Documentation Addendum

Before the next data migration, method documentation was expanded to include `docs/METHOD_ZOO.md` plus the newly created method/protocol notes under `docs/`. The most important current direction is LET, with VISTA/SLA as the closest comparator and source-guided DG methods retained as motivation/diagnostics unless later gates pass.

Current local workspace also contains uncommitted business-code changes for LET/report/OE/parser and several new experimental scripts under `anchor/corrected_sgta/`. These are valuable but should be migrated in a separate audited code commit, not mixed into documentation-only updates.

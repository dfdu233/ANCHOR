# Dataset Policy

ANCHOR is packaged as a private repository with Git LFS for reproducible transfer between servers. The default datasets are chest-Xray only:

- `mimic_cxr_rule`: RULE/MIMIC-CXR Yes/No VQA records in `data/mimic_cxr_rule`, with images resolved through `data/medheval/images`.
- `chexpert_subset_report`: CheXpert subset report-generation records in `data/chexpert_subset_report`.

Large image/table/archive files are tracked by Git LFS according to `.gitattributes`. Model checkpoints, Hugging Face caches, credentials, conda environments, transient logs, and failed exploratory outputs must not be committed.

Use:

```bash
git lfs pull
bash scripts/prepare_data.sh --datasets mimic_cxr_rule,chexpert_subset_report --check-only
```

The CheXpert subset is marked `chexpert_subset_unverified`; do not claim it is the official Stanford CheXpert external-hospital benchmark unless provenance is independently verified.

MIMIC-CXR access remains controlled by PhysioNet credentials. Do not commit `/root/.netrc`, passwords, or download tokens. If missing images must be recovered, use the RULE-aligned manifest and download only required files.

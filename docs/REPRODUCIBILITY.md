# Reproducibility

Fresh server setup:

```bash
git clone git@github.com:dfdu233/ANCHOR.git
cd ANCHOR
git lfs pull
bash scripts/bootstrap.sh
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/summarize_results.sh
```

Default run:

```bash
bash scripts/run_all.sh
```

is equivalent to:

```bash
bash scripts/run_all.sh \
  --datasets mimic_cxr_rule,chexpert_subset_report \
  --methods greedy,source_margin,source_word_center
```

Useful focused runs:

```bash
bash scripts/run_all.sh --datasets mimic_cxr_rule --methods greedy,source_margin,sca_t_tim_kl
bash scripts/run_all.sh --datasets chexpert_subset_report --methods greedy,source_word_center --judges rouge,opencode
```

Set `HF_ENDPOINT=https://hf-mirror.com` for model downloads in China. `OPENCODE_MODE=mock` is the default smoke mode; use a real opencode backend only after credentials and local service configuration are available.

Before pushing changes:

```bash
bash scripts/bootstrap.sh --dry-run
bash scripts/run_smoke.sh --datasets mimic_cxr_rule,chexpert_subset_report
PYTHONPATH=. python -m compileall -q anchor tests
PYTHONPATH=. python -m pytest tests -q
git lfs status
git status --short
```

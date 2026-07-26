# ANCHOR

ANCHOR is a reproducibility package for medical VLM hallucination and domain-shift experiments. It bundles the maintained ANCHOR/SGTA code, RULE, MMed-RAG, MedHEval, adapted mitigation baselines, and dataset manifests for moving experiments to a new server.

Default datasets:

- `mimic_cxr_rule`: RULE/MIMIC-CXR binary VQA.
- `chexpert_subset_report`: CheXpert subset report generation (`chexpert_subset_unverified`).

Quick start:

```bash
git lfs pull
bash scripts/bootstrap.sh
bash scripts/run_smoke.sh
bash scripts/run_all.sh
bash scripts/summarize_results.sh
```

Custom examples:

```bash
bash scripts/run_all.sh --datasets mimic_cxr_rule --methods greedy,source_margin,sca_t_tim_kl
bash scripts/run_all.sh --datasets chexpert_subset_report --methods greedy,source_word_center --judges rouge,opencode
bash scripts/run_all.sh --datasets medheval --methods vcd,dola,opera,pai
```

Models are not committed. Use `HF_ENDPOINT=https://hf-mirror.com` or the proxy variables documented in `docs/REPRODUCIBILITY.md`.

## Upload Handoff

See `docs/UPLOAD_HANDOFF.md` for the uploaded commit, Git LFS status, clean-clone verification, known exclusions, and fresh-server reproduction notes.

# ANCHOR-DG Runbook

Run from `/root/autodl-tmp/Hulu-Med/MedUniEval` with `PYTHONPATH=.`. Never add
MIMIC, CheXpert, PadChest, Harvard-FairVLMed, or a held-out LODO domain to a
filter, style-bank, or training command.

## 1. Validate

```bash
PYTHONPATH=. python -m compileall -q corrected_sgta tests
PYTHONPATH=. python -m unittest tests.test_anchor_dg tests.test_rule_source_group_adapter tests.test_rule_dg_adapter -v
python -m ruff check corrected_sgta/anchor_dg.py corrected_sgta/filter_anchor_dg_chest_sources.py corrected_sgta/build_anchor_dg_style_bank.py corrected_sgta/probe_anchor_dg_interventions.py corrected_sgta/train_anchor_dg.py tests/test_anchor_dg.py
```

## 2. Filter strict chest-radiograph sources

Score source images with the local BiomedCLIP and export a source-stratified,
model-score-blinded audit. The checked-in source paths below are real. The
current LLaVA alignment manifest is a Git-LFS pointer and must not be added.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m corrected_sgta.filter_anchor_dg_chest_sources score \
  --source rule_iuxray=corrected_runs/rule_source_hull_v1/source_manifest/train.rule_iuxray.json \
  --source slake_xray=corrected_runs/rule_source_hull_v1/source_manifest/train.slake_xray.json \
  --source vqa_rad_train=corrected_runs/rule_source_hull_v1/source_manifest/train.vqa_rad_train.json \
  --source-image-root rule_iuxray=. --source-image-root slake_xray=. \
  --source-image-root vqa_rad_train=. \
  --scores corrected_runs/anchor_dg_v2/filter/biomedclip_scores_v2.jsonl \
  --audit corrected_runs/anchor_dg_v2/filter/human_audit_100_v2.json
```

A human must inspect every `image` and fill `human_label` with `cxr` or
`non_cxr`. Do not expose the separate score JSONL during annotation. Calibration
fails closed until all 100 labels exist. This three-source audit is for the
final three-source bank. For a strict LODO fold, rerun `score`, `calibrate`, and
`apply` with the held-out source omitted from the command; a calibration that
has inspected the held-out domain is not reusable for that fold:

```bash
PYTHONPATH=. python -m corrected_sgta.filter_anchor_dg_chest_sources calibrate \
  --scores corrected_runs/anchor_dg_v2/filter/biomedclip_scores_v2.jsonl \
  --audit corrected_runs/anchor_dg_v2/filter/human_audit_100_v2.json \
  --output corrected_runs/anchor_dg_v2/filter/calibration.json
```

For an explicitly user-authorized exploratory run only, all sources can be
retained without inventing audit labels. This path is fingerprinted as
`unverified_source_override=true` and cannot support strict-bank paper claims:

```bash
PYTHONPATH=. python -m corrected_sgta.filter_anchor_dg_chest_sources apply \
  --source rule_iuxray=corrected_runs/rule_source_hull_v1/source_manifest/train.rule_iuxray.json \
  --source slake_xray=corrected_runs/rule_source_hull_v1/source_manifest/train.slake_xray.json \
  --source vqa_rad_train=corrected_runs/rule_source_hull_v1/source_manifest/train.vqa_rad_train.json \
  --source-image-root rule_iuxray=. --source-image-root slake_xray=. \
  --source-image-root vqa_rad_train=. --assume-all-chest \
  --output-dir corrected_runs/anchor_dg_v2/filtered_sources_assume_all
```

After calibration reaches at least 95% precision, emit filtered manifests:

```bash
PYTHONPATH=. python -m corrected_sgta.filter_anchor_dg_chest_sources apply \
  --source rule_iuxray=corrected_runs/rule_source_hull_v1/source_manifest/train.rule_iuxray.json \
  --source slake_xray=corrected_runs/rule_source_hull_v1/source_manifest/train.slake_xray.json \
  --source vqa_rad_train=corrected_runs/rule_source_hull_v1/source_manifest/train.vqa_rad_train.json \
  --source-image-root rule_iuxray=. --source-image-root slake_xray=. \
  --source-image-root vqa_rad_train=. \
  --scores corrected_runs/anchor_dg_v2/filter/biomedclip_scores_v2.jsonl \
  --calibration corrected_runs/anchor_dg_v2/filter/calibration.json \
  --trusted-source rule_iuxray --output-dir corrected_runs/anchor_dg_v2/filtered_sources
```

## 3. Build a robust source-only bank

For LODO, omit the held-out source from both `--source` lists and add
`--heldout-domain NAME`. The builder requires the human-validated filter report
and at least 32 independent images per included source.

```bash
PYTHONPATH=. python -m corrected_sgta.build_anchor_dg_style_bank \
  --source rule_iuxray=corrected_runs/anchor_dg_v2/filtered_sources/rule_iuxray.json \
  --source slake_xray=corrected_runs/anchor_dg_v2/filtered_sources/slake_xray.json \
  --source vqa_rad_train=corrected_runs/anchor_dg_v2/filtered_sources/vqa_rad_train.json \
  --source-image-root rule_iuxray=. --source-image-root slake_xray=. \
  --source-image-root vqa_rad_train=. \
  --filter-report corrected_runs/anchor_dg_v2/filtered_sources/filter_report.json \
  --max-images-per-source 64 \
  --output corrected_runs/anchor_dg_v2/style_banks/all_sources_seed42.npz
```

## 4. Data-driven intervention gate

Exit code `3` means no configuration passed and training must not start. The
nine default `(rho, beta)` configurations are screened on source-only content
and style metrics; at most three reach the VLM gate. Do not edit a failed JSON.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m corrected_sgta.probe_anchor_dg_interventions \
  --source-json rule_iuxray=corrected_runs/anchor_dg_v2/filtered_sources/rule_iuxray.json \
  --source-json slake_xray=corrected_runs/anchor_dg_v2/filtered_sources/slake_xray.json \
  --source-json vqa_rad_train=corrected_runs/anchor_dg_v2/filtered_sources/vqa_rad_train.json \
  --source-image-root rule_iuxray=. --source-image-root slake_xray=. \
  --source-image-root vqa_rad_train=. \
  --style-bank corrected_runs/anchor_dg_v2/style_banks/all_sources_seed42.npz \
  --max-samples 64 --output corrected_runs/anchor_dg_v2/gates/intervention_n64.json
```

## 5. Train only after a passing gate

`task_only` and `generic_augmentation` do not require style arguments. The
source-style objectives require the exact approved `rho` and `beta`. All
variants use identical rank, trust region, source schedule, and optimizer
budget. The trainer aborts on forbidden-path overlap or fingerprint mismatch.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m corrected_sgta.train_anchor_dg \
  --source-json rule_iuxray=TRAIN_IU.json --source-json slake_xray=TRAIN_SLAKE.json \
  --source-image-root rule_iuxray=. --source-image-root slake_xray=. \
  --forbidden-json heldout=HELDOUT.json --forbidden-image-root heldout=. \
  --heldout-domain vqa_rad_train --style-bank BANK_WITHOUT_VQA.npz \
  --gate-json PASSING_GATE.json --style-rho APPROVED_RHO --style-beta APPROVED_BETA \
  --objective anchor_dg --rank 16 --steps 64 --resume \
  --output corrected_runs/anchor_dg_v2/checkpoints/lodo_vqa_seed42.pt
```

Use `corrected_sgta.infer_rule_dg_adapter` for original-image greedy inference
and `corrected_sgta.evaluate_rule_vqa` for official full-text parsing. Do not run
unknown-target or full experiments until the LODO and external-pilot gates in
`docs/user_requirements.md` pass.

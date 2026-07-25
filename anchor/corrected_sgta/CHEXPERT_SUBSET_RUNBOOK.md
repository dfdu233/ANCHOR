# CheXpert Subset Report Evaluation Runbook

This adapter covers the `image`/`report` task exposed by
`ayyuce/chexpert-subset`. Run commands from
`/root/autodl-tmp/Hulu-Med/MedUniEval` with `PYTHONPATH=.`.

## Download and Prepare

```bash
env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u http_proxy -u https_proxy \
  HF_ENDPOINT=https://hf-mirror.com \
  hf download ayyuce/chexpert-subset README.md \
  data/train-00000-of-00001.parquet --repo-type dataset \
  --local-dir /root/autodl-tmp/datasets/chexpert-subset

PYTHONPATH=. python -m corrected_sgta.prepare_chexpert_subset \
  --parquet /root/autodl-tmp/datasets/chexpert-subset/data/train-00000-of-00001.parquet \
  --dataset-readme /root/autodl-tmp/datasets/chexpert-subset/README.md \
  --output-dir /root/autodl-tmp/datasets/chexpert-subset/processed-v1 \
  --revision 372166fb5f5004176fd0642f2290574958034629
```

The converter refuses to overwrite manifests. Existing extracted images are
reused only when their exact SHA-256 matches.

## Evaluation Inputs

- MMed-RAG annotation:
  `/root/autodl-tmp/MMed-RAG/data/test/report/chexpert_subset_test.json`
- MMed-RAG image root:
  `/root/autodl-tmp/datasets/chexpert-subset/processed-v1/images`
- RULE-shaped report interchange:
  `/root/autodl-tmp/RULE/data/test/chexpert_subset_report_test.jsonl`
- Maintained target manifest:
  `/root/autodl-tmp/datasets/chexpert-subset/processed-v1/anchor_report_manifest.json`

The MMed-RAG annotation loads with its native `MimicDataset` class. The checked
MMed-RAG release does not contain a complete standalone report decoder or
clinical evaluator, and RULE's released evaluator is binary VQA only. Therefore
the maintained path for model generation and scoring is ANCHOR's task-agnostic
runner plus report evaluator:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m corrected_sgta.run_anchor_transport \
  --manifest /root/autodl-tmp/datasets/chexpert-subset/processed-v1/anchor_report_manifest.json \
  --bank SOURCE_BANK.json --model llava \
  --raw corrected_runs/chexpert_subset/llava_pilot.raw.jsonl \
  --output corrected_runs/chexpert_subset/llava_pilot.json \
  --max-samples 128 --resume

PYTHONPATH=. python -m corrected_sgta.evaluate_anchor evaluate \
  --cache corrected_runs/chexpert_subset/llava_pilot.json \
  --manifest /root/autodl-tmp/datasets/chexpert-subset/processed-v1/anchor_report_manifest.json \
  --evaluator report --lambda-value 0 \
  --output corrected_runs/chexpert_subset/llava_pilot_eval.json
```

Add `--clinical --clinical-cache /root/autodl-tmp/model_cache/report_metrics`
for RadGraph, RaTEScore, and CheXbert. Without it, ROUGE-L is diagnostic only.

## Claim Boundary

The dataset card gives no institution, patient/study identity, official split,
or disease labels. All artifacts therefore record
`eligible_as_unknown_institution=false`. Do not call this Stanford CheXpert,
construct keyword-derived VQA labels, claim patient-disjoint evaluation, or use
it as the second-hospital DG endpoint until provenance is independently proven.

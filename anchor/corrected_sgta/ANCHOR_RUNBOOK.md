# ANCHOR Evidence Transport Runbook

Run from `/root/autodl-tmp/Hulu-Med/MedUniEval` with `PYTHONPATH=.`.
The primary implementation is non-parametric and never reads target labels
during generation or selection.

## 1. Build a Source Bank

The existing frozen RULE source manifests are supported directly, including
their two-message `conversations` schema.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m corrected_sgta.build_anchor_source_bank \
  --source rule_iuxray=corrected_runs/rule_source_hull_v1/source_manifest/train.rule_iuxray.json \
  --source slake_xray=corrected_runs/rule_source_hull_v1/source_manifest/train.slake_xray.json \
  --source vqa_rad_train=corrected_runs/rule_source_hull_v1/source_manifest/train.vqa_rad_train.json \
  --model llava \
  --raw corrected_runs/anchor_v1/llava_source.raw.jsonl \
  --output corrected_runs/anchor_v1/llava_source_bank.json \
  --resume
```

Replace `--model llava` and output names with `hulu` for Hulu-Med. Model
artifacts, tokenizer/config files, input manifests, code, record order, and
trajectory parameters are SHA-256 bound. The final bank contains no answer
text or answer token IDs.

## 2. Candidate-Oracle Source Pilot

Run each held-out source separately so leave-one-domain-out retrieval excludes
the held-out domain without exposing deployment target identity:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m corrected_sgta.run_anchor_transport \
  --manifest corrected_runs/rule_source_hull_v1/source_manifest/dev.rule_iuxray.json \
  --bank corrected_runs/anchor_v1/llava_source_bank.json \
  --model llava \
  --raw corrected_runs/anchor_v1/llava_iu_dev.raw.jsonl \
  --output corrected_runs/anchor_v1/llava_iu_dev.json \
  --exclude-source-domain rule_iuxray \
  --max-samples 128 \
  --resume
```

Repeat for `dev.slake_xray.json` and `dev.vqa_rad_train.json`. Each sample gets
exactly eight full responses: one greedy response and seven seeded nucleus
samples. `--candidate-batch 1` gives the most transparent seed provenance;
larger batches improve throughput but do not change the candidate budget.

## 3. Fit One Model-Level Lambda

Prepare a small JSON bundle:

```json
{
  "entries": [
    {
      "cache": "corrected_runs/anchor_v1/llava_iu_dev.json",
      "manifest": "corrected_runs/rule_source_hull_v1/source_manifest/dev.rule_iuxray.json",
      "evaluator": "vqa"
    }
  ]
}
```

Then fit only on source-held-out records:

```bash
PYTHONPATH=. python -m corrected_sgta.evaluate_anchor fit-lambda \
  --bundle corrected_runs/anchor_v1/llava_source_bundle.json \
  --lambda-grid 0,0.25,0.5,1,2 \
  --output corrected_runs/anchor_v1/llava_lambda.json
```

The fit objective is max-min over source domains, followed by macro and micro
utility. The output also records leave-one-source-domain-out performance.

## 4. Locked Unknown-Source Pilot

The target manifest must supply complete prompts and image paths. Labels may
be present for later evaluation, but are discarded before inference. Raw
MMed-RAG report-test files do not contain prompts and must first be converted
to the documented normalized schema; do not invent prompts inside evaluation.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m corrected_sgta.run_anchor_transport \
  --manifest TARGET_NORMALIZED.json \
  --bank corrected_runs/anchor_v1/llava_source_bank.json \
  --model llava \
  --raw corrected_runs/anchor_v1/llava_target256.raw.jsonl \
  --output corrected_runs/anchor_v1/llava_target256.json \
  --max-samples 256 \
  --resume
```

Do not pass `--exclude-source-domain` for deployment or locked target runs.

## 5. Full-Text Evaluation and ConfGen

VQA evaluation parses only the selected complete sentence:

```bash
PYTHONPATH=. python -m corrected_sgta.evaluate_anchor evaluate \
  --cache corrected_runs/anchor_v1/llava_target256.json \
  --manifest TARGET_NORMALIZED.json \
  --evaluator vqa \
  --lambda-config corrected_runs/anchor_v1/llava_lambda.json \
  --output corrected_runs/anchor_v1/llava_target256_eval.json
```

For reports, add `--evaluator report --clinical
--clinical-cache /root/autodl-tmp/model_cache/report_metrics`; this uses the pinned
RadGraph, RaTEScore, and CheXbert implementations. Without `--clinical`,
ROUGE-L is explicitly marked diagnostic and is not a paper endpoint.

Calibrate candidate sets on source-only bundles:

```bash
PYTHONPATH=. python -m corrected_sgta.evaluate_anchor fit-conformal \
  --bundle corrected_runs/anchor_v1/llava_source_bundle.json \
  --lambda-config corrected_runs/anchor_v1/llava_lambda.json \
  --coverage 0.90 \
  --output corrected_runs/anchor_v1/llava_confgen90.json
```

An infinite threshold is retained and reported as
`vacuous_guarantee: true`; it is never silently replaced by a finite set.

## Gates

Stop before target/full generation unless source-held-out candidate-oracle
headroom is at least 3 percentage points. Full-scale evaluation additionally
requires positive paired-bootstrap CI, unchanged parsing coverage, consistent
direction on at least two model–dataset endpoints, and no significant clinical
metric regression above 1 percentage point.

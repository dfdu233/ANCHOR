# Optimized SGTA / SGTA-ConfGen execution

All commands run from `/root/autodl-tmp/Hulu-Med/MedUniEval`. The scripts cover Hulu-Med and LLaVA-Med on the four MedHEval close-ended tasks and two open-ended tasks. They are resumable: a cache row is skipped only when its protocol fingerprint matches.

## 1. Small validation

```bash
cd /root/autodl-tmp/Hulu-Med/MedUniEval
bash corrected_sgta/run_optimized_all.sh smoke
```

The CE smoke run uses 128 rows per task. The OE smoke run uses 16 rows and four candidates. These runs validate loading, image resolution, cache fingerprints, and analysis; they are not statistically meaningful paper results.

## 2. Full experiments

```bash
cd /root/autodl-tmp/Hulu-Med/MedUniEval
bash corrected_sgta/run_optimized_all.sh full
```

To use several GPUs, give a comma-separated list. One persistent worker is assigned to each GPU, and independent model/dataset jobs are distributed round-robin:

```bash
GPU_IDS=0,1 bash corrected_sgta/run_optimized_all.sh full
```

For a single high-memory GPU:

```bash
GPU_IDS=0 MAX_IMAGE_SIDE=384 CANDIDATE_BATCH=4 \
  bash corrected_sgta/run_optimized_all.sh full
```

For paper tables, repeat the label-free split analysis without repeating CE inference:

```bash
ANALYSIS_SEEDS="42 43 44 45 46" GPU_IDS=0,1 \
  bash corrected_sgta/run_optimized_ce.sh full
```

OE accepts the same `ANALYSIS_SEEDS`. To repeat stochastic generation itself, use a new `OUT_DIR` for every `GENERATION_SEED`; incompatible seeds are intentionally rejected by the cache fingerprint.

Do not launch two runs with the same output directory and different options. The metadata fingerprint will reject incompatible reuse. To override output locations, set `OUT_DIR` separately when invoking the CE or OE script.

## 3. Useful overrides

```bash
# Faster CE pilot with one low-frequency radius
FEDDG_L_VALUES="0.003" MAX_SAMPLES=256 \
  bash corrected_sgta/run_optimized_ce.sh smoke

# More expensive OE candidate budget
CANDIDATES=12 CANDIDATE_BATCH=4 \
  bash corrected_sgta/run_optimized_oe.sh full
```

The default CE style bank contains the original image, four medical domain centers at two low-frequency radii, and two gamma views. All CE views are sent to the model together. The OE full style stream uses the original image, all four domain centers, and gamma views with the same total candidate budget as the original-image ConfGen baseline.

## 4. Result files

CE output is under `corrected_runs/optimized_ce_{smoke,full}_v53`:

- `*.summary.seed<seed>.json`: baseline, FedDG, TTA, fixed SGTA, LAME/LATA, LAC/APS. Use `point_accuracy_test_only` in paper tables.
- `*.sgta_optimized.seed<seed>.json`: calibration-CV-selected domain-center SGTA. Use `test_results.domain_calibrated_sgta` and `test_delta_vs_baseline`.

OE output is under `corrected_runs/optimized_oe_{smoke,full}_v53`:

- `*.confgen_optimized.seed<seed>.json`: original ConfGen, equal-budget style-augmented ConfGen, and domain-calibrated SGTA-ConfGen.
- Use the `sgta_confgen_domain_calibrated` method for the proposed method and report coverage, coverage gap, set size, empty-set rate, and reduced-output metrics together.

Aggregate repeated analysis seeds into means and sample standard deviations:

```bash
python -m corrected_sgta.summarize_optimized \
  --ce-dir corrected_runs/optimized_ce_full_v53 \
  --oe-dir corrected_runs/optimized_oe_full_v53 \
  --output corrected_runs/optimized_v53_aggregate.json
```

Historical `full_v52` and `oe_full_v52` results remain valid for their recorded configurations and can be reused as baselines. They must not be relabeled as multi-domain-center or domain-calibrated results.

## 5. Methodological guardrails

The optimized CE selector learns style reliability and graph settings only within the outer calibration split using internal cross-validation; test labels are evaluation-only. SGTA-ConfGen further separates style-reliability training, beta validation, proper conformal calibration, and test evaluation. ROUGE-L admissibility is a reproducible lexical proxy, not a clinical hallucination judge; an AAAI submission should additionally report the official MedHEval judge or a preregistered clinical evaluator.

# L1 — Sparse Lesion Boundary: small evidence is systematically diluted

## Natural question

Before designing a patch method, test the phenomenon itself:

> Among chest-X-ray findings that all three radiologists independently marked
> present, does a smaller annotated lesion extent receive a lower VLM
> `supported − refuted` score?

Only `3/3` positives are used, so this is not ordinary label disagreement.  The
area is the union of released radiologist boxes divided by image area.  For
each finding separately, rank correlation is computed between log area and the
final claim margin; correlations are macro-averaged so aortic enlargement and
nodule/mass cannot be pooled into a spurious disease-size trend.

The gate requires macro within-finding Spearman at least `0.20`, a stratified
bootstrap lower bound above zero, within-finding permutation `p≤0.05`, and at
least two thirds of findings in the positive direction.

## Results

| Split | Model | n | Macro within-finding Spearman | 95% stratified bootstrap CI | Positive findings | Miss rate (`margin≤0`) | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| Development | Huatuo | 480 | 0.232 | [0.143, 0.314] | 6/8 | 19.2% | PASS |
| Development | Hulu | 480 | 0.475 | [0.390, 0.544] | 8/8 | 34.6% | PASS |
| Fresh confirmation | Huatuo | 133 | 0.323 | [0.158, 0.462] | 5/7 | 27.8% | PASS |
| Fresh confirmation | Hulu | 133 | 0.415 | [0.222, 0.559] | 7/7 | 38.3% | PASS |

All four within-finding permutation tests are significant (`p≤0.0006`).  In
the fresh split, median annotated area ranges from `0.34%` for nodule/mass to
`8.39%` for cardiomegaly, but the test does not compare these medians against
one another; it asks whether size ranks margin *within the same finding*.

A stricter cross-split miss-prediction supplement is mixed.  A development
logistic model with finding identity alone versus finding plus log area gives:

| Model | Finding-only miss AUROC | + log area | Delta (95% CI) | NLL: base → area |
|---|---:|---:|---:|---:|
| Huatuo | 0.751 | 0.835 | +0.083 [0.035, 0.134] | 0.536 → 0.477 |
| Hulu | 0.841 | 0.855 | +0.015 [-0.063, 0.089] | 0.605 → 0.509 |

Both area coefficients have the expected negative sign and both NLL
improvement intervals exclude zero, but Hulu does not pass the frozen `+0.05`
incremental-AUROC gate.  Thus lesion extent is a replicated continuous
calibration factor, not a universal stand-alone error detector.

## What is and is not established

Established as an exploratory replicated phenomenon:

- even when readers unanimously see a finding, its spatial extent strongly
  predicts how much claim evidence reaches the VLM answer;
- the effect appears in two different medical VLMs and two image-disjoint
  artifacts;
- the result gives a natural reason why global pooling and whole-image decoding
  can miss small findings.

Not established:

- a patch statistic has not yet outperformed the final margin;
- box area is an imperfect proxy for conspicuity and does not measure contrast;
- the confirmation endpoint was defined after the artifact had already been
  opened for prior research, so this is not a new prospective confirmation;
- the result currently concerns omission/false negatives more directly than
  fabricated positive claims.

## Authorized next gate

Fit a simple finding direction on development global visual features, project
every visual patch onto it, and compare whole-image mean with a penalized
multiscale spatial scan on the fixed fresh panel.  Advance only if the scan adds
at least `0.02` macro AUROC beyond final margin, improves NLL with paired CI
excluding zero, and is positive on at least five of seven findings in both
models.  This gate distinguishes a real sparse-evidence mechanism from merely
renaming the familiar small-object problem.

Artifacts:

- `corrected_runs/daylong_idea_search_v1/sparse_lesion_boundary_development_v1.json`
- `corrected_runs/daylong_idea_search_v1/sparse_lesion_boundary_v2.json`
- `corrected_runs/daylong_idea_search_v1/lesion_area_miss_prediction_v1.json`

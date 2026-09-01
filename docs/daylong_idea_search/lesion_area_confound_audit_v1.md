# Lesion-area confound audit v1

## Question

The earlier VinDr result linked smaller released box extent to lower final
`supported - refuted` margins. This CPU-only audit asks whether that association
survives measured alternatives: radial/x/y position, local absolute intensity
contrast, local texture ratio, box count, fragmentation, and inter-reader mask
and centroid disagreement. Box extent is an annotation geometry proxy, **not**
clinical conspicuity ground truth.

## Frozen design

- Dataset: VinDr-CXR train, only 3/3 reader-positive claims with R8/R9/R10
  released boxes.
- Models: HuatuoGPT-Vision-7B and Hulu-Med-4B.
- Development: 420 claims/model; fresh confirmation: 133 claims/model; image
  overlap: 0.
- Models and regularization were selected only by image-grouped development CV.
- Confirmation inference used finding fixed effects and all confounds above,
  then tested the incremental value of log box-union area.
- Uncertainty: confirmation-image bootstrap, 5,000 draws, seed 20260812.

## Exact confirmation results

| Model | adjusted rank rho [95% CI] | area coefficient: margin / miss | miss AUROC delta [95% CI] | NLL improvement [95% CI] | margin-MSE improvement [95% CI] |
|---|---:|---:|---:|---:|---:|
| Huatuo | 0.239 [0.036, 0.419] | +0.232 / -0.352 | +0.0245 [0.0039, 0.0481] | +0.0190 [-0.0005, 0.0392] | +0.0836 [0.0345, 0.1326] |
| Hulu | 0.420 [0.218, 0.559] | +0.633 / -1.295 | +0.0158 [-0.0161, 0.0514] | +0.0609 [0.0100, 0.1128] | +0.2955 [0.1182, 0.4735] |

Positive margin coefficients and negative miss coefficients have the expected
direction: larger annotated extent predicts a more positive margin and fewer
margin-defined misses. The continuous-margin result survives in both models;
binary miss AUROC is small and its CI crosses zero for Hulu, while Huatuo is
positive but below a strong +0.05 effect threshold.

## Decision

Do **not** withdraw the narrow natural phenomenon: after these measured proxies,
box extent retains incremental association with the continuous clinical-claim
margin in both models on image-disjoint confirmation data. However, weaken any
claim that area is a universal error detector or lesion conspicuity measure.
This result justifies testing a sparse-evidence/search mechanism; it does not by
itself establish that mechanism, localization, causality, or hallucination
mitigation.

## Provenance

- Script: `anchor/corrected_sgta/audit_lesion_area_confounds_v1.py`
- Result: `corrected_runs/daylong_idea_search_v1/lesion_area_confound_audit_v1.json`
- Method: finding-fixed, regularized multivariable development-to-confirmation
  audit; no GPU and no baseline-process changes.
- Full command, input hashes, source hash, seed and bootstrap count are stored in
  the JSON artifact.

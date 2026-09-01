# Paired DICOM-render pilot decision

Artifact: `corrected_runs/vindr_v2/dicom_render_huatuo_pilot_v1/analysis_v1.json`

SHA-256: `6070cc9454aa4f96f0987600da3dfc59a73314cf667862f53c2500a7b6089aaa`

## Decision

**Stop this branch before Hulu or held-out replication.** The exploratory progression gate passed for 0/4 frozen findings (required 3/4). It does not support a common display-style direction, a training-source-domain center, or a mitigation method.

## What the experiment does support

Clinically audited continuous DICOM rendering changes can move Huatuo's claim polarity on individual cases. The descriptive median render-orbit diameter was approximately 0.70 reader steps for aortic enlargement, 0.80 for cardiomegaly, and 0.93 for pleural effusion. Prediction-flip rates were 5.0%, 2.5%, and 10.0%, respectively. Pulmonary fibrosis had a 17.5% flip rate, but its estimated reader-step slope had a confidence interval crossing zero, so its nominal 4.57 reader-step orbit is not a valid calibrated magnitude.

This is **heterogeneous render sensitivity**, not a coherent center.

## Why the center hypothesis failed

For each finding, half A selected the clinically eligible transform with the largest absolute signed median shift. On untouched half B:

| Finding | Selected transform | Held-out signed reader-equivalent effect (95% CI) | Sign agreement | Formal finding gate |
|---|---|---:|---:|---:|
| Aortic enlargement | center + 0.05 width | -0.321 [-0.810, 0.023] | 72.2% | fail |
| Cardiomegaly | center + 0.05 width | 0.015 [-0.604, 0.198] | 45.0% | fail |
| Pleural effusion | width × 1.25 | -0.020 [-0.427, 0.596] | 55.6% | fail |
| Pulmonary fibrosis | center - 0.05 width | 0.126 [-2.065, 2.266] | 47.4% | fail |

All selected-transform intervals cross zero. The exact lossless-duplicate control passed, so the failure is not numerical noise; instead, the direction of the response varies by image. Same-reader-support image swaps were also larger than the median clinical render orbit (orbit/swap ratios 0.41–0.67 for the three calibrated findings), which bounds the practical scale of display variation.

## Research interpretation

The original observation—different image styles can produce different answers—was real but underidentified. It did not distinguish:

1. a stable model preference for one source/display center;
2. case-specific threshold crossings under weak evidence;
3. generic sensitivity to any input perturbation.

The paired DICOM experiment rejects explanation 1 as a cross-finding mechanism. It is compatible with explanation 2, but that question is already tested more directly by the formal reader-unanimity layer-boundary experiment. No second-model compute is justified from this pilot.

The result remains an exploratory negative result and cannot enter the main paper as confirmation. It may appear in an appendix as an example of why style disagreement alone does not identify a source-domain center.

# Domain-Orbit Canonicalization fatal audit (2026-08-03)

## Frozen question

Do style/render residuals in HuatuoGPT-Vision projector-token space form a
domain nuisance subspace that can be removed without removing reader-supported
pathology evidence?

This is a mechanism feasibility audit, not a mitigation efficacy experiment.
No rank or coefficient was selected from the labels.

## Correction to the proposed method

For residual matrix `R[(views*tokens), width] = U S W^T`, the feature-space
basis is the right singular basis `W`, not `U`. The tested intervention is

`V_doc = V0 - alpha * (V0 - Vmean) W_r W_r^T`.

Full-rank removal reduces exactly to the orbit mean, so orbit mean and
equal-displacement interpolation are mandatory controls. A random subspace is
also rescaled to have exactly the same Frobenius displacement as DOC.

## Data and setting

- Model: HuatuoGPT-Vision-7B, frozen.
- Data: VinDr-CXR pilot split, 16 balanced claims.
- Findings: aortic enlargement, cardiomegaly, pleural effusion, pulmonary
  fibrosis.
- Reader support: one claim from each `finding x {0/3,1/3,2/3,3/3}` stratum.
- Representation: 576 projector tokens of width 3584, immediately before the
  language model.
- Ranks: 1, 2, 4, 8; alpha: 0.5 and 1.0.
- Primary report below: alpha 1.0. The pilot is descriptive and not powered for
  a final efficacy claim.

Two independent transformation instruments were tested:

1. **DICOM render orbit**: four fitting views and two held-out window/render
   views.
2. **Source-radial spectrum orbit**: a continuous path with weights
   `0.0,0.1,0.2,0.3` for fitting and `0.4,0.5` held out, using the audited
   PubMedVision-CXR source radial spectrum bank built from 5,846 images.

## Results

| Orbit | Mean cumulative variance, rank 1 / 2 | Mean held-out attenuation, rank 2 / 4 / 8 | DOC rescues (all ranks) | Three-state accuracy, original / DOC | Clear-case aligned margin delta, ranks 1 / 2 / 4 / 8 |
|---|---:|---:|---:|---:|---:|
| DICOM render | 70.5% / 89.3% | 85.8% / 88.4% / 90.4% | 0 | 25.0% / 25.0% | -0.002 / -0.075 / -0.100 / -0.118 |
| Source radial | 69.0% / 89.1% | 87.6% / 89.8% / 91.8% | 0 | 25.0% / 25.0% | -0.012 / -0.073 / -0.092 / -0.069 |

The low three-state accuracy is mainly because Huatuo never selected the
`undetermined` verbalizer for the eight 1/3 or 2/3 cases. The negative result
does not depend on that issue: among only clear 0/3 and 3/3 cases, original and
DOC accuracy were both 50%, and the label-aligned continuous margin did not
improve.

One equal-displacement orbit-mean control rescued one clear case in the DICOM
orbit, while DOC rescued none. This isolated pilot flip is not evidence for the
mean method, but it rules out claiming that the learned subspace is superior to
the simple control.

## Gates and decision

| Gate | Render | Source radial |
|---|---:|---:|
| Orbit is low dimensional | PASS | PASS |
| Fitted tangent attenuates held-out styles | PASS | PASS |
| Any DOC label rescue | FAIL | FAIL |
| DOC beats both equal-displacement controls | FAIL | FAIL |

**Decision: stop scaling DOC as a hallucination mitigation method.** The
experiment establishes that a style tangent exists, but not that the tangent is
a safely removable nuisance or a direction toward clinical correctness.

## Token-stability gating follow-up

VinDr reader boxes were mapped to the 24x24 visual-token grid to test the
attachment's simpler token-gating proposal. Among 12 claims with boxes, the
median instability ratio inside versus outside a reader box was 0.111; among
the four unanimous-positive claims it was 0.270. Most render instability was
therefore outside the annotated finding, although one pleural-effusion case was
a clear counterexample (ratio 1.81).

This justified one outcome-blind screening test: attenuate the most unstable
25% of tokens, restore the original global activation norm, and compare with
the same number of random tokens and the least-unstable tokens.

| Method | Three-state accuracy | Clear-case accuracy | Rescue / harm | Clear label-aligned margin delta |
|---|---:|---:|---:|---:|
| Original | 25.0% | 50.0% | 0 / 0 | 0.000 |
| Unstable 25%, half attenuation | 25.0% | 50.0% | 0 / 0 | -0.043 |
| Unstable 25%, full attenuation | 25.0% | 50.0% | 0 / 0 | -0.378 |
| Random 25%, full attenuation | 31.2% | 62.5% | 1 / 0 | -0.123 |
| Least-unstable 25%, full attenuation | 25.0% | 50.0% | 1 / 1 | -0.224 |

The stability-selected gate caused no prediction flip or rescue and moved the
continuous margin in the wrong direction. The isolated random-mask rescue
prevents attributing benefit to the stability rule. Token-stability gating is
therefore also stopped at the pilot stage.

This result agrees with the earlier 128-case frozen audit: mild-style drift was
a worse error detector than the original decision margin (style-drift AUROC
0.425--0.446 versus margin-error AUROC 0.798), and all six unique flips occurred
in the low-margin half. The parsimonious explanation is ordinary decision-boundary
susceptibility, not a hallucination-specific style channel.

## Novelty collision audit

The exact medical, single-image, projector-token experiment was not found, but
its components are strongly occupied:

- VISTA (ICML 2025) uses PCA over style hidden-state differences and activation
  intervention for VLM hallucination.
- Robustifying Zero-Shot VLMs by Subspaces Alignment (ICCV 2025) represents an
  image and its augmentations as a subspace.
- SubTTA (arXiv 2026) aligns principal visual/text subspaces at test time.
- FOCAL (ICML 2025) generates a transformation set per test image and selects a
  canonical view.
- VACoDe/VSCoDe uses multiple augmentations for contrastive hallucination
  decoding.
- OptTTA (MIDL 2022), ISR (ICML 2022), and LRDG (NeurIPS 2022) already cover
  source-statistic adaptation and invariant/domain-specific subspace recovery
  under different assumptions.

Therefore `multi-view + SVD + nuisance removal` cannot itself carry an ICLR
contribution. Only a new causal mechanism showing which transformation response
contains clinical evidence, and why, could revive this family.

## Reproducible artifacts

- Implementation: `anchor/corrected_sgta/domain_orbit_diagnostic.py`
- Runner: `anchor/corrected_sgta/run_huatuo_domain_orbit_diagnostic_v1.py`
- Analyzer: `anchor/corrected_sgta/analyze_domain_orbit_diagnostic_v1.py`
- Unit tests: `tests/test_domain_orbit_diagnostic.py` (3 passed)
- Render result: `corrected_runs/vindr_v2/domain_orbit_huatuo_pilot16_v2/result.json`
- Render analysis: `corrected_runs/vindr_v2/domain_orbit_huatuo_pilot16_v2/analysis_v2.json`
- Source-radial result: `corrected_runs/vindr_v2/domain_orbit_huatuo_source_radial_pilot16_v1/result.json`
- Source-radial analysis: `corrected_runs/vindr_v2/domain_orbit_huatuo_source_radial_pilot16_v1/analysis.json`
- BBox localization: `corrected_runs/vindr_v2/domain_orbit_bbox_localization_pilot16_v1/render_result.json`
- Token gate result: `corrected_runs/vindr_v2/domain_stability_token_gate_huatuo_pilot16_v1/result.json`

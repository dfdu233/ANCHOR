# Style Hypothesis Audit and Mechanism Pivot

Decision date: 2026-07-30

## Executive decision

The broad claim that medically irrelevant image style changes generally switch
a medical VLM between source-domain priors is **not supported** in the current
HuatuoGPT-Vision / RULE-MIMIC binary CXR setting.  It must not motivate a
style-matched decoder or headline a paper.

The more credible next question is narrower and mechanism-first:

> Does medical-VLM cross-view instability contain hallucination-specific
> information, or is it mostly ordinary decision-boundary susceptibility
> already captured by the unperturbed logit margin?

This is a research lead, not yet a claim.  The local evidence supports running
the decisive experiment, but is restricted to one model, a binary task, and
three mild transforms.

## Evaluation freeze

All new comparisons follow `anchor-eval-contract-v1` in
`docs/UNIFIED_EVALUATION_CONTRACT.md`.

- The present experiments use CE-D: FP32 `Yes` versus `No` verbalizer logits
  at the first answer position.
- The manifest contains 128 balanced questions, one question per patient.
- Transform guards require PSNR at least 18 and edge correlation at least
  0.90.  All three transform banks passed for all samples.
- The preregistered phenomenon criterion was a decision-flip rate of at least
  5% for at least two of three styles.
- A decision change is not called a hallucination, and pixel-level guards are
  not treated as proof of clinical semantic preservation.
- Historical results produced with reconstructed lexical parsing, unequal
  token budgets, or incompatible prompts remain evidence grade C until
  rescored or rerun.

## What the controlled probes showed

### 1. Additive source-style prior: not supported

The proposed signature was agreement between the full-image style-induced
Yes-minus-No margin drift and a style-only null drift.  On 16 samples with
eight null replicates:

- Spearman rho: 0.276;
- one-sided permutation p: 0.152;
- null split-half rho: 0.009, p = 0.974.

The null measurement was not reliable, so this experiment is inconclusive
rather than positive.  The result is stored under
`corrected_runs/style_prior_probe/huatuo_rule_mimic_n16_v3_mc8`.

### 2. Evidence-gain modulation: apparent signal was an artefact

The naive statistic correlated `(original - zero-view)` with
`(styled - original)`.  It produced rho = -0.538, p = 0.031, but reused the
original margin with opposite signs and was confounded by the Yes/No label.
After symmetric support construction and within-label checks:

- label-residual rho: 0.038, p = 0.888;
- within-label permutation p = 0.554;
- leave-one-out rho ranged from -0.168 to 0.143.

The attractive initial result is rejected as a mathematically coupled false
lead.

### 3. Material style sensitivity: preregistered criterion failed

| View | Accuracy | Flip rate | 90% Wilson CI | Rescue / harm | Mean absolute margin drift |
|---|---:|---:|---:|---:|---:|
| Original | 0.766 | — | — | — | — |
| VQA-RAD low-frequency style | 0.750 | 3.13% | [1.41%, 6.78%] | 1 / 3 | 0.094 |
| Gamma 0.9 | 0.758 | 2.34% | [0.94%, 5.72%] | 1 / 2 | 0.050 |
| Gamma 1.1 | 0.750 | 1.56% | [0.52%, 4.61%] | 0 / 2 | 0.044 |

No style reached 5%, so the registered status is
`phenomenon_not_confirmed`.  The complete result is in
`corrected_runs/style_phenomenon/huatuo_rule_mimic_n128_v1/summary.json`.

## Post-hoc mechanism lead: boundary susceptibility

For an original binary logit margin \(m(x)\) and transform-induced change
\(\delta_T(x)\), a decision flip requires only that the perturbation cross the
existing boundary.  It does not require a hallucination-specific mechanism:

\[
\operatorname{sign}(m(x)) \ne \operatorname{sign}(m(x)+\delta_T(x)).
\]

The fingerprinted post-hoc analysis found:

| Error-risk score | AUROC | Bootstrap 90% CI |
|---|---:|---:|
| Negative absolute original margin | 0.798 | [0.727, 0.866] |
| Mean absolute style drift | 0.425 | [0.329, 0.522] |
| Maximum absolute style drift | 0.446 | [0.346, 0.546] |
| Mean drift / (absolute margin + 0.1) | 0.711 | [0.621, 0.793] |

The relative-instability score was still worse than margin by 0.087 AUROC;
its paired-bootstrap 90% interval for the difference was
[-0.167, -0.011].  Across the three styles, all six unique decision flips were
in the low-margin half and none in the high-margin half.  Within the low-margin
half, mean style drift remained non-predictive of error (AUROC 0.435).

This result does **not** prove that every stability method reduces to
confidence.  It shows only that raw style-view instability has no demonstrated
incremental value in this local setting and that margin is the simpler current
explanation.

Reproduction:

```bash
PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
  anchor/corrected_sgta/analyze_style_boundary.py \
  --input corrected_runs/style_phenomenon/huatuo_rule_mimic_n128_v1/raw.jsonl \
  --output corrected_runs/style_phenomenon/huatuo_rule_mimic_n128_v1/boundary_analysis.json
```

## Collision map

The next question is crowded but not yet duplicated under the retrieved
keywords.

- **LENS** (anonymous ACL submission, 2026) is the closest image-side work. It
  treats a cross-view stability gap as a mitigation signal and uses
  attribution-guided counterfactual views. The proposed delta is to test
  whether that signal adds information after conditioning on the original
  margin and a perturbation-matched null, not to propose another view bank.
  <https://openreview.net/pdf?id=oh3c2ieVab>
- **PSF-Med** (Sadanandan and Behzadan, arXiv 2026) measures medical-VLM
  paraphrase flips, traces a prompt-framing feature that shifts the decision
  margin, and warns that low flip rates need not mean visual grounding. It
  studies text paraphrases, not the conditional specificity of image-view
  instability. <https://arxiv.org/abs/2602.21428>
- **Reference-free Hallucination Detection for Large Vision-Language Models**
  (Li et al., Findings of EMNLP 2024) compares uncertainty- and
  consistency-based detectors, but does not make margin-conditioned
  counterfactual specificity the causal object.
  <https://aclanthology.org/2024.findings-emnlp.262/>
- **VES-RFT** (Hou et al., CVPR 2026) rewards image-attributable entropy
  change between image and no-image passes. It optimizes evidence sensitivity;
  it does not ask whether sensitivity is incremental to ordinary boundary
  geometry.
  <https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VES-RFT_Rewarding_Visual_Evidence_Sensitivity_to_Mitigate_Hallucinations_in_Large_CVPR_2026_paper.html>
- **Your other Left! / MIRP** (Wolf et al., MICCAI 2025) is the methodological
  model for beginning with a sharply isolated medical failure and controls,
  rather than attaching a generic mitigation method.
  <https://papers.miccai.org/miccai-2025/1027-Paper0530.html>

No directly overlapping work was retrieved for the exact combination:
token-level medical hallucination, image-view stability, original-margin
conditioning, and perturbation-matched null residualization.  This is a
retrieval result, not proof of novelty.

## Frozen next idea: Stability Needs a Null

### Problem freeze

Medical hallucination work often interprets cross-view disagreement as weak
grounding.  The hidden assumption is that disagreement is specific to missing
visual evidence rather than a generic consequence of a small decision margin.

### Mechanism freeze

Decompose observed view drift into:

\[
d_T = b(|m|, q, t, \kappa_T) + e_T,
\]

where \(m\) is the original decision margin, \(q\) the question or clinical
concept, \(t\) the generation step, \(\kappa_T\) the calibrated perturbation
strength, \(b\) ordinary boundary susceptibility, and \(e_T\) excess
view-specific evidence sensitivity.

The candidate statistic is the cross-fitted residual:

\[
\operatorname{EVS}_T =
|d_T| -
\widehat{\mathbb E}\!\left[|d_N|
\mid |m|,q,t,\kappa_N=\kappa_T\right],
\]

where \(N\) is a clinically irrelevant, response-strength-matched null family.
Only positive excess sensitivity is eligible to gate a contrastive penalty.

### Opposing predictions

- **Boundary-only explanation:** after conditioning on margin and null
  sensitivity, EVS adds no held-out hallucination information; flips
  concentrate near the boundary regardless of whether the view targets
  clinical evidence.
- **Grounding-specific explanation:** EVS retains incremental predictive value
  within margin strata, has a clinically correct direction under
  evidence-targeted removal, and transfers across models or modalities.

### Decisive minimum experiment

1. Use token-level supported/unsupported clinical findings in report
   generation; keep binary VQA only as a diagnostic.
2. Freeze three view roles before test evaluation: clinically
   nuisance-preserving views, pathology-targeted evidence removal, and
   response-strength-matched nulls.
3. Fit the null response curve on a calibration split only.
4. On a locked patient-level test split compare original margin, raw
   stability, relative stability, and EVS with paired cluster bootstrap and
   within-margin-stratum analysis.
5. Repeat on at least two model families and two datasets; include null-image
   and shuffled-image grounding controls.

The preregistered continue criterion should be incremental EVS AUROC of at
least 0.03 over margin, with a positive 90% paired-bootstrap lower bound, in at
least two of three primary model-task pairs.  If this fails, reject EVS and
retain the negative mechanism result.  Do not tune the threshold after seeing
the locked test set.

### Method only after mechanism validation

If EVS survives, the elegant mitigation is
**margin-conditioned stability residual decoding**: apply the usual
counterfactual logit penalty only to tokens whose view drift exceeds their
margin-matched null expectation.  This is analogous in spirit to an adaptive
mask, but the mask is defined by specificity rather than entropy alone.

If EVS fails, the practical conclusion is simpler: use calibrated
margin-based selective prediction or abstention instead of paying for
multi-view decoding.

## Reviewer-style idea verdict

### Original style-prior decoder

- Paper type: Novel Method.
- Fatal flaw: **CRITICAL, data-refuted premise.** Its required phenomenon did
  not meet the preregistered criterion, the additive-prior control was
  unreliable, and the apparent evidence-gain correlation vanished under a
  valid analysis.
- Verdict: **Reject and Pivot.**

### Stability Needs a Null

- Paper type: Novel Problem / mechanism audit.
- One-sentence story: before using cross-view instability to suppress medical
  tokens, subtract the instability any low-margin decision would exhibit and
  ask whether anything hallucination-specific remains.
- Fatal flaw F1, **MAJOR:** LENS and PSF-Med make a broad “stability versus
  confidence” story too crowded. The defense is the exact conditional,
  matched-null, token-level medical test above.
- Fatal flaw F6, **MAJOR:** current binary VQA errors do not verify a claim
  about free-form clinical hallucinations. The defense is a locked
  token-level report experiment with clinical evidence controls.

| Dimension | Score | Ground |
|---|---:|---|
| Higher | 5 | No mitigation gain is established; this stays neutral until EVS validates. |
| Faster | 7 | Mechanism-based: a one-pass margin may replace multi-view inference if EVS fails; runtime is unmeasured. |
| Stronger | 8 | Mechanism-based: explicitly separates grounding specificity from generic perturbation sensitivity. |
| Cheaper | 7 | Mechanism-based: can remove unnecessary views or restrict them to an adaptive subset. |
| Broader | 7 | The conditional test can cover image views, prompt paraphrases, and sampling consistency, but only medical image views are in scope now. |

Paradigm-shift probe: First Principles = yes; Elephant in the Room = yes;
Technology Cycle = no; Hamming's Rule = partial.  The disruptive seed is the
challenge to the field's untested interpretation of stability, not a new
decoder.

Capability fit remains yellow because weekly time and full infrastructure were
not declared.  The local substrate has already run a 7B white-box model and
128-sample paired study, so the pilot is technically feasible; multi-model
token-level report adjudication is the main data and compute risk.

Verdict: **Accept with Revisions — worth pursuing only through the decisive
validation experiment.**  Do not yet position it as a mitigation paper.

## Execution handoff

- Canonical repository: `/home/dbw/ANCHOR`
- Model exercised: HuatuoGPT-Vision-7B
- Current task/data: RULE-MIMIC binary CXR, 128 unique patients
- Evaluation contract: `docs/UNIFIED_EVALUATION_CONTRACT.md`
- Phenomenon runner:
  `anchor/corrected_sgta/run_huatuo_style_phenomenon_confirm.py`
- Prior mechanism probe:
  `anchor/corrected_sgta/run_huatuo_style_prior_probe.py`
- Boundary analyser:
  `anchor/corrected_sgta/analyze_style_boundary.py`
- Claim-grade status: no positive mechanism claim; negative phenomenon result
  is protocol-grade for this narrow setting, while the boundary analysis is
  explicitly post-hoc lead-only.


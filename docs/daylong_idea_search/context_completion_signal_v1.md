# Context Completion Signal v1 — strict post-hoc audit

## Decision

**NO-GO. Do not spend a fresh holdout or a second model on this candidate.**

The paired response to real context is label-associated by itself, but adds no
case-level information beyond a strong `full + crop` base. The apparent signal
is therefore redundant, not a new mitigation handle.

This is especially important because the Huatuo confirmation panel and its
render summaries had already been inspected before this candidate was frozen.
Every number below is exploratory/hypothesis-generating, never confirmatory.

## Frozen candidate and why it is a useful control

For the neutral question prompt, define

\[
\Delta_{ctx}(x,c)=
 m(x_{zoom+true\ context},c)-m(x_{zoom+phase\ sham},c),
\]

where `m` is the model's Yes-minus-No logit margin. Both images contain the
same enlarged ROI in the same panel layout. The only intended difference is a
small panel containing the original full radiograph versus a deterministic
phase-scrambled version with approximately matched spectrum and exactly
rank-matched intensity histogram.

This asks a narrow question: does restoring genuine global anatomy change the
claim margin differently in truly positive and negative cases? It does **not**
by itself prove localization, causal clinical evidence, or hallucination
mitigation.

## Data and evaluation

- Model: HuatuoGPT-Vision-7B.
- Images: 124 VinDr cases, balanced 62 negative / 62 reader-unanimous positive.
- Findings: seven; positive and negative counts matched within finding.
- Strong base: cross-fitted L2-logistic calibration of full-image and sham-crop
  margins plus finding identity.
- Enhanced model: exactly the same base plus `Delta_ctx`.
- Fitting: 5-fold finding-and-label-stratified cross-fitting, repeated 20 times.
- Uncertainty: 5,000 stratified image-bootstrap draws.
- Frozen routing gate required standalone `Delta_ctx` AUROC >= .60 with lower
  CI > .50, at least +.01 cross-fit AUROC with lower CI > 0, NLL improvement
  with lower CI > 0, and majority-finding consistency.

## Results

| Quantity | Result |
|---|---:|
| `Delta_ctx` AUROC | **0.6574** [0.5797, 0.7340] |
| `Delta_ctx` mean, negative | -0.1472 |
| `Delta_ctx` mean, positive | +0.0444 |
| Positive-minus-negative mean | +0.1915 [0.0907, 0.2944] |
| Strong base AUROC / NLL | 0.8548 / 0.4887 |
| Base + `Delta_ctx` AUROC / NLL | 0.8431 / 0.5021 |
| Incremental AUROC | **-0.0117** [-0.0330, +0.0044] |
| NLL improvement (positive is good) | **-0.0134** [-0.0348, +0.0038] |
| Brier improvement (positive is good) | **-0.0044** [-0.0129, +0.0021] |
| True-panel minus sham-crop raw AUROC | +0.0127 [-0.0203, +0.0418] |
| Sham-crop minus true-panel raw NLL | +0.0279 [+0.0035, +0.0521] |

Interpretation:

1. Genuine context produces a paired label-associated response: negatives are
   suppressed more than positives. This is real on this inspected panel.
2. The response is already predicted by the full-image and crop margins. Once
   they are admitted, `Delta_ctx` decreases AUROC and worsens NLL/Brier.
3. The true-context panel's significant raw NLL improvement but nonsignificant
   AUROC change is consistent with a calibration/operating-point shift, not a
   new ordering of individual cases.

## Finding consistency and sensitivity

Standalone `Delta_ctx` AUROC was above 0.5 in 7/7 findings, but adding it to
the strong base improved AUROC in only 2/7 and NLL in only 2/7. Several finding
cells are small (12–32 total), so standalone per-finding values are descriptive.

The incremental conclusion was insensitive to L2 penalty:

| Ridge | AUROC gain | NLL improvement |
|---:|---:|---:|
| 0.01 | -0.0075 | -0.0191 |
| 0.1 | -0.0060 | -0.0169 |
| 1 | -0.0117 | -0.0134 |
| 10 | -0.0034 | -0.0040 |
| 100 | +0.0021 | +0.0049 |

No setting had a positive bootstrap lower bound for AUROC or NLL gain. These
settings are a robustness audit, not test-set hyperparameter selection.

## Collision audit

The residual candidate is too close to an established family to justify a
weak fresh test:

- [Fine-Grained Visual Prompting (NeurIPS 2023)](https://papers.neurips.cc/paper_files/paper/2023/hash/4e9fa6e716940a7cfc60c46e6f702f52-Abstract-Conference.html)
  shows that retaining the target while blurring its surroundings preserves
  spatial coherence and improves referring-expression recognition. Our
  `native_context_removed` render is already conceptually adjacent.
- [HALC (ICML 2024)](https://arxiv.org/abs/2403.00425)
  explicitly combines local focal views and global context through adaptive
  focal-contrast decoding.
- [AGLA (CVPR 2025)](https://arxiv.org/abs/2406.12718) assembles global
  generative and prompt-relevant local discriminative features in calibrated
  decoding.
- [LENS](https://openreview.net/pdf?id=oh3c2ieVab) uses cross-view logit
  stability under semantics-preserving medical counterfactuals for training
  and inference-time hallucination mitigation.

Thus `true context - sham context` would need a strong, reproducible
incremental clinical signal to clear the collision. It has the opposite result
here. The finding does retain diagnostic value: real context can change
calibration without adding case discrimination, reinforcing the project's
broader warning that **response is not evidence**.

## Boundary and next action

Close context-completion as a method candidate. Do not describe the standalone
AUROC as evidence of a recoverable clinical channel. Reopen only if a new,
independently motivated theory predicts information that is absent from both
the full and local margins—not merely another fusion of them.

Auditable outputs:

- Script: `anchor/corrected_sgta/analyze_context_completion_signal_v1.py`
- Result: `corrected_runs/daylong_idea_search_v1/context_completion_signal_huatuo_v1.json`

# Evidence Recoverability first screen: result

Date: 2026-08-03

## Outcome

The sampled answer-position logit-lens version of Evidence Recoverability is
falsified on the preregistered two-model VinDr confirmation screen.  It must not
be used to justify ETD tuning or a causal evidence claim.

## Setup

- Models: HuatuoGPT-Vision-7B and Hulu-Med-4B.
- Development: 640 reader-grounded VinDr claims per model.
- Independent confirmation: 1,920 claims per model, covering eight findings and
  0/3, 1/3, 2/3, 3/3 reader-vote bins.
- FP/FN truth: only 0/3 and 3/3 claims; 1/3 and 2/3 remain ambiguous.
- Layers: Huatuo 7/14/21/28; Hulu 9/18/27/36.
- Margin: diagnostic `supported - refuted` at the answer position.
- Calibration: per-finding/per-layer thresholds fit on development only.
- Null: exchange one complete sampled-layer trajectory with a different
  confirmation record having the same finding and same ground-truth polarity.
- Uncertainty: 5,000 image-cluster bootstrap/randomization draws.

This is an image-disjoint screen. Patient-disjointness cannot be verified from
the available manifest.

## Primary results

| Model | Error | n | raw any-layer | calibrated any-layer | matched null | excess | one-sided p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Huatuo | FP | 277 | 0.0% | 84.5% | 90.6% | -6.1 pp | 1.0000 |
| Huatuo | FN | 92 | 100.0% | 78.3% | 88.4% | -10.1 pp | 0.9996 |
| Hulu | FP | 74 | 0.0% | 74.3% | 94.6% | -20.3 pp | 1.0000 |
| Hulu | FN | 166 | 100.0% | 94.0% | 97.3% | -3.3 pp | 0.9946 |

The raw oracle gives the visually striking but invalid pattern “no FP is
recoverable; every FN is recoverable.”  Its cause is visible directly in the
layer marginals: the first two sampled layers are positive for 100% of claims
in both models.  Therefore every positive-label FN automatically looks correct
early and every negative-label FP automatically looks wrong early.

After dev calibration, an `any layer` oracle remains numerically high, but it
is lower than the truth/finding-matched shuffled baseline in all four cells.
The trajectories belonging to final errors are not unusually recoverable.

## Interpretation boundary

This result falsifies only the current sampled **answer-position logit-lens**
representation. It does not prove that correct clinical signal is absent from
visual tokens, projector features, unsampled layers, or a robust nonlinear
claim probe. It also does not establish a causal boundary for open generation.

The convex-hull statement is retained only for a fixed binary claim margin.
It cannot be generalized to full-vocabulary top-1 generation.

## Literature collision and revised research question

`Evidence Recoverability` alone is not novel enough. Yu et al. already divide
language-model factual errors into knowledge-enrichment and answer-extraction
failures and use causal analysis ([Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.466/)).
DeCo already exploits correct visual objects appearing in preceding VLM layers
([ICLR 2025 code/paper](https://github.com/zjunlp/DeCo)). VISTA also studies
hidden genuine visual information and cross-layer intervention
([official code](https://github.com/LzVv123456/VISTA)).

The defensible next question is narrower and harder:

> Is reader-grounded clinical evidence observable in visual/projector tokens
> but not safely controllable at the answer position?

This creates three non-overlapping operational states:

- `Absent`: no robust visual-token or answer-position evidence.
- `Stranded`: robust visual-token evidence exists but never reaches the answer
  margin.
- `Overridden`: the answer margin is robustly correct for consecutive layers
  and later reverses.

## Decision

Do not run ETD or answer-position causal transport from this screen.  The next
bounded experiment may test the Observability--Controllability Gap using a
dev-trained visual-token claim probe, prompt paraphrases, image swaps, dense
layers, and then matched causal transport.  It advances only if the robust
visual probe generalizes to confirmation and predicts correction/harm beyond
finding and truth-class baselines.

Artifacts:

- `corrected_runs/vindr_v2/evidence_recoverability_confirmation_huatuo_v2/analysis.json`
- `corrected_runs/vindr_v2/evidence_recoverability_confirmation_hulu_v2/analysis.json`
- `docs/EVIDENCE_RECOVERABILITY_PROTOCOL_20260803.md`

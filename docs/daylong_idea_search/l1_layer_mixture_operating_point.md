# L1 — Convex layer mixture: evidence or operating-point shift?

## Question

LET/ETD-like methods mix an intermediate diagnostic margin with the final-layer
margin.  This audit separates two effects:

1. **ranking information** — true positive cases should outrank true negative
   cases more often, measured by AUROC and independent of a decision threshold;
2. **operating point** — a score can simply cross zero more often, making the
   model report more positive claims without improving ranking.

The development artifact selects one intermediate layer and a convex weight
`alpha` from `0, 0.05, ..., 1`.  The already-opened fresh 532-image artifact is
used once for this exploratory confirmation.  Only unanimous `0/3` and `3/3`
reader cases enter the binary audit (`n=266`); seven findings are macro-averaged.

## Results

| Model | Selected layer / alpha | Final AUROC | Mixture AUROC | Delta (95% paired bootstrap CI) | Fixed-zero positive rate: final → mixture | Fixed-zero BAcc: final → mixture | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Huatuo | 21 / 0.75 | 0.7099 | 0.7273 | +0.0174 [0.0044, 0.0309] | 0.598 → 0.861 | 0.624 → 0.519 | FAIL |
| Hulu | 18 / 0.55 | 0.8405 | 0.8520 | +0.0115 [-0.0032, 0.0277] | 0.380 → 0.605 | 0.737 → 0.684 | FAIL |

Frozen gate: AUROC delta at least `0.02`, bootstrap lower bound above zero, and
positive deltas on at least four of seven findings.  Neither model passes; Hulu's
CI also includes zero.  Development-fitted per-finding thresholds yield only
`+0.75 pp` BAcc over final-thresholded Huatuo and `+2.26 pp` on Hulu.

## Decision boundary

- Close **convex answer-position layer mixture as a strong incremental-evidence
  route**.  The audit does not close arbitrary spatial visual-token statistics.
- The Huatuo point is a weak but real ranking signal, so it is inaccurate to say
  every historical gain is *only* threshold shift.
- The very large positive-rate movement and worse fixed-zero BAcc show that
  uncalibrated recall gains cannot be interpreted as recovered clinical evidence.
- This is a post-hoc exploratory audit of already-opened artifacts, not a new
  prospective confirmation.

Artifact:
`corrected_runs/daylong_idea_search_v1/layer_mixture_operating_point_v1.json`.


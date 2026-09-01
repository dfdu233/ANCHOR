# Preregistration — Sparse Evidence Boundary and Penalized Spatial Scan v1

## Research question

Does a medical VLM fail on spatially small findings because whole-image mixing
dilutes rare visual evidence, and can a scale-adaptive spatial statistic recover
that evidence without accepting isolated noise peaks?

This is not an early-layer superiority claim.  It compares two ways of
aggregating the same frozen visual patch scores.

## Statistical background

Let an image contain `N` visual patches.  For a claim `c`, let `z_j(c)` be a
standardized local evidence score.  Under claim absence, scores are centered
near zero.  Under claim presence, suppose an unknown connected region `R*` of
`k` patches has mean shift `mu`.

A global standardized mean has signal-to-noise ratio approximately

```text
S_mean = mu * k / sqrt(N).
```

Thus a real lesion is diluted by the fraction of occupied patches.  A normalized
sum over the correct local region has

```text
S_local = mu * sqrt(k).
```

Because the region is unknown, searching many windows can select random peaks.
The fixed scale-adaptive statistic is therefore

```text
T(c) = max over square windows R [ sum_{j in R} z_j(c) / sqrt(|R|)
                                  - sqrt(2 log M_|R|) ],
```

where `M_|R|` is the number of searched windows at that scale.  The subtraction
is a multiple-search penalty, not a tuned threshold.  It makes a single random
high patch insufficient simply because thousands of candidate regions were
examined.

The resulting nontrivial regime is

```text
mu * k / sqrt(N)  <  detection threshold
mu * sqrt(k)      >  sqrt(2 log M_k) + detection threshold.
```

Here global averaging is asymptotically powerless while the spatial scan is
detectable.  This is the candidate mechanism behind the observed lesion-area
boundary.  The theorem is a research framing and must not be claimed as a new
generic scan-statistic theorem; novelty can only come from establishing that
medical VLM visual evidence actually occupies this regime and using the
boundary to design decoding.

## Frozen implementation

1. On the old development split only, fit one diagonal-LDA direction per
   finding from pre-projector global visual means and clear `0/3` versus `3/3`
   labels.  This supervised direction is a mechanism probe, not the final
   training-free algorithm.
2. Project every pre-projector visual patch onto each frozen direction.
3. Estimate positionwise null mean and variance from development `0/3` images.
4. Freeze square windows of side `1, 2, 4, floor(grid_side/2)` and the penalty
   `sqrt(2 log(number of windows at that scale))`.
5. Fit two fixed-regularization development logistic models.  The strong base
   is `finding + final_margin + patch_mean + patch_max + patch_top5`; the
   enhanced model adds only `scan`.  Including all standard poolers together
   is stricter than selecting one after seeing confirmation labels.
6. Open the already existing 532-image confirmation artifact only through the
   frozen directions, transforms, and models.  Report mean, max, top-5%, Higher
   Criticism, and scan individually, but the only primary candidate is scan.

Huatuo is screened first.  Hulu patch collection occurs only if Huatuo passes.

## Gate

For each model:

- scan-enhanced macro AUROC minus base at least `0.02`;
- paired finding×label bootstrap AUROC-delta lower 95% bound above zero;
- paired NLL-improvement lower bound above zero;
- positive AUROC delta on at least five of seven findings;
- development fitted scan coefficient positive.

Joint mechanism promotion requires both models.  Failure on Huatuo closes this
specific supervised local-score family and prevents Hulu GPU expenditure.

## Interpretation boundaries

- PASS proves incremental sparse-patch decodability, not lesion localization,
  causal use, report-level hallucination reduction, or training-free mitigation.
- The final method must replace supervised finding directions with a frozen
  model-native or text-conditioned local score and must pass the same gate.
- Any later decoding result must report false positives and false negatives,
  matched claim count, response length, rejection, and criterion sensitivity.
- Ordinary lesion blur is already NO-GO and is not part of this method.

Primary outputs:

- `corrected_runs/daylong_idea_search_v1/sparse_patch_scan_huatuo_v1.json`
- conditional `corrected_runs/daylong_idea_search_v1/sparse_patch_scan_hulu_v1.json`

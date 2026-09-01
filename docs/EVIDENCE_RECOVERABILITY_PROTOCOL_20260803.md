# Evidence Recoverability: frozen first-screen protocol

Frozen on 2026-08-03 before inspecting the independent confirmation outputs.

Protocol amendment v2 was made after the first Huatuo v1 analysis but before
inspecting Hulu: shuffled donors must also have the same ground-truth polarity
and cannot be the target record itself.  The original wording controlled only
finding and therefore did not fully preserve class-conditioned trajectories.
The v1 artifact is retained.  This amendment is stricter and was not motivated
by a positive Huatuo result: Huatuo v1 had already failed with negative excess.

## Question

For a clinical claim whose final diagnostic margin has the wrong polarity, does
an earlier sampled decoder layer contain **case-specific** correct polarity, or
does an apparent oracle merely exploit a generic layer/finding class bias?

This screen is necessary but not sufficient evidence.  A positive result must
later survive dense-layer collection and causal activation transport.

## Data and roles

- Development: existing 640-claim VinDr split; thresholds are fit here only.
- Confirmation: independent 1,920-claim VinDr image split.
- Findings: the eight preselected findings, stratified by 0/3, 1/3, 2/3, 3/3
  reader votes.
- Primary truth: 3/3 is positive and 0/3 is negative.  The disagreement bins
  are retained for later calibration analyses but do not define FP/FN here.
- Actual FP/FN: sign of the unmodified final support-minus-refute diagnostic
  margin.  Earlier-layer calibration never changes which cases count as errors.
- Models: HuatuoGPT-Vision-7B and Hulu-Med-4B in the first screen.

The manifest is image-disjoint.  Patient-disjointness cannot currently be
verified because the available manifest has no patient identifier; this is a
declared limitation, not silently treated as patient-disjoint.

## Measurements

1. **Native convex reachability:** whether any sampled pre-final raw margin has
   the correct sign.  This is the literal necessary boundary for convex fusion,
   but may be dominated by early-layer class bias.
2. **Dev-calibrated reachability:** subtract a per-finding, per-layer threshold
   fit on development clear cases, without ever flipping the polarity axis.
3. **Case-specific excess:** compare observed calibrated reachability with a
   within-finding, same-truth shuffled null.  The donor's whole sampled-layer
   trajectory is exchanged as one unit, preserving truth-conditioned class
   bias and cross-layer covariance; self-donation is prohibited.
4. Report FP and FN separately, plus layer-wise clear-case accuracy and the
   sampled taxonomy `absent`, `transient_early_correct`, and
   `prefinal_correct_final_reversal`.

The sampled taxonomy is descriptive.  It must not be called an all-layer state
or causal clinical evidence.

In particular, `stranded` is reserved for a future causal experiment where a
clinical state is decodable from a visual/claim representation but never enters
the answer-position polarity margin.  A sampled answer-position sign change is
only called a reversal here.

## Frozen decision rule

- **Advance to dense-layer causal transport:** the same error type has at least
  30 errors per model, observed case-specific excess is at least 0.10 in both
  models, and its one-sided whole-trajectory randomization p-value is below
  0.05 in both models.
- **Falsify the current logit-lens version:** excess is at most 0.05 for both FP
  and FN in both models.  Then raw FP/FN asymmetry is interpreted as layer-wise
  affirmative bias, not Evidence Recoverability.
- **Mixed result:** do not tune thresholds.  Inspect per-finding replication
  and one dense-layer, one-model audit only if the sign is consistent and the
  failure is plausibly sampling resolution; otherwise stop this representation.

No ETD threshold tuning, method comparison, or causal claim is permitted before
this gate is evaluated.  A convex reachability oracle is only a necessary upper
bound: it does not supply a label-free selector and is not itself a decoding
method.

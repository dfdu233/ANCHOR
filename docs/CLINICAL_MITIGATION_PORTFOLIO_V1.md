# Clinical Mitigation Portfolio (CMP) v1

> **Disposition, 2026-08-10:** retained as a competition baseline, rejected as
> the current paper core. QueryBandits already establishes per-query mitigation
> selection, V-ITI establishes selective intervention, and the required
> pre-treatment gate failed locally: source-trained plain prediction/NLL/length
> improved CXR BAcc by only +0.14pp (95% CI -0.29 to +0.61pp), requested RAG on
> 90.1% of cases, and increased FP. Post-treatment response stacking remains
> useful, but cannot satisfy the one-treatment deployment or novelty claims
> below.

## One-sentence idea

Hallucination mitigation is not one universally beneficial decoder: it is a
set of treatments with heterogeneous benefits and adverse events.  Learn the
individual treatment effect of each mitigation and run only the method that is
Pareto-safe for fabrication, omission, and compute.

## Why this is a new problem rather than another ensemble

Existing papers normally report the average effect of VCD, RAG, SPIN, beam, or
another decoder.  Our cached results show that this average hides large
crossovers: two Huatuo probes disagree on 21.9% of CXR claims; when they agree,
error is 10.6%, but when they disagree it is 45.7%; each probe rescues more than
half of the other's errors.  A method can therefore have a negative average
effect while being the best treatment for a recognizable subgroup.

For claim `i` and mitigation `m`, let the potential outcome be

\[
Y_i(m)=(F_i(m),O_i(m),C_m),
\]

where `F` is fabrication loss, `O` is omission loss, and `C` is inference cost.
The scientific question is not “which decoder wins on average?” but:

> Can a model predict the individual mitigation effect before paying for all
> mitigations, and does this effect transfer across hospitals, findings, and
> tasks?

## Algorithm

### 1. Offline counterfactual table

On training/OOF data only, run the complete faithful baseline portfolio
`M={greedy, beam, VCD, DoLa, SPIN, VISTA, SECOND, RAG, ...}`.  Convert every
answer to the same clinical claims and record, for every method:

- fabricated positive claims;
- omitted supported claims;
- polarity/attribute/location errors;
- reader-disagreement calibration;
- latency and generated-token cost.

This creates observed potential outcomes for every training claim without
assuming that the highest-scoring method is universally safe.

### 2. Cheap state, not all test-time answers

From one greedy prefill (or one short draft), collect an outcome-blind state
`s_i`: final and selected-layer margins, entropy, image-null gap, prompt/image
attention, finding identity, and base response length.  Full mitigation outputs
are labels for policy training, not test-time features in the final method.

### 3. Multi-outcome treatment model

For each method, a shared low-rank router predicts two conditional risks:

\[
(\hat r^F_m(s),\hat r^O_m(s))
=g_\theta(s,m).
\]

Training uses doubly robust/offline-policy losses when the full table is
incomplete, and direct supervised potential-outcome loss when all methods were
run.  Group-DRO takes the worst risk over hospital/domain, finding, and prompt
families, preventing a pooled win from hiding a failed subgroup.

### 4. Pareto-safe policy

Greedy is the reference treatment.  Method `m` is admissible only when its
calibrated upper risk bounds are non-inferior in both clinical directions:

\[
U^F_m(s)\le U^F_0(s),\qquad
U^O_m(s)\le U^O_0(s).
\]

Among admissible methods, choose

\[
\pi(s)=\arg\min_m
\left[U^F_m(s)+\lambda_O U^O_m(s)+\lambda_C C_m\right].
\]

If no method is admissible, keep greedy.  For OE abnormality listing, positive
claim count `K` is fixed before policy comparison, so deletion cannot masquerade
as hallucination mitigation.

### 5. One selected treatment

The router selects one method before full generation; only that treatment is
run.  An adaptive variant first runs greedy and requests a second treatment
only for uncertified claims.  It is not a majority vote and does not require
all portfolio members at deployment.

## Current evidence and its boundary

- Huatuo Knowledge-MIMIC, strict fit/tune/test split: an intervention-code
  candidate improved test BAcc `0.7275→0.7846`, FP `67→61`, FN `45→29` on
  407 samples.
- A simpler source-trained two-probe router transferred Knowledge-MIMIC→CXR and
  improved BAcc `0.8187→0.8445` with 1.67 average probes.
- A pooled direction-wise safety certificate did not preserve omission under
  shift.  This is evidence that single-domain marginal certification is
  insufficient, not evidence that CMP is already safe.
- Current pilots use post-intervention response codes.  The final pre-generation
  router and multi-domain Pareto bounds remain unverified.

## ICLR novelty delta that must survive review

1. Introduce **individual mitigation effect** as the object of study and show
   that average decoder rankings conceal predictable treatment crossover.
2. Provide the first claim-level potential-outcome matrix spanning perception,
   decoding, calibration, and retrieval treatments in medical VLMs.
3. Learn an OOD, multi-adverse-event treatment policy rather than a generic
   hallucination detector or model router.
4. Prove/verify that policy benefit is predicted by treatment-effect
   heterogeneity across models and domains, while fixed-coverage FP and FN do
   not trade off.

This delta is invalid if CMP reduces to majority vote, uses every method at
test time, or differs from HALP/generic LLM routing only by medical terminology.

## Fast experiment ladder

1. **Retrospective oracle:** produce method-by-sample FP/FN matrices and quantify
   crossover, oracle headroom, error correlation, and ranking instability.
2. **Honest policy pilot:** fit/tune/calibrate/test by patient; compare best
   single, majority, stacking, HALP-style correctness routing, CMP, and oracle.
3. **Cross-domain test:** train policy on Knowledge-MIMIC + VinDr dev and open
   CXR-VisHal/Hulu only once.  No target threshold fitting.
4. **Prefill distillation:** replace response-code features with base prefill
   state; require at least 90% of the full-code gain at one selected treatment.
5. **OE fixed-K:** run on VQA-RAD/Visual-MIMIC; claim count and coverage matched.
6. **Report transfer:** claim-wise policy on IU-Xray/MIMIC-CXR; no real clinical
   hallucination claim without expert labels.
7. **Mechanism:** test whether treatment selection aligns with recoverability,
   visual reliance, and retrieval dependence using activation patching; these
   explain the policy but do not define its labels.

## Fatal gates

- At least two models and two unopened datasets: BAcc/claim utility +2pp with
  image-cluster CI excluding zero.
- FP relative reduction at least 20%, FN/omission non-inferior within 1pp.
- Fixed-K OE gain; no answer shortening, refusal, parse failure, or all-negative
  shortcut.
- Prefill router retains at least 90% of full response-code gain and averages
  at most 1.5 full generations.
- Group-wise worst-domain policy beats the best fixed treatment.
- Collision audit leaves a substantive difference from HALP, HalluTrace,
  generic model routing, and medical ensembles.

Until all gates pass, CMP is an experiment-backed candidate, not an ICLR-ready
paper and not a safety guarantee.

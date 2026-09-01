# ASCC: frozen mechanism contract

Date: 2026-08-02  
Status: v1 construct invalidated; symmetric factorial v2 primary screen failed.

## Superseding construct decision

The complete v1 Huatuo run is engineering-complete, but it is not scientific
evidence.  Before its outcomes were admitted, the construct audit found five
fatal defects: the linear `K` was not definite-state mass, `unlikely` was not a
symmetric negative endpoint, prompt pairs jointly changed noun and speech act,
neutral third-state admission was not established, and independently sampled
images were treated as paired bootstrap clusters.  The authoritative record is
`corrected_runs/ascc/huatuo_score_v3/INVALIDATED_BEFORE_OUTCOME_INSPECTION.json`.

The later exploratory v1 analysis and `primary_progression_decision_v1.json`
are retained only as provenance of an inadmissible assay; their numerical gate
must not be cited as an ASCC positive or negative result.  The only authorized
successor is the pre-outcome symmetric `absent/uncertain/present` factorial v2,
which orthogonalizes clinical noun (`abnormalities-findings`) and speech act
(`list-describe`) and bootstraps independent images within vote strata.
Replication edges, a second model, hidden-state patching, and mitigation remain
unauthorized unless every factorial-v2 primary screen gate passes.

## Symmetric factorial v2 final decision

The Huatuo primary edge completed all 2,036 registered forwards.  The marker
instrument is native rather than off-manifold: every final-layer cell has at
least 93.99% probability mass on `absent/uncertain/present`, and its restricted
top-1 rate is 100%.  The clear-bin cross-fit is also valid (`R2=0.9983/0.9981`).

Nevertheless, the clinical-noun DID is effectively zero (`0.00280`, stratified
bootstrap 95% CI `[-0.00982, 0.01523]`) versus the frozen minimum `log(1.5) =
0.4055`.  The raw ambiguous-bin shifts are positive, but local interactions do
not exclude zero, neutral third-state admission fails on the negative boundary,
and the shifts do not survive clear-bin affine residualization.  The evidence
therefore identifies a generic lexical calibration shift, not an
ambiguity-selective commitment mechanism.  The v2 branch is a hard NO-GO; its
hash-bound decision is
`corrected_runs/ascc/huatuo_factorial_score_v2/primary_progression_decision_v2.json`.

## Question

The surviving question is **Ambiguity-Selective Commitment Collapse (ASCC)**:

> When the image, observation, diagnosis candidate, polarity choices, and
> response vocabulary are fixed, does abnormality-focused framing selectively
> suppress the reader-disagreement state during language commitment, rather
> than adding a constant assertiveness prior?

This is not a claim that prompts generally matter, that medical VLMs are
uncalibrated, or that a new uncertainty direction exists.  Those cells are
occupied by prompt-induced hallucination, Tinted Frames, certainty calibration,
generic verbal-uncertainty features, and multi-reader uncertainty prediction.
The only potentially new cell is a reader-ambiguity-dependent gate on how a
known uncertainty representation is deployed after a clinical proposition is
fixed.

## Why CIPCA v1 was KILLed

The four discovery pairs had exact sign-test `p=0.125` and one repeated language
template.  Neutral answers retained alternative diagnoses whereas existential
answers removed them, so the observed change mixed differential pruning with a
certainty-marker replacement.  The parser also counted `no signs of ... would
suggest pneumonia` and `rule out` constructions as positive definite diagnoses.
Those bugs are now regression-tested, but repaired regex output still cannot
define clinical certainty truth.

Therefore the 1,393-image / 2,786-generation free-OE substrate was never run on
GPU.  Free generation can later test external validity by intent to treat, but
post-output exact-overlap pairs can never authorize the mechanism.

## Frozen controlled instrument

Every model input fixes a complete assistant prefix such as:

```text
Findings: Lung opacity is present. Impression: Pneumonia is
```

Only three single contextual tokens are scored:

```text
 unlikely | possible | present
```

Two independently worded framing pairs contain the same state vocabulary and
explicitly license present, absent, and uncertain answers.  The manipulation is
only `findings/description` versus `abnormalities/list`; neither prompt inserts
the candidate diagnosis.

For logits `(z-, z?, z+)`, the frozen two-plane readout is:

\[
K=\tfrac12(z_+ + z_-)-z_?, \qquad \pi=z_+-z_-.
\]

`K` measures definite mass relative to the missing third state, while `pi`
tracks polarity.  ASCC requires a change in `K` that is not explained by a
change in `pi`.

## Reader-grounded design

All 200 discovery images are excluded.  Selection uses only the exact
R8/R9/R10 panel.  Parent observations require at least 2/3 support.  Diagnosis
support retains all four bins.  Within each edge, `0/3↔1/3` and `2/3↔3/3` are
matched exactly on parent vote count and released-DICOM aspect bucket.  The
released DICOMs omit `ViewPosition`, so no false view-matching claim is made.

Frozen census:

- Lung Opacity→Pneumonia: 94 negative-boundary pairs and 100
  positive-boundary pairs;
- Infiltration→Pneumonia: 31 and 53 pairs;
- Nodule/Mass→Lung tumor: 72 and 34 pairs;
- 768 image-claim rows, four prompts each, 3,072 single-forward jobs;
- substrate fingerprint:
  `039fe7486d583dab03a051ec8bd8de49956fe547398bda9337f70496d97f63e4`.

There is no low-powered 50/50 confirmatory split.  No controlled logits were
seen before the census, prompts, markers, estimand, and analysis code were
frozen.  Gatekeeping is sequential: Huatuo opacity first, then a second model on
opacity, then replication edges.

## Primary estimand and gates

For support bin `s`, let `Delta_s` be existential-minus-neutral `K`, averaged
over the two frozen prompt pairs.  The primary interaction is:

\[
\theta=\tfrac12[(\Delta_1-\Delta_0)+(\Delta_2-\Delta_3)].
\]

The mechanism proceeds only if all hold:

1. neutral polarity directionally admits reader support before framing is
   analyzed;
2. matched-pair bootstrap 95% CI for `theta` lies above zero;
3. both local contrasts are positive;
4. both prompt pairs have the same interaction direction;
5. the 90% CI of the analogous polarity interaction lies wholly inside
   `[-0.2, 0.2]` log-odds;
6. a second model and at least one replication edge agree;
7. text-only and image-swap effects cannot explain the image-conditioned
   interaction.

A constant framing shift across vote bins is a generic prompt prior and KILLs
the headline.  A visual-attention/evidence change is a Tinted Frames boundary
replication.  Removal by generic-VUF residualization makes this a domain audit,
not an ICLR mechanism paper.

## Conditional mechanism and method

Hidden-state patching is unauthorized until the behavioral gate passes.  If it
does, neutral↔existential activation interchange must change `K` selectively on
1/3 and 2/3 images while preserving `pi`, claim identity, and the observation
and diagnosis tokens.  The causal target is prompt-conditioned gain on the
known verbal-uncertainty representation, not a newly claimed direction.

Only after causal success may mitigation be built: remove the framing-induced
gain increment only when latent reader ambiguity is present.  It must preserve
claim count, diagnosis, polarity, length, and coverage; beat temperature,
prompt normalization, generic VUF steering, random/norm-matched steering, and
instruction contrastive decoding; and reduce physician-rated OE
overcommitment without increasing omission.

## Authoritative artifacts

- `anchor/corrected_sgta/prepare_ascc_interaction_v1.py`
- `anchor/corrected_sgta/run_huatuo_ascc_interaction_v1.py`
- `anchor/corrected_sgta/analyze_ascc_interaction_v1.py`
- `corrected_runs/ascc/confirmatory_substrate_v1/substrate_config.json`
- `corrected_runs/ascc/confirmatory_substrate_v1/selected_manifest.jsonl`
- `tests/test_ascc_interaction_v1.py`

The detached primary run is `huatuo-ascc-primary-v1`; it is atomically
resumable and survives terminal disconnection.

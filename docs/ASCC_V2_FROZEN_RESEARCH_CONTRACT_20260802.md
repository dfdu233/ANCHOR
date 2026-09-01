# ASCC-v2: symmetric factorial screen

Date: 2026-08-02  
Status: frozen before outcomes; Huatuo primary scoring running.

## Construct repair

ASCC-v1 was invalidated without reading its scores.  The successor fixes every
construct-level flaw identified by two independent blind audits:

- symmetric states are now `absent / uncertain / present`; `unlikely` is no
  longer misrepresented as definite negative;
- commitment is restricted-state definite mass,
  `C=logsumexp(z_present,z_absent)-z_uncertain`, with independent polarity
  `P=z_present-z_absent`;
- prompts form a true `describe/list × findings/abnormalities` 2×2 factorial;
  the primary noun contrast changes one word within each speech act;
- the full eligible census is retained; independent images are not converted
  into artificial statistical pairs;
- discovery exclusion comes from the frozen selection manifest, not generation
  outputs;
- neutral third-state admission, local polarity admission, positive ambiguous-
  bin shifts, per-boundary and per-speech-act CIs, local polarity equivalence,
  and a clear-bin five-fold affine/temperature residual are mandatory.

The v1 seal is
`corrected_runs/ascc/huatuo_score_v3/INVALIDATED_BEFORE_OUTCOME_INSPECTION.json`.

## Frozen assay

The fixed assistant prefix supplies the observation and diagnosis claim, so
this is a controlled commitment instrument, not natural OE evidence.  The four
prompt cells differ only by the two declared factors and share the exact state
instruction.  Three marker tokens are context-certified for every prefix.

The discovery-disjoint exact-panel census contains:

- Lung Opacity→Pneumonia: 103/108/171/127 images in 0/1/2/3-reader bins;
- Infiltration→Pneumonia: 36/39/74/79;
- Nodule/Mass→Lung tumor: 191/72/61/44;
- 1,105 image-claim rows and 4,420 jobs per model;
- fingerprint
  `a9191fa3fb6fe5866b754a62419a1a44e032232234c4c52c33dd2d9ada4cecde`.

Model weights, tokenizer/config files, Huatuo source, runtime versions, adapter
identity, prompts, markers, DICOM hashes and shard identities are bound into
provenance.  Ordinary full-vocabulary CausalLM logits are authoritative;
restricted marker probability mass and top-1 conformance are reported so the
assay cannot be mistaken for unconstrained generation.

## Primary statistics

For speech act `a`, define the within-image noun contrast

\[
\delta C_{i,a}=C_{i,a,abnormalities}-C_{i,a,findings}.
\]

Estimate two support-local effects:

\[
L_- = E[\delta C\mid 1/3]-E[\delta C\mid 0/3],\qquad
L_+ = E[\delta C\mid 2/3]-E[\delta C\mid 3/3].
\]

The screen statistic is `theta=(L_-+L_+)/2`.  Every local effect uses frozen
harmonic-overlap weights over parent-vote and DICOM-aspect strata and an
independent within-stratum image bootstrap.  There is no invented image pair.

Before framing can proceed, findings prompts must show separately that
uncertainty preference rises 0→1 and 3→2, while polarity rises 0→1 and 2→3.
Both local noun interactions, both speech-act replications and the noun shift
within each ambiguous bin require 95% CIs above zero.  Local polarity-
interaction 90% CIs must each lie inside `[-0.2,0.2]`.

A five-fold, clear-bin-only affine map
`z_abnormalities ≈ a*z_findings+b` is fitted out of fold.  Local and
ambiguous-bin effects must survive its residual, preventing a global
temperature/logit-scale change from masquerading as selective suppression.
The overall raw effect must also be at least `log(1.5)`.

Passing is named only `primary_edge_screen_passed`.  Global promotion still
requires a second model, a replication edge, text-only and image-swap controls,
generic-VUF separation, natural OE intent-to-treat evaluation, and physician
construct review.

## Artifacts

- `anchor/corrected_sgta/prepare_ascc_factorial_v2.py`
- `anchor/corrected_sgta/run_huatuo_ascc_factorial_v2.py`
- `anchor/corrected_sgta/analyze_ascc_factorial_v2.py`
- `corrected_runs/ascc/confirmatory_substrate_v2/`
- `tests/test_ascc_factorial_v2.py`

The active detached job is `huatuo-ascc-factorial-primary-v2`; it is PPID1
supervised, atomic and strictly resumable.

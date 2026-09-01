# CECD OE/report transfer：outcome-blind、content-conserving protocol

**Freeze:** 2026-08-03.  **Current-cache verdict:** **STRICT NO-GO for the
audited VQA-RAD/MIMIC/Physician-OE artifacts**.  **GPU:** not authorized.  This
document specifies what becomes executable only if the Huatuo+Hulu CECD
mechanism and closest-work envelope first pass their sealed CE gates.  It does
not adjudicate the separately audited VinDr reader-vector ontology-listing
track.

## 1. The transfer claim is deliberately narrow

The transfer question is not whether CECD can make an answer shorter or more
hedged.  It is:

> For a clinical proposition that a medical VLM emitted in a native open
> answer, does the render x prompt mixed derivative of the *same exact atomic
> assertion* predict an independently adjudicated support--commitment error
> beyond the clean assertion score and both marginal sensitivities; and can
> removing that interaction change certainty without changing proposition
> identity, polarity, positive `K`, required-claim coverage, refusal, or answer
> length?

The primary error label is claim-level and physician-defined:

- `fabricated`: a definite visual claim is **refuted** by the image;
- `overcommitted`: a definite visual claim is **undetermined** by the image;
- `supported`: the image supports the proposition at its expressed certainty;
- `unobservable`, knowledge, history, treatment and prognosis claims are kept
  visible but excluded from the visual-hallucination denominator.

A content-locked intervention can reduce inappropriate certainty or harm.  It
cannot truthfully be reported as removing a fabricated positive claim, because
that claim remains present.  A later fixed-`K` claim-exchange experiment would
be a separate preregistration and is **not authorized by this protocol**.

## 2. Current-cache census

The write-once audit is
`corrected_runs/cecd/oe_report_transfer_substrate_audit_v4/audit.json`
(SHA-256 `5d5c27e5eabc90aff2dc941b5150bd2e2830f914f3075bdd57c9b14ea88bb73f`).
It read only identities, counts, file hashes and the presence of provenance
fields.  It did not extract claims, compare a prediction with a reference,
call an LLM judge, compute efficacy, or use a GPU.

| Source | What is genuinely reusable | Fatal gap for formal CECD transfer |
|---|---|---|
| VQA-RAD qualified native OE | 200 questions for each of Huatuo, Hulu and LLaVA; 120 distinct images; all current 256/512-token artifacts pass the native generation gate | zero patient IDs; 80 questions reuse an image; benchmark direct answers are not exhaustive truth for claims added by a long response; mixed image modalities have no common CECD render admission |
| MIMIC report cache | 694 common image rows, 647 studies and 218 recoverable patients for each cached model | the pair is Hulu+LLaVA, not CECD's Huatuo+Hulu; zero independent image-level atomic truth; the single report and RadGraph are not visual truth; no MIMIC-JPEG render-equivalence admission |
| Physician-OE pack | blinded schema correctly separates visual, knowledge and unobservable claims | only 24 image groups/101 unique answer units; zero completed returns; no patient identity; no CECD orbit or intervention arms |

Therefore the current caches are useful as draft candidates and workflow
fixtures only.  They do not authorize a mechanism transfer, an efficacy test,
or GPU scoring.  In particular, waiting for the existing 24-image physician
returns would improve the baseline audit but would still not close patient
identity, sample size, target-model or product-orbit gates.

## 3. One admissible new clinical pack option

The following MIMIC construction is a sufficient fallback specification, not
the unique OE route.  A separately constructed VinDr fixed-ontology listing
track may close the truth problem more directly if it independently proves
patient/group isolation, retains every reader vote per ontology atom, passes
native multi-claim admission and obeys the same product-orbit and
content-conservation rules.

Use one MIMIC-CXR frontal study/image from each of **200 unique patients**:

- 80 patients in development and 120 in locked test;
- no patient, study or image crosses the split;
- the same patient cohort receives two tasks: native abnormality listing and
  native concise report generation;
- Huatuo and Hulu each generate one canonical draft per task before any orbit
  score is computed: `200 patients x 2 tasks x 2 models = 800` native units;
- this paired use of the same cohort tests output-form transfer, not
  independent-dataset replication.

### Independent clinical truth

Automatic extraction may propose spans, but every atom and its normalized
`finding + polarity + uncertainty + anatomy + attributes` record must be
human-confirmed while source model is blinded.  For each image/task, two
independent radiologists inspect the image with the benchmark report hidden
and separately provide:

1. `supported / refuted / undetermined / unobservable` for every emitted
   visual claim;
2. `required / optional / out_of_scope` relevance;
3. an image-derived exhaustive set of required claims for the task, created
   independently of both model drafts, so omission has a real denominator.

A third radiologist adjudicates disagreements.  The proposition, image/span
pointer, two initial records, adjudication and hashes remain linked.  RadGraph,
CheXbert, GREEN and an LLM judge may be auxiliary diagnostics but cannot fill
any of these truth fields.

Before orbit scoring, each model/task/split must contain at least:

| Split | supported | refuted | undetermined |
|---|---:|---:|---:|
| dev | 40 | 20 | 20 |
| test | 60 | 30 | 30 |

These are claim counts per model and task, not duplicated orbit cells.  If a
natural draft cohort misses a cell, the transfer is inconclusive; cases may
not be retrospectively oversampled after inspecting CECD scores.

### New equivalence admission

The VinDr DICOM admission does not silently transfer to MIMIC JPEGs.  A
separate, outcome-blind pack of at least 60 MIMIC images must admit at least
three non-identical render realizations and three proposition/speech-act-
preserving prompt realizations.  Two independent radiologists plus a third
adjudicator verify that each render preserves support and visibility; prompt
review also binds a clinical-template reviewer and language equivalence.  The
admission artifact and exact render/prompt implementation hashes are frozen
before model scores.

## 4. Atomic-claim teacher-forcing product orbit

Generate the native draft once on the canonical image and prompt.  After
human-confirmed atomization, freeze one exact `target_assertion` byte string
for each atomic claim.  That same byte string is teacher-forced in every cell:

```text
all admitted science renders x all admitted science prompts
+ identity-render x every science prompt
+ canonical-render x exact duplicate prompt
```

With the minimum `3 x 3` science grid this is 13 cells per claim; using the
current CECD `5 x 3` grid gives 19.  Every claim must have a complete orbit.
The score is length-normalized teacher-forced log-probability of the exact
assertion, accumulated in FP32.  Generating a fresh answer in each cell is
forbidden because it changes claim identity and confounds interaction with
content selection.

For claim `c`, render `r`, prompt `p`, let `m(c,r,p)` be the fixed-target score:

\[
I(c,r,p)=m(c,r,p)-\bar m(c,r,\cdot)-\bar m(c,\cdot,p)+\bar m(c,\cdot,\cdot).
\]

Development may fit only patient-grouped models.  The baseline contains clean
score, render and prompt main effects/sensitivities, full-orbit mean, claim
length, finding, task, expressed commitment and model-specific nuisance
controls.  CECD adds only centered interaction residuals.  Locked-test
inference uses patient-cluster bootstrap; two tasks on one patient are kept in
the same resample cluster.

Transfer requires, separately in Huatuo and Hulu:

- error AUROC improvement of at least `0.03` with bootstrap 95% CI above zero;
- harmful interaction alignment for independently adjudicated refuted or
  undetermined definite claims;
- both marginal-only and equally expensive full-orbit controls lose;
- direction holds in both OE listing and report output forms;
- identity-render and duplicate-prompt residuals remain below one tenth of the
  clinical interaction.

Failure in either model, or a result driven only by one output form, terminates
the broad transfer claim.  Layer localization and intervention remain
unauthorized until this behavioral transfer passes.

## 5. Content-conserving intervention contract

The causal intervention receives the frozen draft claim slots.  For every
case/model pair it must preserve exactly:

- ordered claim IDs and slot count;
- `finding`, anatomy and attributes;
- polarity and positive `K`;
- the set of physician-required claims covered by the draft;
- refusal and cap-hit state;
- answer word count within `[0.90, 1.10]` of the draft.

Only uncertainty/commitment realization may change.  Hence omission is
structurally equal, rather than merely statistically similar.  A method is
rejected if it obtains a lower overcommitment rate by hedging clear supported
claims: clear supported definite performance may fall by at most 1 percentage
point.  The minimum efficacy gate remains a 20% relative reduction in
physician-adjudicated overcommitment, patient-bootstrap CI excluding zero, in
both models.  Fabrication and false-negation counts are reported unchanged or
worse; they are never relabeled as fixed by a hedge.

Temperature scaling, uniform hedge, random/norm-matched direction, both CECD
marginals, full-orbit averaging, both Treble common-protocol semantics and
main-effect removal remain mandatory controls.  The fixed-content branch is a
mechanism-specific certainty test, not a general OE hallucination decoder.

## 6. Executable fail-closed boundary

The current census is implemented in
`anchor/corrected_sgta/audit_cecd_oe_report_transfer_substrate_v1.py`.
The future pack and intervention contracts are implemented in
`anchor/corrected_sgta/validate_cecd_oe_report_transfer_pack_v1.py`.

The future manifest validator requires the 200-patient split, both target
models, both tasks, clinical truth strata, a hash-bound admitted orbit, exact
target bytes, native-output quality and human-confirmed atomization.  Its
optional paired-output validator recomputes identity/polarity/`K`/coverage/
refusal/length conservation.  A successful structural validation authorizes
only orbit scoring; it explicitly emits `efficacy_claim_authorized=false`.

```bash
cd /home/dbw/ANCHOR

PYTHONPATH=.:anchor .venv-full/bin/pytest -q \
  tests/test_cecd_oe_report_transfer_v1.py

PYTHONPATH=.:anchor .venv-full/bin/python -m \
  anchor.corrected_sgta.validate_cecd_oe_report_transfer_pack_v1 \
  --manifest /path/to/frozen_manifest.json \
  --output /new/write_once/structural_validation.json

# Only after a behavioral pass and a separately frozen intervention run:
PYTHONPATH=.:anchor .venv-full/bin/python -m \
  anchor.corrected_sgta.validate_cecd_oe_report_transfer_pack_v1 \
  --manifest /path/to/frozen_manifest.json \
  --intervention-pairs /path/to/paired_outputs.jsonl \
  --output /new/write_once/content_conservation.json
```

## 7. Decision

The audited VQA-RAD/MIMIC/Physician-OE cache collection is **not weak evidence;
it is no evidence for the proposed transfer estimand**.  Conditional on a CECD
CE pass, admissible next substrates include the separately gated VinDr
reader-vector ontology-listing branch or the MIMIC patient-grounded fallback
specified here.  What is ruled out is retrospective RadGraph scoring of the
existing Hulu/LLaVA reports or promotion of the 24-image physician baseline
screen as formal CECD transfer evidence.

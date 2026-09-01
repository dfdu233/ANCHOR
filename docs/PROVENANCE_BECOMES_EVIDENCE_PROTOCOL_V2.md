# Provenance Becomes Evidence: controlled protocol v2 — NO-GO

> **Outcome-blind construct audit (2026-08-03): full GPU scoring is not
> authorized.**  Real outer shells can carry donor anatomy/pathology; a single
> scrubbed child is not a specific association-breaking control; the proposed
> DDD algebra cancels the parent; and training-seed and claim-level units were
> underspecified.  This file is retained as an audit trail.  The replacement
> design uses complementary randomized provenance assignment with matched
> children and treats the training run as the experimental unit.

## Frozen question

Does medical visual adaptation learn a shortcut from non-anatomical training-
source provenance to the source corpus's clinical-claim occurrence pattern,
causing weak-evidence target images to inherit that pattern as pseudo-evidence?

The proposed paper claim is deliberately narrower than style robustness:

> With anatomy held fixed, the same source-provenance counterfactual produces a
> claim-specific effect in a medically adapted child beyond its exact parent;
> the effect matches a source-only occurrence fingerprint, concentrates at
> independent reader disagreement, and is reduced when the child is trained on
> provenance-scrubbed versions of the same image-text examples.

This protocol does not call source-text occurrence a disease prevalence or a
pure reporting propensity.  It does not infer training influence from nearest
neighbours.  It does not claim that generic style instability is new.

## Why v2 is identifiable where v1 was not

Version 1 learned a source direction from clinically unmatched source and
target representations.  It was rejected before GPU work because source
disease content could define the direction and norm restoration could reinsert
polarity.  Version 2 therefore makes four changes:

1. the primary instrument is a fixed pixel counterfactual shared by parent and
   child, not separately fitted hidden directions;
2. the target radiograph core is bit-identical between the paired views;
3. the child is continued directly from the measured parent under an auditable
   training recipe;
4. a provenance-scrubbed child sees the same source examples and text, serving
   as a training-time causal control.

Hidden-state directions are permitted only after the primary effect, as a
mediation analysis with forward and reverse activation patching.

## Source-only semantic admission

Use actual PubMedVision assistant responses intersected with the frozen CXR
source index.  Alignment and instruction-tuning stages remain separate.  The
primary source fingerprint uses only generic, non-presuppositional Alignment
questions:

`b_c = logit P_source(assistant explicitly asserts claim c)`.

The denominator is every admitted generic Alignment response, not only rows
mentioning `c`; unmentioned is not negative.  Questions are audited separately.
Article/figure groups, not VQA rows, define source train/dev/review splits.
Uncertainty in `b_c` is estimated by PMC-group bootstrap.

The quartet source audit is only an extractor pilot.  Before model scoring, the
broad atomic VinDr ontology and eligibility rules are frozen.  A claim needs:

- at least 20 positive generic assertions in source-train and five in
  source-dev;
- blinded positive-state precision at least 0.90 and macro-F1 at least 0.80;
- at least 100 target images in every reader-vote bin used for a formal local
  boundary.

Raw occurrence is an admitted `b_c` only after human extraction review.  Sparse
claims remain sparse and are not replaced after model outcomes.

## Exact parent and controlled children

The primary model organism uses a public general VLM checkpoint whose full
weights, tokenizer, processor, vision tower, projector and chat template remain
fixed and available.  A practical candidate is Qwen2.5-VL-3B-Instruct, subject
to a pretraining-overlap audit.  From the identical parent checkpoint train:

- `child-natural`: medical adaptation on the frozen PubMedVision CXR source
  examples as released;
- `child-scrubbed`: the same example IDs, prompts, responses, order, optimizer,
  steps, token weights and random seed, but every training image receives the
  frozen provenance scrubber;
- optionally, a second independent seed for each child; seeds are replications,
  never pooled as image-level independent observations.

The first implementation uses parameter-efficient adaptation with a frozen
vision tower.  The exact LoRA targets/rank, optimizer, learning rate, batch,
epochs, precision and seed are selected from source-only development loss and
locked before any VinDr output.  Target data and target logits cannot select a
checkpoint.  Training must save base/adapter hashes and a row-order hash.

Both parent and children must pass the same contextual-marker semantic check
and native reader-directional admission.  A parent that cannot move its claim
score in the correct direction with reader votes is not a valid DiD control for
that claim.

HuatuoGPT-Vision is an external natural-model replication only unless an exact
pre-medical multimodal checkpoint lineage becomes auditable.  An approximate
Qwen/LLaVA relative cannot support an adaptation-causal claim.

## Primary provenance instrument: source shell

After the model processor's deterministic resize/pad geometry is frozen, define
an inner radiograph core and an outer shell.  For each VinDr target, construct:

- `neutral-shell`: target core on the frozen neutral canvas;
- `source-shell`: the identical target core at the identical coordinates, with
  only the outer shell replaced by a real PubMedVision shell donor.

Mandatory invariants:

- the target core arrays and alpha mask are byte-identical between views;
- the shell contains no OCR token, laterality marker, arrow, caption fragment,
  measurement, or explicit anatomical/pathology drawing after a frozen OCR and
  connected-component scrubber;
- no target label, model output, question or claim selects the donor;
- donors come from source-train groups disjoint from semantic review and all
  source evaluation groups;
- donor assignment is a deterministic hash of target image ID and seed;
- source-ID first-stage AUROC is measured using shell pixels alone on untouched
  source/target images and must have a lower 95% bound above 0.80;
- a shell-only classifier cannot predict target reader labels above its frozen
  equivalence margin;
- at least 100 paired views receive blinded clinical equivalence review before
  the full model assay.

The shell width is fixed from source-only/target-unlabelled geometry before
claim scoring.  One width is primary; no dose is selected from clinical logits.
If a source shell cannot produce an admitted source-ID shift without explicit
text/markers, this instrument fails.

## Overidentification instruments

One instrument is not sufficient for the paper claim.  Before GPU scoring,
freeze at least one additional same-image operator:

1. a rank-preserving source histogram/window map with unchanged spatial ranks;
2. a phase-preserving, low-amplitude source spectrum map with a hard structural
   equivalence gate.

Each operator has a matched neutral/nuisance counterpart, a source-ID first
stage, identical claim-independent donor assignment, source-coordinate dose
calibration and an immediate vision-only clinical equivalence test.  At least
two heterogeneous instruments must yield the same signed source fingerprint.
Failure of one cannot be repaired by changing its strength after logits are
seen.

## Double counterfactual outcome

Every model receives the exact same paired pixels and the exact same neutral
three-state prompt.  Score contextual `present`, `absent`, and `uncertain`
markers without generation.  The single signed primary is

`q = z_present - logsumexp(z_absent, z_uncertain)`.

Polarity `z_present-z_absent` and symmetric commitment
`logsumexp(z_present,z_absent)-z_uncertain` are mandatory decomposition/safety
outcomes and cannot replace a failed primary.

For model `m`, image `i`, and claim `c`:

`D_mic = q_mic(source-cue) - q_mic(neutral-cue)`.

The adaptation estimands are

`DD_natural,ic = D_child-natural,ic - D_parent,ic`,

`DD_scrubbed,ic = D_child-scrubbed,ic - D_parent,ic`,

and the training-control contrast

`DDD_ic = DD_natural,ic - DD_scrubbed,ic`.

The same counterfactual pixels make these differences comparable.  The primary
mechanism requires `DDD` to align with `b_c` specifically in the ambiguous
reader bins.

For each claim compute both local boundaries and their fixed average:

`delta^-_c = DDD_c(1/3) - DDD_c(0/3)`,

`delta^+_c = DDD_c(2/3) - DDD_c(3/3)`,

`delta_c = 0.5*(delta^-_c + delta^+_c)`.

The two local effects must be reported separately and have compatible signs.

## Inference and evidence levels

The effective source-prior sample size is the number of claims, not the number
of image-claim rows.  Inference nests:

1. PMC article-group bootstrap for `b_c`;
2. target image-cluster bootstrap for `delta_c`;
3. child-seed replication when available;
4. leave-one-claim-out influence analysis.

Raw claim permutation is descriptive because claims are not exchangeable.
Confirmatory regression must first residualize, using outcome-independent
covariates, source opportunity/count reliability, target prevalence, native
margin/reader separability, and extractor reliability; use a stratified
Freedman--Lane permutation or a prespecified hierarchical errors-in-variables
model.  The exact method, one-sided sign and multiplicity correction are frozen
before full scoring.

VinDr alone is a discovery substrate if fewer than 12 atomic claims qualify.
Paper-level vector evidence requires at least 12 eligible claim-dataset units
across at least two independent multi-reader datasets, with family/dataset-
stratified inference.  No image-row bootstrap can waive this requirement.

## Gates

GPU scoring is unauthorized until all of the following are independently
audited:

1. source semantic extractor and source occurrence fingerprint;
2. exact parent/child lineage and reproducible training manifest;
3. source-shell first stage, OCR/marker absence and byte-identical target core;
4. blinded clinical equivalence pack;
5. a second fully specified provenance instrument;
6. target sample/power table and untouched model-evaluation split;
7. contextual marker semantics and native directional admission for parent and
   both children;
8. a locked signed-primary analysis with nested source/target uncertainty.

The discovery mechanism advances only if:

- `DDD`--`b` alignment has the frozen sign and uncertainty interval excluding
  zero;
- both ambiguity boundaries agree and clear-case performance falls by less
  than one percentage point;
- at least two provenance instruments agree after source-ID dose matching;
- source-prior permutation, wrong-source shell, neutral donor, matched nuisance,
  output-temperature, and unrelated-token controls fail to explain the effect;
- the provenance-scrubbed child retains source-dev quality and clear target
  performance while materially reducing the echo.

If only generic instability remains, the result is assigned to LENS/VCD-style
robustness and this mechanism is killed.

## Mitigation ceiling

The first mitigation is training-time provenance randomization/scrubbing,
because it directly breaks the identified channel.  A test-time method is
allowed only after causal mediation is established; it may subtract the
counterfactual provenance residual along the independently estimated source
fingerprint, gated by calibrated ambiguity.  OE evaluation fixes positive
claim count and coverage.  Shortening, refusal, universal hedging or claim
deletion is failure.

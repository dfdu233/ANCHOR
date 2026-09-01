# Provenance Prior Imprinting (PPI): randomized model-organism protocol v3

> **Decision lock (2026-08-03): conditional GO for CPU/operator preflight only.**
> Full GPU training is NO-GO until the randomization law, processor-visible
> shell, held-out cue families, identical three-arm optimizer, and multiple
> orthogonal fingerprints are implemented and audited. Even after a positive
> experiment, the allowed conclusion is that a controlled VLM continuation can
> learn a clinically empty provenance cue as a clinical prior. It must not be
> presented as a causal explanation of the natural Huatuo checkpoint without a
> separately admitted natural-family bridge and a shared preregistered
> mediation signature.
>
> **Final v3/v3.1 progression decision:** the controlled CPU operator passed,
> but the repaired PubMedVision source substrate retained only 2 of the required
> 8 natural claims. The natural bridge therefore failed and GPU work is not
> authorized under this protocol.

> **Construct correction:** all executable two-plane assignment semantics are
> superseded by `docs/PROVENANCE_TWO_PLANE_BINDING_PROTOCOL_V3_2.md`. Source
> precision means PubMedVision linguistic definiteness; VinDr reader votes are
> external evaluation only and may never enter training assignment.

## Core question

Can clinically empty image provenance be learned as claim evidence during
medical adaptation, and does that learned cue cause false clinical commitment
when independent readers do not unanimously support the claim?

The experiment does not infer causality from a naturally confounded source
domain.  Instead, it randomizes the provenance--claim association while holding
the complete radiograph, answer text, example set and training compute fixed.

## One clean intervention

No padding band is assumed. Gate A must trace the frozen processor through
smart resize, `image_grid_thw`, patch merge and attention to identify whether
any processor-visible but clinically empty support actually exists. If ordinary
padding is masked or absent, the experiment must use an explicitly named empty
visual-token frame or a preregistered unmasked border whose induced clinical
rescaling is measured; it may not call that intervention unchanged padding.
The cues contain no donor image, anatomy, device, text, marker, arrow or
measurement. `A` and `B` are *families* of matched nonsemantic provenance cues,
not two fixed textures. Train and test exemplars are disjoint. The families are
matched for occupied area, intensity histogram, edge energy, seam, compression
and source-ID magnitude, and held-out-family transfer is mandatory.

The Qwen2.5-VL CPU trace establishes the concrete admissible construction:
cache one clinical ROI, paste it into neutral/A/B fixed canvases whose ROI and
frame boundaries are multiples of 28, and pin `use_fast=False`. Every frame
patch is attended; this is an **unmasked attended visual-token frame**, never
padding or metadata. The neutral framed canvas is the parent/control input.
Comparing an unframed parent with framed children is prohibited because it
changes the visual-token count (64 versus 100 in the registered trace).

The following identity is tested on serialized processor tensors, not merely
on pre-processor PNGs:

`clinical_tensor(image,A) == clinical_tensor(image,B)`

inside the frozen clinical mask.  Maximum absolute difference must be exactly
zero.  Shell-only classifiers and blinded clinical review verify that neither
shell carries clinical content.  A shell that cannot survive the processor
without changing the clinical field fails before training.

## Randomized provenance fingerprints

Freeze all eligible atomic claims before creating assignments. Precompute the
complete admissible set `R*` of balanced sign vectors under source-only
feasibility rules, then draw `r` uniformly from `R*` using a registered seed.
No claim may be removed after this draw. The assignment and randomization unit
is a unique image/figure (PMC-group clustered), never an exchangeable VQA row.
For every randomization replicate the same assignment optimizer is rerun; the
observed achieved `g` is never merely permuted. The target vector is a
randomized experimental treatment, not a source statistic.

Discovery uses at least two preregistered approximately orthogonal/Hadamard
fingerprints. This identifies a response matrix `M` in `t=M g`, rather than
mistaking a generic abnormality or affirmative-language direction for
claim-specific learning. Claim-specific wording is allowed only if diagonal
response exceeds off-diagonal and global-abnormality responses. Otherwise the
result is named multivariate clinical-prior-vector imprinting.

For the complete fixed source example set, solve a label-aware but
outcome-blind binary assignment `a_i in {A,B}` that:

1. exactly balances total A/B counts and training-token mass;
2. balances source stage, PMC article group, image count, response length and
   all frozen nonclinical nuisance strata;
3. makes the achieved positive-assertion contrast
   `g_c = P(y_c=1|A)-P(y_c=1|B)` follow `r_c` with a frozen minimum magnitude;
4. limits pairwise claim co-occurrence imbalance and records the complete
   achieved `g` vector;
5. never changes an image, prompt, response, claim label or loss weight.

The assignment optimizer and acceptance thresholds are locked using source
labels only.  If it cannot realize the target vector without violating balance,
the affected claims are removed before any model training according to frozen
rules; they are never replaced from model outcomes.

Construct three matched training arms for each registered training seed:

- `child-plus`: example `i` receives shell `a_i`;
- `child-minus`: every assignment is complemented, A<->B;
- `child-balanced`: the same optimizer targets the zero vector under the same
  constraints and tolerance used by the two signed arms.

All arms start from the identical parent weights and use the identical example
IDs, text, row order, optimizer, hyperparameters, token weights, step count,
byte-identical LoRA initialization, dropout/RNG streams and optimizer state. A
minimum of five matched seed triplets is
required for a confirmatory controlled-model claim; three is a discovery
screen.  The child run/seed is the top-level experimental unit.

## Why the complementary arm matters

At test time every target radiograph is rendered twice, once with A and once
with B, while its clinical tensor remains identical.  For claim `c`, define the
single signed primary:

`q = z_present - logsumexp(z_absent,z_uncertain)`

and

`D_mic = q_mic(A)-q_mic(B)`.

The exact parent may have an arbitrary generic preference for A or B.  Medical
adaptation is tested by

`DD_plus = D_child-plus-D_parent`,

`DD_minus = D_child-minus-D_parent`,

and `DD_balanced = D_child-balanced-D_parent`.

The preregistered crossover predictions are:

- `DD_plus` aligns with `+g`;
- `DD_minus` aligns with `-g`;
- the common drift satisfies `0.5*(DD_plus+DD_minus) ~= DD_balanced`;
- `0.5*(DD_plus-DD_minus)` aligns with `g` and exceeds all sham contrasts.

The parent is not allowed to disappear algebraically: the first three
adaptation contrasts are co-primary admissions; crossover establishes treatment
specificity only after they pass.

This flip is the mechanism test.  Generic brightness/style sensitivity,
natural source disease mix, donor pathology and ordinary language prior cannot
predict opposite claim-specific effects in complementary children trained on
the same images and text.

## Model lineage and training gate

Use either an exact public parent followed by our controlled continuation, or a
released medical checkpoint only as external validation.  The initial lineage
candidate is Qwen2.5-VL-7B-Instruct with a frozen PEFT continuation; the
official Huatuo Qwen2.5-VL checkpoint is not called its exact child until model
card, conversion recipe and tensor-distance audits agree.

Before target scoring, training must pass:

- byte-identical source example/text manifests across arms;
- exact A/B and token-mass balance;
- source-only loss and held-out clinical-description quality equivalence across
  arms;
- adapter/base/config/tokenizer/processor hashes;
- no target image, reader vote or target-model output used for training,
  checkpoint choice or early stopping;
- fixed LoRA targets/rank, optimizer and checkpoint step chosen once from a
  source-only pilot.

## Staged authorization

### Gate A: CPU preflight

- human-admitted claim extractor;
- shell tensor identity and shell-only clinical null tests;
- three-arm manifests and achieved randomized `g`;
- source-only power analysis with training seed as the experimental unit;
- exact parent and runtime hashes.

### Gate B: tiny GPU admission

Train one deliberately small discovery triplet or the registered first seed,
then use a fixed disjoint target admission set only to verify:

- the contextual three-state markers have the intended semantics;
- parent and all children move in the correct native direction with reader
  votes for each admitted claim;
- the compositor still changes the shell/source-ID coordinate after the actual
  model processor;
- no immediate vision-only clinical probe changes beyond a frozen equivalence
  margin.

The balance audit additionally covers total positive-claim count, no-finding,
question type, polarity, uncertainty, anatomy, view and lexical style. Frozen
negative outcomes include a nonclinical cue label, generic affirmative-language
score and a near-zero fingerprint.

Gate B cannot estimate the primary crossover or select claims, shell strength,
layer, prompt or checkpoint.  Failure kills the run design.

### Gate C: locked evaluation

Only after Gate B is sealed are all matched training seeds and the untouched
target split scored.

## Target outcomes

VinDr reader votes define supported states, not a metaphysical visual-clarity
ground truth.  Report 0/3, 1/3, 2/3 and 3/3 separately.  The primary mechanism
is the randomized crossover alignment.  Safety outcomes are:

- false-positive threshold crossings on 0/3 cases;
- reader-distribution-incongruent definite-positive crossings on 1/3 and 2/3
  cases;
- false-negative and omission crossings for fingerprint-negative claims;
- reversal or degradation on clear 3/3 cases;
- reader-distribution Brier/NLL;
- positive-claim count and coverage for OE.

Claims about weak visual evidence require a separate admitted clarity measure;
reader disagreement alone is named reader disagreement.

## Inference

Because `r` was randomized by the experiment, a randomization test is valid.
The primary statistic is the seed-averaged signed alignment between the
observed crossover vector and achieved `g`.  Inference uses:

1. exact permutation under the registered balanced claim-sign randomization;
2. matched seed-triplet random effects or a seed-level randomization summary;
3. target image-cluster bootstrap within claim/vote bin;
4. source PMC-group bootstrap for achieved `g`;
5. leave-one-claim-out and leave-one-seed-out influence analyses.

No more than one frozen claim-level nuisance covariate is allowed in the
primary.  Per-image rows do not inflate the claim or training-run sample size.
Seven unanimous claim signs can license a discovery result; a paper-level claim
requires replication in a second target dataset or a second exact-parent model
organism.

## Natural validity and novelty ceiling

After the randomized mechanism succeeds, test a natural source occurrence
fingerprint in Huatuo/PubMedVision.  The source-only pilot already shows that
some claims are too sparse; only human-admitted, source-count-qualified claims
enter.  Natural results remain correlational unless exact continuation lineage
is established.

The paper contribution is not that styles alter answers.  It is the randomized
crossover showing that medical adaptation can assign clinical semantics to an
empty provenance coordinate, plus evidence that an actual medical VLM exhibits
the corresponding source fingerprint.

## Mitigation

`child-balanced` is the mechanism-matched mitigation: provenance remains
present and equally frequent, but is independent of claim occurrence.  Success
requires both differential and absolute checks:

- A-vs-B claim effects become equivalent to zero;
- error is not merely relocated from A to B or from source to neutral views;
- source-dev quality and clear-case target accuracy decrease by at most one
  percentage point;
- OE positive claim count and matched coverage are preserved.

Only after activation mediation is established may a test-time provenance
residual be attempted.  Universal hedging, refusal, shorter output or claim
deletion is failure.

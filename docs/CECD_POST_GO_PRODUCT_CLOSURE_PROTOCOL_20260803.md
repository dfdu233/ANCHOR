# CECD post-GO product-closure causal and mitigation protocol

**Date:** 2026-08-03  
**Status:** outcome-blind design only; **not authorized and not executable**  
**Scope boundary:** no sealed outcome or human return was inspected, no model was
loaded, and no GPU experiment was started while writing this protocol. It does
not modify or supersede any existing admission, behavioral, three-stage, or
preflight gate.

## 1. One-line question and hard start condition

> Conditional on clinician-admitted render and wording equivalence, does the
> hidden render×wording product component causally worsen reader-grounded loss,
> and can removing that component rescue the target claim more selectively than
> four-call smoothing or generic subspace editing?

Nothing below may execute until the existing **behavioral-GO artifact** passes
for both Huatuo and Hulu and its exact path, content hash, protocol hash, model
hashes, admitted operation IDs, dev/confirmation manifests, and reviewer
provenance have been bound into a new post-GO preflight. A missing, stale,
partial, one-model, diagnostic-only, or historical v3 result is a hard stop.
The post-GO preflight must still require an explicit launch transition; this
document is not that transition.

The intended claim is narrow:

> A clinician-admitted nuisance product can create a reader-harmful hidden
> interaction at a frozen support-to-language location.

It is **not** a universal medical-hallucination method. It cannot repair absent
visual perception, external-knowledge errors, treatment/reasoning mistakes,
unobservable history or prior-study questions, or arbitrary prompts for which
an equivalence orbit cannot be admitted.

## 2. Target-anchored four-cell object

Use one model, image, atomic claim and target wording. For the natural-target
track, assign the untouched deployment query to `h11` before seeing outcomes:

| Cell | Image/render | Wording | Role |
|---|---|---|---|
| `h11` | target/original | target/original | deployment target |
| `h10` | target/original | admitted alternative | wording marginal reference |
| `h01` | admitted alternative | target/original | render marginal reference |
| `h00` | admitted alternative | admitted alternative | double-alternative reference |

This relabeling is fixed in the manifest and never selected by which cell is
wrong. The existing factorial track, in which `h11` is the joint transformed
cell, remains a mechanistic stress test; only the target-anchored track can
support natural-distribution utility.

At frozen layer `l` and a token-aligned claim readout, define

\[
\Delta h_l=h_{11,l}-h_{10,l}-h_{01,l}+h_{00,l},\qquad i_l=\Delta h_l/4.
\]

Three interventions must not be conflated:

1. **Leave-three-fixed target closure**

   \[
   h^{\mathrm{close}}_{11,l}=h_{10,l}+h_{01,l}-h_{00,l}
   =h_{11,l}-\Delta h_l.
   \]

   This makes the four-cell contrast exactly zero while leaving the other
   three mediator values fixed. It subtracts `4i` from `h11`.

2. **Balanced additive projection**

   \[
   h^{\mathrm{bal}}_{11,l}=h_{11,l}-i_l.
   \]

   This is the `h11` cell of the minimum-Euclidean-norm correction that would
   distribute `-s_ij i` over all four cells. It preserves factorial marginal
   means only when all four corrected cells are considered together.

3. **Reader-harm orientation edit**

   \[
   h^{\mathrm{orient}}_{11,l}
   =h_{11,l}-P_{U^{m,f}_l}\Delta h_l.
   \]

   `U` is a low-rank direction learned only on an inner development split after
   residualizing clean polarity, both marginal effects, interaction norm and
   spectrum. Confirmation labels or reader votes may not choose the direction,
   layer, rank, sign, strength or gate. An oracle reader-gradient edit is an
   upper-bound diagnostic and is excluded from every method comparison.

The first is the primary causal probe requested here. The second is a dose and
off-manifold control. The third tests the PAEL orientation hypothesis but has
high novelty collision with HulluEdit/CIPHER/Treble-style subspace editing and
cannot be presented as a new generic editing algorithm.

## 3. What a positive patch can and cannot identify

The hidden states come from mutually exclusive image–prompt inputs. Linear
combination is meaningful only because all four use the same frozen model,
layer, residual-stream coordinate system and atomic claim token. It remains a
cross-world mediator construction, not a randomized biological intervention.

A selective rescue establishes only that:

- the measured product residual at layer `l` is causally usable by the frozen
  downstream network for this target error; and
- replacing it by the admitted additive counterfactual changes reader-grounded
  loss more than matched generic edits.

It does not establish that every coordinate of `Delta h` is pathological, that
the model ought to be internally additive, or that the layer is the unique
origin of the error. The downstream network may recreate the interaction, and
the edited state may be off manifold. Accordingly, record all of:

- pre-hook and immediate post-hook `||Delta h||`;
- output-logit interaction and Brier PAEL_Haar after the remaining layers;
- Mahalanobis distance to unedited dev activations;
- activation norm, cosine to `h11`, output entropy and logit margin;
- next-layer and final-layer re-emergence of the interaction.

If the hook residual is removed but the output interaction reappears before the
next registered checkpoint, the chosen location is not a sufficient causal
bottleneck. If a random or isospectral patch rescues equally, the result is
generic perturbation/calibration, not product closure.

## 4. Token, cache and layer freeze

### CE primary

- Use the last query token immediately before the one-token
  `supported/refuted/undetermined` readout.
- Run the four cells with exactly the frozen prompt template and teacher-forced
  prefix. Capture the same residual-stream site after decoder block `l` and
  before block `l+1`.
- Continue only the edited `h11` branch through the unchanged remaining model.
  The other three branches supply mediators; their KV states must never be
  spliced into the target branch.
- The hook must prove that an identity edit reproduces native logits within the
  frozen fp32 tolerance and that activation restoration survives exceptions.

### Layer selection

Split development clusters once into `dev-localize` and `dev-freeze`.

1. On `dev-localize`, find the earliest layer at which hidden PAEL orientation
   adds information beyond clean margin, both marginals, energy/spectrum,
   entropy and behavioral MMI.
2. On `dev-freeze`, compare only that layer and its immediately adjacent layer;
   freeze one layer per architecture using the causal selectivity score below.
3. Freeze method strength to `{balanced: 1/4, closure: 1}`. No continuous alpha
   search is allowed. Freeze orientation rank from a nested training fold.
4. Apply once to locked confirmation. No confirmation layer, rank, direction,
   seed, threshold or temperature scan is reportable.

Architecture-specific layers are expected. A shared “early layer” claim is not
required.

## 5. Compute accounting: four calls are part of the method

Exact target closure needs four context-conditioned branches. Batching four
cells may reduce wall time but does not make the method single-pass.

Report for every method:

- number of target-model branch calls;
- layer-weighted FLOPs, wall latency at batch 1 and throughput-matched batch;
- peak memory, cache bytes and generated/teacher-forced tokens;
- preprocessing and calibration cost.

If reference branches stop after layer `l`, the ideal decoder cost is roughly

\[
C_{close}/C_{canonical}\approx 1+3l/L,
\]

excluding the shared vision encoder and preprocessing. This can range from
well below four complete forwards to nearly `4x`; it must be measured, not
marketed as “one forward.” Logit averaging and ordinary TTA require complete
logits from all four branches. HulluEdit-style editing is approximately one
model pass and therefore remains a latency Pareto competitor even when its
clinical gain is smaller.

## 6. Locked comparator matrix

All stochastic seeds and hyperparameters are selected on dev and hash-bound.
Every four-call baseline uses the identical images, prompts, preprocessing and
batch schedule as closure.

| Family | Exact arm | Calls | Question answered |
|---|---|---:|---|
| No intervention | canonical/target `h11` | 1 | Native error and cost |
| Calibration | dev-frozen scalar temperature on `h11` | 1 | Is gain ordinary calibration? |
| Four-cell output ensemble | equal-weight **logit** mean; probability mean as sensitivity | 4 full | Is gain just output smoothing? |
| Visual TTA | four clinician-admitted renderings, fixed target wording, equal-logit mean | 4 full | Is generic image augmentation enough? |
| Prompt ensemble | target image, four admitted paraphrases, equal-logit mean | 4 full | Is generic wording averaging enough? |
| Hidden smoothing | full hidden-orbit grand mean; render-only and prompt-only averages | 4 to hook | Does destroying marginals help just as much? |
| Product closure | `h11-Delta h` | 4 to hook | Primary leave-three-fixed causal probe |
| Balanced product removal | `h11-Delta h/4`; plus the complete four-cell projection diagnostic | 4 to hook | Dose/off-manifold and exact factorial projection |
| Equal-norm random | add a dev-seeded random vector with `||delta||=||Delta h||`, orthogonal to `Delta h` where dimension permits | 4 to hook | Any same-energy edit? |
| Norm-only | rescale the unedited `h11` to each edited norm, with no direction change | 1 | Norm artifact? |
| Isospectral stress | reshape the aligned token×hidden or head×width product tensor, rotate left/right within its centered subspaces, preserve singular values, then apply equal closure energy | 4 to hook | Product orientation beyond spectrum? |
| Sign/image control | whole-orbit sign or image-cluster permutation of `Delta h`; never cellwise label-aware permutation | 4 to hook | Instance-specific clinical orientation? |
| Orientation edit | `h11-P_U Delta h`, dev-frozen `U` | 4 to hook | Is only the reader-harmful component necessary? |
| Dynamic subspace baseline | paper-native HulluEdit if compatible; otherwise explicitly named clean-room HulluEdit-style evidence/prior/residual edit | 1 | Can a generic adaptive subspace editor match the gain? |

For a single-token vector, “isospectral” collapses to norm preservation and is
not a distinct control. It is admissible only for a genuinely aligned 2-D
token/head tensor; otherwise report it as unavailable rather than inventing a
reshape after outcomes.

[HulluEdit](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_HulluEdit_Single-Pass_Evidence-Consistent_Subspace_Editing_for_Mitigating_Hallucinations_in_Large_CVPR_2026_paper.html)
uses a sample-adaptive visual-evidence subspace, an orthogonal anti-prior
subspace, adaptive contraction and norm restoration in a single pass. The
audited released engines do not natively cover the present Huatuo/Hulu pair, so
a port must be called **principle-matched**, not paper-native, unless exact
architecture and source conformance is independently established. Generic
projection and minimum-distortion novelty are also bounded by
[LEACE](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html),
activation steering, and the counterfactual intervention space occupied by
[Treble](https://aclanthology.org/2025.findings-emnlp.1000/). Product closure is
therefore evidence for CECD specificity, not standalone operator novelty.

## 7. Outcomes and causal selectivity

The **unique confirmatory endpoint** is the intervention-induced reduction in
reader-distribution **Brier PAEL_Haar**, using the dev-frozen monotone
calibrator. PAEL_Haar is a same-spectrum orientation contrast, not an exact
Haar randomization test; inference comes from the paired whole-cluster
bootstrap. Raw target Brier, soft Bernoulli NLL, clear-case
introduced/repaired/net error, polarity, entropy and output-interaction energy
are prespecified clinical/selectivity guardrails or sensitivities, not extra
confirmatory endpoints. Units are complete admitted orbits; bootstrap clusters
are whole patients when available, otherwise whole images, with identical
multiplier draws across methods and models.

For each method report:

1. target-cell Brier and NLL relative to unedited `h11`;
2. post-edit Brier PAEL_Haar and output interaction;
3. clear-case accuracy and reader-disagreement calibration;
4. claim identity, polarity, certainty and answer-space changes;
5. activation/output distances and the compute ledger;
6. orbit-first estimates followed by the **16 equally weighted** strata
   (`4 findings × 4 reader-vote bins`) within each model; never pool cells or
   models to let a frequent finding or architecture dominate.

Marginal preservation cannot be “proved” merely because a target-only patch
does not execute on the other three cells. Before outcomes, target-anchor each
cell in turn and apply its corresponding leave-three-fixed correction
`delta_ij=-s_ij Delta h`, with `s=(+1,-1,-1,+1)`, as a placebo; also run the
complete balanced correction on all four cells. A CECD-specific intervention
should preferentially rescue the predeclared joint target; comparable changes
in marginal or clean cells imply a generic hidden-state edit.

## 8. Natural-distribution relevance

The factorial grid is an intervention distribution. A result only on the
joint-transformed `h11` does not show that ordinary clinical use benefits.
Three outcome-blind checks are mandatory:

1. **Untouched target anchoring:** evaluate the relabeling in Section 2, where
   `h11` is the original image and original wording. Alternatives exist only to
   estimate its product residual.
2. **Canonical prediction:** on confirmation, dev-frozen hidden PAEL must
   predict canonical reader-grounded error beyond canonical margin, entropy,
   finding, view, both marginal sensitivities, interaction energy/spectrum and
   behavioral synergy.
3. **Operation prevalence:** report how often the admitted rendering change
   corresponds to a real scanner/site/view preprocessing variation and how
   often the wording pair occurs in held-out clinical/user prompts. Synthetic
   style transforms cannot be relabeled as naturally prevalent.

If closure works only when both artificial alternatives replace the original,
retain it as a controlled causal result and reject mitigation/natural-impact
language.

## 9. Fixed-K OE extension without saying less

Free-running four-cell strings are not token-aligned. OE begins only after the
CE mechanism passes and uses a two-stage, fixed-content protocol.

### Stage A: candidate freeze

1. Generate one deterministic draft from the untouched target query only.
2. Normalize it into atomic visual claims
   `(finding, polarity, uncertainty, anatomy, attributes)`.
3. Candidate set is `draft claims ∪ fixed VinDr ontology`. It is frozen before
   any method score.
4. Set `K` to the number of positive image-grounded claims in the unedited
   draft. `K` is shared by every method. `K=0` cases are specificity cases and
   cannot contribute a hallucination-reduction numerator.

### Stage B: aligned scoring and realization

- Teacher-force the same atomic claim template over all four cells. Apply the
  frozen CE layer edit only at the aligned polarity/certainty decision token.
- **Certainty-only arm:** keep claim identities, locations, attributes and `K`
  exactly fixed; only polarity/certainty may change.
- **Evidence-conserving exchange arm:** rank the frozen candidate set by the
  patched support score and select exactly `K` positive claims. One selected
  weak draft claim can only be exchanged for one stronger ontology claim.
- Realize claims with a deterministic template or minimally edit the canonical
  draft. Match number of claims, positive count, slots and maximum length.

No arm may lower `K`, delete difficult claims, emit a global refusal, convert
all findings to negative, or add blanket hedging. Measure positive-content
precision, finding/location/attribute recall, matched-claim coverage, Brier,
claim count, positive count, length, refusal and uncertainty rate. A fixed-K
exchange must improve precision **and** noninferior recall; because fixed `K`
couples precision and selected-set recall, also report recall against the full
reference ontology and errors by attribute/location.

Reports and unrestricted knowledge OE remain outside this protocol. A
teacher-forced ontology experiment is an external-validity bridge, not proof of
general report-generation mitigation.

## 10. Frozen decision rules

There are three separate decisions. Passing a later decision requires all
earlier ones; failure is not repaired by averaging metrics or shortening text.

### A. Causal product-closure GO

Both models must independently satisfy all conditions on locked confirmation:

- immediate post-hook `||Delta h'||/max(||Delta h||, eps) <= 0.05` for closure;
- target-cell reader Brier improves by at least **5% relative**, with paired
  shared-cluster bootstrap 95% CI excluding zero;
- the unique confirmatory quantity, post-edit Brier PAEL_Haar reduction, is at
  least the prospectively frozen **20% relative** mechanism threshold, and its
  paired whole-cluster CI excludes zero;
- clear-case accuracy loss is at most **1 percentage point**, with the
  one-sided 95% bound inside that noninferiority margin;
- closure beats equal-norm random and isospectral stress edits by at least half
  of its own Brier gain, with paired CI excluding zero;
- joint-target gain exceeds each marginal-cell placebo gain; the paired
  joint-minus-largest-placebo CI excludes zero;
- direction holds in at least **3 of 4 findings**, with no finding showing a
  relative Brier worsening of 5% or more;
- final output-logit interaction RMS falls by at least **25% relative**, with
  its paired CI excluding zero.

Failure of any item terminates the claim that the hidden product component is a
selective causal mediator. A positive behavioral PAEL result may still be
reported without a hidden causal story.

### B. Mitigation utility GO

In addition to A:

- closure beats the best of four-cell logit averaging, visual TTA and prompt
  ensembling by at least **5% relative Brier**, with paired CI excluding zero;
- the gain survives dev-frozen temperature calibration and is not reproduced
  by full-hidden-orbit smoothing;
- the untouched target-anchored track improves by at least **5% relative
  Brier**, with CI excluding zero;
- activation Mahalanobis distance stays below the dev-frozen 99th percentile
  of admitted unedited cells, or the method is explicitly classified as an
  off-manifold causal probe;
- both architectures pass independently at frozen, possibly different layers;
- latency/FLOP/memory results are reported. If a one-pass HulluEdit-style edit
  is within a **1% relative Brier noninferiority margin** while preserving clear
  cases, four-call closure loses the deployment claim even if causal decision A
  remains positive.

If logit averaging or TTA matches closure, the parsimonious mechanism is
ensembling/smoothing. If only the dev-learned orientation subspace works, the
method contribution collapses toward generic subspace editing; CECD may remain
the discovery/evaluation setting, not a new editor.

### C. Fixed-K OE GO

After A and B only:

- positive-content hallucination falls by at least **20% relative** at exactly
  matched `K`, with patient/image-cluster CI excluding zero;
- full-reference finding recall is noninferior within **1 percentage point**,
  and location/attribute omission does not increase by 1 point or more;
- answer claim count and positive count are exact matches by construction;
  median length differs by no more than one templating token, and refusal does
  not increase;
- reader-distribution Brier improves by at least **5% relative**;
- both models and at least 3/4 findings agree; automatic judging is not the sole
  truth source.

Any gain caused by smaller `K`, shorter output, more refusal, blanket hedging or
uniform negativity is an automatic OE failure.

## 11. Stop table and permitted conclusions

| Observation | Required conclusion |
|---|---|
| Behavioral-GO absent | Do not execute this protocol |
| Hidden closure fails A | Product orientation is behavioral, not a localized causal mediator |
| Random/isospectral edit matches closure | Generic energy/perturbation effect |
| Four-cell logits/TTA match closure | Four-call smoothing, no mechanism-specific mitigation |
| Only balanced `Delta/4` works | Full closure is off-manifold/over-correction; retain dose result only |
| Only full `Delta` works | Other three cells act as valid fixed references; still test natural anchoring |
| HulluEdit-style one-pass is noninferior | No deployment case for four-call closure |
| Natural target fails but transformed joint passes | Controlled causal probe only |
| Fixed-K OE loses recall | Content exchange failed; no hallucination-mitigation claim |
| Both models pass A and B, OE passes C | Narrow product-closure mitigation for admitted image-grounded claims; still not universal medical hallucination mitigation |

## 12. Minimal execution order after authorization

1. Bind behavioral-GO and all source/input hashes into a new non-overwriting
   preflight.
2. Run CPU-only hook/identity synthetic conformance.
3. Freeze per-model layer, optional orientation subspace, temperatures, seeds
   and bootstrap multipliers on nested dev only.
4. Run a small operational pilot that cannot change scientific choices.
5. Execute locked CE confirmation once for all comparator arms from shared raw
   four-cell caches.
6. Decide A, then B. Do not launch OE if either fails.
7. If authorized by A+B, execute fixed-K ontology OE and decide C.

This sequence is intentionally smaller than a general mitigation campaign. Its
value is a hard causal discrimination: either a clinician-admitted product
residual is selectively actionable, or CECD stops at a credible behavioral
phenomenon.

# Specificity Ratchet: conditional ICLR paper skeleton

> **METHOD-NOVELTY DOWNGRADED 2026-08-03.**  The later fatal-collision
> recheck in `docs/SPECIFICITY_RATCHET_FATAL_COLLISION_RECHECK_20260803.md`
> found that Hierarchical Selective Classification (NeurIPS 2024) directly
> occupies uncertainty-driven ancestor retreat, while CEBC, ZINA and CoEV
> occupy minimal evidence editing and claim correction.  Any ancestor backoff
> below is a causal readout/mitigation protocol and required baseline, not an
> algorithmic contribution.  The only conditionally open contribution is the
> physician-grounded native parent-to-descendant late-crossing mechanism.

> **MEASUREMENT-NOVELTY AMENDMENT 2026-08-03 (pre-outcome).**  A late increase
> of added-constraint token logits is not, by itself, a parent-to-child state
> transition. VISTA/Hidden Life of Tokens, CEI/Inject to Heal, decoder
> overthinking, VLI and related work already make generic visual-loss,
> commitment-depth and layer-intervention claims. The paper may use the word
> `crossing` only when the parent state is directly observed before the added
> constraint, parent identity/support remains stable, and the constraint state
> alone reverses. Otherwise the result is `late constraint amplification` and
> the mechanism contribution is rejected.

Date: 2026-08-02
Paper type: **New Problem/Setting**, with a conditional mechanism only
Current status: **scientifically frozen, empirically unadmitted; no positive result may be written**

## One-line paper test

The paper is worth writing only if it can show that an open-ended medical VLM
spontaneously generates a physician-rejected descendant of a supported claim,
that the added constraint crosses late for a reason not explained by perception,
text frequency, prompt demand or length, and that a fixed-claim-count causal
intervention removes the descendant without damaging supported specificity.

This is deliberately narrower than “medical VLM hallucination.” It is also
stronger than a new detector: the paper must identify a state transition and
separate it causally from simpler alternatives.

## Thinking template

| Cell | Frozen content |
|---|---|
| Research background | Medical VLMs can generate fluent fine-grained details that are not supported by the supplied image. Recent work localizes or edits hallucinated spans ([ZINA](https://arxiv.org/abs/2506.13130)), constrains externally unsupported entities ([CEBC](https://aclanthology.org/2026.acl-long.2142/)), verifies medical entities counterfactually ([CounterVHD](https://arxiv.org/abs/2606.28520)), or trains on fine-grained negative queries ([FINER](https://arxiv.org/abs/2603.17662)). In clinical use, however, backing off from “small left pleural effusion” to “pleural effusion” is materially different from deleting the finding: the parent may be supported while only the descendant constraint is not. |
| Limitation 1 | Existing span/entity evaluation treats a clinical claim as flat. It does not establish that a rejected child entails an independently supported parent plus exactly one physician-admitted constraint. |
| Limitation 2 | Final-output errors do not distinguish spontaneous specificity escalation from a static visual mistake, a post-hoc synthetic query effect, lexical frequency, prompt compliance or longer text. |
| Limitation 3 | Many mitigation gains can be purchased by fewer claims, shorter answers, blanket hedging or refusal, so reduced hallucination does not imply a more faithful clinical answer. |
| Our goal | Define and causally test whether medical VLM generation crosses from a visually supported clinical parent to an unsupported descendant constraint while the parent remains available. |
| Challenge 1 | Clinical nesting and support are semantic constructs: an automatic parser or the model output itself cannot admit the edge or define visual truth. |
| Challenge 2 | The spontaneous generated child must be linked to an exact layerwise estimand while holding image and question fixed and separating token count, lexical likelihood, prompt demand and static perception. |
| Challenge 3 | Any correction must act on only the added constraint while preserving parent identity, polarity, claim count, response length and legitimately supported children. |
| Methodology topic sentence | Specificity Ratchet couples physician-admitted nested claims with exact observed-generation traces; fixed-K ancestor retreat is used only as a content-preserving causal readout after validation. |
| Module 1 | **Specificity Edge Admission.** Two blinded physicians and a blinded physician adjudicator admit adjacency, parent/child support and evidence-source state; the compiler groups images and refuses incomplete truth. |
| Module 2 | **Spontaneous Parent-State Replay.** Replay the model's complete frozen visible OE answer and require an identifiable parent-before-constraint state (or a separately preregistered, validated parent-state readout). Measure parent preservation and added-constraint reversal jointly. Relative-position-matched tokens, at least two same-modality/anatomy different-case image swaps with exactly equal visual-token length, and text-only remain controls; they cannot substitute for observing the parent state. |
| Module 3 | **Selective Fixed-K Intervention.** If and only if Module 2 passes, a causal patch or projection backs an unsupported child off one-for-one to its supported ancestor, with no deletion, new hedge, refusal or claim-count reduction. |

## Contribution slots and evidence status

1. **Problem formulation — partially delivered.** The project defines
   spontaneous support-to-specificity escalation at a physician-admitted
   parent/child edge. The novelty claim is bounded to this output-side causal
   estimand, not fine-grained hallucination generally. This maps to Sections 1–2.

2. **Construct and measurement — engineering delivered, scientific evidence
   pending.** Blinded review packs, fail-closed validation, an image-disjoint
   full-visible-answer manifest compiler, direct native-generation identity
   canary, resumable replay contract and dev-frozen case-clustered conjunctive
   analyzer exist. No clinical edge is admitted yet. This maps
   to Section 3.

3. **Mechanistic finding — NOT MEASURED.** This contribution exists only if the
   held-out error-versus-supported-control late-minus-early contrast survives
   all nuisance and causal controls in at least two model families and three
   edge types. This would map to Sections 4–5.

4. **Causal readout — NOT AUTHORIZED AND NOT METHOD-NOVEL.** Fixed-K
   nearest-ancestor projection may validate content preservation only after
   selective causal patching passes. It must treat HSC as the first baseline
   and compare with CEBC/ZINA/CoEV-style correction at matched claim count and
   coverage. Any utility gain supports Contribution 3; it is not a separate
   algorithm contribution. This would map to Section 6.

The abstract and introduction must currently mention only Contributions 1–2 as
designs, not results. Contributions 3–4 remain empty evidence slots.

## Central estimand and alternatives

For a complete visible model answer \(Y\), exact added-constraint tokens
\(T_\Delta\), relative-position-matched non-constraint tokens \(T_M\), and an
independently validated parent-state readout \(P_l\), let

\[
d_l(I)=\operatorname{mean}_{t\in T_\Delta}\log p_l(Y_t\mid I,q,Y_{<t})
      -\operatorname{mean}_{t\in T_M}\log p_l(Y_t\mid I,q,Y_{<t}),
\]

and define the image-specific control

\[
g_l=d_l(I_{own})-\frac{1}{K}\sum_{k=1}^{K}d_l(I_{swap,k}),\qquad K\ge2.
\]

The load-bearing crossing estimand is conjunctive, not \(g_l\) alone:

\[
P_{early}>t_P,\qquad P_{late}\ge P_{early}-\epsilon_P,\qquad
C_{early}<t_C,\qquad C_{late}>t_C,
\]

where \(C_l\) is the added-constraint commitment after subtracting the matched
token and swap controls. The cleanest admissible row has a semantically
complete parent already present in the native answer before the constraint
span. Any learned/probed alternative to that prefix-realizable state must be
frozen on dev, validate parent identity rather than generic truthfulness, and
beat a polarity/lexical-only readout on held-out cases. If no such parent-state
measurement is available, \(d_l\) and \(g_l\) are retained only as
amplification diagnostics and cannot authorize G3/G4.

Every swap is from a different case in the same split, modality and anatomy
stratum and must have identical native visual-token length. The paper test is
conjunctive: supported controls must have an early image-specific advantage in
\(g\); errors must show a larger own-image \(d_{late}-d_{early}\) shift than
controls; the error-selective late shift must remain positive under both frozen
image swaps; and the lower bootstrap bound for the swap/own transition ratio
must exceed 0.50. Thus most of the late escalation must survive removal of the
own image, while early visual support still distinguishes valid specificity.
Tests are case-clustered under a pre-data-frozen nuisance specification and
must pass independently on both label-blind frozen splits. The four
interpretable outcomes are:

| Observed pattern | Interpretation | Paper decision |
|---|---|---|
| Error/control already separate early; no selective late shift | Static perceptual or representation error | Kill the ratchet mechanism; do not relabel it |
| Both roles show the same late shift, reproduced text-only | Lexical frequency/target construction | Kill |
| Only errors show held-out late shift, but patch harms supported children equally | Correlational detector, not causal ratchet | Kill the mechanism/method claim |
| Error-only late shift plus selective causal rescue, replicated | Specificity Ratchet supported | Advance to fixed-K mitigation |

## Experimental gates

### G0 — construct admission

- Two independent physicians, distinct IDs, blinded adjudication and
  attestations all pass.
- Report edge-admission rate, role counts, edge-type counts, image counts,
  Cohen/Fleiss agreement with clustered uncertainty, and disagreement reasons.
- Require both supported-child controls and observable parent-only errors in an
  image-disjoint dev/test split. No count threshold may be relaxed after seeing
  model traces.
- Before any GPU trace, issue a construct-prevalence certificate showing that
  both splits contain at least 10 repeated semantic constraint blocks across at
  least three edge types and that the retained rows expose a valid
  parent-before-constraint state. The current 70-case pack cannot meet the
  already-frozen 10-block floor from lexical ceilings alone and therefore can
  authorize only a bounded construct pilot.
- Failure means **kill the direction**, not replace physicians with an LLM.

### G1 — exact runtime conformance

- The active Huatuo bridge scores only complete visible answers. The obsolete
  isolated parent/child CLI is executable-hard-refused.
- After physician admission, deterministically regenerate one frozen dev case
  under its original greedy-512 contract, directly capture
  `output.sequences`, and require exact decoded visible-text identity. A failed
  sidecar is frozen; no substitute answer may be generated.
- Contextual token mapping must pass ASCII, UTF-8, repeated constraints,
  punctuation, leading spaces and an intentionally unidentifiable merged-token
  rejection.
- Final-layer gold probabilities must numerically match ordinary model logits.
- Own-image, every swap-image and text-only trace must have identical complete
  answer token IDs, offsets, layer IDs and template ID.
- Any conformance failure blocks scientific scoring; it is not an exclusion
  that can be hidden in coverage.

### G2 — held-out mechanism

- Select layers and residualizer on dev only; evaluate test once.
- Primary bootstrap clusters by case/image.
- Adjust exact constraint/target token counts, text-only NLL, edge type,
  modality, anatomy, prompt-request status and answer length.
- Negative controls: length-exact role permutation, random equal-norm
  directions, at least two exact-visual-length image swaps and secondary
  text-only traces.
- Report parent-state stability and constraint-state reversal separately.
  Constraint-token late gain without stable earlier parent representation is a
  generic commitment/amplification result and fails this gate.
- Report effect sizes and confidence intervals per model and edge type; a pooled
  significant coefficient cannot conceal opposite model-family effects.

### G3 — causal and replication gate

- A parent-directed patch/projection reduces unsupported child commitment while
  changing supported-child performance and parent retention by at most 1 pp.
- Replicate direction and selectivity in at least two model families and three
  physician-admitted edge types.
- A Huatuo-only result is a bounded pilot, not the paper's main conclusion.

### G4 — mitigation without exchange

- Replace exactly one unsupported descendant with exactly one supported
  ancestor: \(K_{after}=K_{before}\).
- Primary outcomes: physician-grounded positive-content precision, parent
  retention and unsupported-descendant rate at matched claim count/coverage.
- Mandatory diagnostics: answer length, refusal, hedging, omission, supported
  specificity retention and clinical usefulness.
- Compare to equal-count random ancestor replacement, lexical-frequency-matched
  replacement, length controls, abstention, self-consistency and identity-
  qualified decoding baselines. Lexical F1/RadGraph/GREEN remain auxiliary.

## Minimal figure and table program

- **Figure 1 — the missing clinical operation.** One image and OE answer showing
  supported parent, unsupported child, and one-for-one backoff. Beside it, show
  why deletion and hedging change content while nearest-ancestor projection
  does not.
- **Figure 2 — decisive mechanism plot.** Full-answer constraint-versus-matched
  token own-image-minus-swap contrast across normalized decoder depth for
  supported-child versus parent-only cases, with text-only and random-direction
  controls. This is the paper's load-bearing
  figure; without a clean separation there is no mechanism paper.
- **Figure 3 — causal selectivity.** Unsupported-child suppression versus
  supported-child damage and parent retention under the frozen intervention.
- **Table 1 — construct quality.** Physician admission/agreement, role and edge
  counts, exclusions and image-disjoint split statistics.
- **Table 2 — mechanism and controls.** Per-model/per-edge-type held-out effect,
  clustered CI, nuisance-adjusted effect and all negative controls.
- **Table 3 — no-exchange mitigation.** Clinical outcomes at fixed K and matched
  coverage, with length/refusal/omission diagnostics.

Large method-zoo or dataset-average tables are appendix material. The main
paper should be organized around one transition, one separating intervention
and one no-exchange result.

## Section skeleton

1. **Introduction:** the clinical difference between removing a finding and
   backing off an unsupported modifier; precise contribution ceiling.
2. **Specificity as a support hierarchy:** observability, parent/child edge,
   spontaneous occurrence and exclusions.
3. **Physician-admitted Specificity Ratchet protocol**
   - 3.1 Blinded edge/support/source admission
   - 3.2 Native-generation identity anchoring and grouped splits
   - 3.3 Full-visible-answer constraint replay and image-swap controls
4. **Does specificity ratchet across decoder depth?**
   - 4.1 Huatuo dev/test screen
   - 4.2 Static-perception, lexical, prompt and length alternatives
   - 4.3 Cross-family and edge-type heterogeneity
5. **Causal localization and selectivity**
   - 5.1 Parent-directed intervention
   - 5.2 Random/equal-norm and supported-child controls
6. **Conditional fixed-K correction**
   - 6.1 Nearest-supported-ancestor projection
   - 6.2 Matched-content clinical evaluation and baselines
7. **Scope, failure modes and evidence-source boundary**

Sections 4–6 must remain unwritten as affirmative prose until their gates pass.

## Self-consistency audit

| Check | Result | Reason |
|---|---|---|
| Limitations → Goal | PASS | The goal introduces the missing nested support unit, spontaneous transition test and content-preserving correction target. |
| Goal → Challenges | PASS | Clinical admission, temporal/causal separation and no-exchange correction arise directly from testing the goal. |
| Challenges → Modules | PASS | The three challenges map one-to-one to admission, trace and intervention modules. |
| Modules → Contributions | CONDITIONAL PASS | Modules exist as protocols/code, but mechanism and mitigation contributions are explicitly withheld until evidence exists. |

### Evidence gaps

- **CRITICAL:** physician review/adjudication has not occurred; there are zero
  admitted scientific edges today.
- **CRITICAL:** no real-model Specificity Ratchet trace, held-out mechanism
  effect, causal intervention or second-family replication exists.
- **MAJOR:** Huatuo's full-answer bridge, engineering canary and complete
  per-case native-ID capture are code-complete but still need the
  physician-gated real-model run. A one-case pass alone cannot authorize
  scientific replay.
- **MAJOR:** Hulu helper plumbing is processor-audited but scientifically
  disabled. Cross-family replication requires Hulu's own spontaneous
  full-answer substrate and fresh physician admission; Huatuo answers cannot
  be replayed as Hulu-native evidence.
- **MINOR:** the final paper title should be chosen only after the dominant
  admitted edge types and mechanism pattern are known.

## Oral-level decision rule

An ICLR Oral-quality submission requires a visually compelling state
transition, a separating causal intervention, cross-family replication and a
strict fixed-content improvement. A statistically significant final-layer
detector, a Huatuo-only observation, or a gain from shorter answers is not
close enough and should not be packaged as such.

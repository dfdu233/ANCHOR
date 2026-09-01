# CECD post-collision idea evaluation and paper skeleton

**Freeze date:** 2026-08-03

**Scope:** outcome-blind planning; no sealed human decision, model output, or evaluation result was read.

**Candidate:** fusion-induced clinical orientation selection, read out by a
clinician-admitted render × wording product orbit (CECD).
**Status:** **Conditional Keep — PAEL is an estimand, not a contribution; the
paper survives only if a fusion-to-decoder orientation mechanism and selective
causal rescue are confirmed. It is not currently ICLR-oral-ready.**

## 1. Idea-evaluator verdict

### First impression

- **Paper type:** conditional Mechanism Paper. Without layerwise
  energy–orientation dissociation and causal rescue, it downgrades to a narrow
  clinical measurement result rather than a new problem, metric, decoding
  method, or benchmark.
- **One-sentence story:** A medical VLM may preserve the amount and spectrum of
  render × wording interaction while cross-modal fusion rotates that interaction
  toward reader-grounded clinical loss; CECD asks when this harmful orientation
  emerges and whether it can be removed without changing either marginal.

### Fatal-flaws audit

| # | Flaw | Severity | Concrete defense |
|---|---|---:|---|
| 1 | **F1, high-severity novelty collision.** MetaRA already combines paraphrases with benign/style/background image transformations; composite metamorphic-relation work already shows that individually tested relations can reveal additional failures when composed; Semantic Robustness Certification already occupies text-proxy semantic planes and norm-preserving rotations. PAEL therefore cannot be a standalone metric or method contribution. | MAJOR | Use PAEL only as a frozen reader-grounded readout. The sole headline candidate is fusion-induced clinical orientation selection: interaction energy/spectrum pre-exists, harmful reader-loss alignment jumps at a model-specific fusion-to-decoder transition, and an upstream spectrum/norm/marginal-preserving intervention selectively rescues it. If generic joint failure plus semantic-boundary proximity absorbs PAEL, or the causal rescue fails, F1 becomes CRITICAL and the paper stops. |
| 2 | **F7, independent clinical evidence is on the critical path.** Role-isolated equivalence review, independent adjudication, and a joint-product human control cannot be replaced by automatic judges or author-generated labels. Axis-wise admission alone cannot establish that the product defect is model-specific. | MAJOR | Keep all inference fail-closed until independent returns and adjudication are hash-bound. For the model-specific claim, use the separately frozen recall-free control: preferably 240 unique image-claims, four fixed clinicians, and a Latin-balanced one-cell-per-clinician-per-claim allocation. A 160-claim minimum requires outcome-blind variance support; a two-clinician between-image fallback forces a claim downgrade. If this capacity is not secured by the preregistered execution deadline, escalate F7 to CRITICAL and do not substitute an automatic judge. |

No CRITICAL flaw is declared yet: the defining product interaction is untested rather than refuted. A failed locked CECD gate immediately turns the candidate into a negative result and terminates this paper framing.

#### Closest-work boundary checked independently

| Work | Occupied object | Surviving CECD axis |
|---|---|---|
| [MetaRA](https://arxiv.org/abs/2605.19307) (Xu et al., arXiv 2026) | Joint VQA metamorphic tests using question paraphrases and benign/local/style/background image transformations | It lacks the complete factorial marginals, clinician admission, reader-distribution proper loss and layerwise causal mechanism. It kills any “first joint image × wording robustness test” claim. |
| [How Composite Metamorphic Relations Enhance Test Effectiveness](https://doi.org/10.1109/TSE.2026.3675285) (Wu et al., IEEE TSE 2026) | Composite versus individual metamorphic relations and additional failure revelation | It kills the high-level “individually safe transformations need not compose” novelty; only the clinical orientation mechanism can remain. |
| [Semantic Robustness Certification for Vision-Language Models](https://arxiv.org/abs/2606.18839) (Yang et al., ICML 2026) | Text-proxy semantic plane, norm-preserving embedding rotation and prediction-invariant intervals | It does not model a render × user-wording factorial product or reader loss, but kills semantic-plane/rotation novelty and motivates a frozen decision-boundary-proximity control. |
| [PSF-Med](https://arxiv.org/abs/2602.21428) (Sadanandan & Behzadan, arXiv 2026) | Clinically equivalent paraphrases, medical-VLM answer flips, SAE features and causal clamping | It occupies the wording-only clinical sensitivity and representation intervention; CECD must add the product-specific, spectrum-matched cross-modal path. |
| [Medical Context Distorts Decisions in Clinical Vision Language Models](https://arxiv.org/abs/2605.17436) (Restrepo et al., 2026) | Medical image–text alignment, irrelevant context, and semantically equivalent prompt sensitivity | It does not require both axes to be independently clinically admitted or estimate a product-only clinical-error residual beyond both marginals. |
| [When Background Matters: Breaking Medical Vision Language Models by Transferable Attack](https://aclanthology.org/2026.acl-long.1768/) (Ghosh et al., ACL 2026) | Coordinated image–text adversarial perturbation and attention distraction | Its goal is targeted diagnostic misdirection; it does not establish two meaning-preserving equivalence axes or test whether their composition alone breaks invariance. This is the closest newly surfaced product-axis collision. |
| [Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](https://aclanthology.org/2026.acl-long.1941/) (Rudman et al., ACL 2026) | False-presupposition prompts and model-specific prompt-copy heads | CECD's wording operation must preserve proposition and speech act; prompt-head effects are a mandatory alternative, not CECD novelty. |
| [Mechanistic Analysis and Inference-Time Control of Modality Conflict in VLMs](https://mechinterpworkshop.com/posters/virtual/) (He et al., ICML 2026 Mechanistic Interpretability Workshop) | Controlled image–statement conflict, causal characterization of architecture-dependent modality preference, and inference-time latent control | It occupies generic modality-conflict localization and steering. CECD must show incremental reader-loss orientation under two clinician-admitted meaning-preserving axes and beat a dev-fitted explicit-conflict direction plus matched steering at fixed spectrum, norm and marginals. |
| [HulluEdit](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_HulluEdit_Single-Pass_Evidence-Consistent_Subspace_Editing_for_Mitigating_Hallucinations_in_Large_CVPR_2026_paper.html) (Lin et al., CVPR 2026) | Evidence/prior/residual subspaces and orthogonal activation editing | It kills subspace-projection mitigation novelty but does not study a clinician-admitted render–wording product orbit. |

No exact duplicate of the full conjunction was retrieved; that absence does not
prove novelty. The differentiating axis is no longer composition failure or
PAEL itself. It is the **fusion-induced selection and causal removal of a
reader-harmful interaction orientation at matched spectrum and marginals**.

### Lifecycle and capability match

| Aspect | Available evidence | Assessment |
|---|---|---|
| Idea category | Frontier exploration plus data-intensive clinical validation | 6–9 month lifecycle; novelty is moving quickly, so the mechanism decision must happen early. |
| Effective hours | Not specified | Unknown; no optimistic estimate is assumed. Scope is therefore frozen to two primary medical VLMs and the admitted VinDr product orbit before any broad expansion. |
| Engineering/data capability | VinDr is local; Huatuo/Hulu native adapters, three-stage execution, fixed-​K listing runtime, provenance and persistent monitors exist | Green for implementation and reproducibility. |
| Domain capability | Independent clinical-equivalence reviewers/adjudicators, a fixed four-clinician joint-product panel, plus any post-gate listing review capacity, are not yet secured | Yellow/red critical-path risk until both axis admission and the recall-free product-control roles are secured. A fixed panel supports image-level inference only, not clinician-population generalization. |
| Fit | Strong engineering fit; unresolved clinical-access fit | **Yellow.** The mechanism screen is feasible; an ICLR-scale clinical conclusion is not yet secured. |

### Five-dimension radar

Scores start at 5 and move only for a stated mechanism or observed resource. No efficacy number is invented.

| Dimension | Score | Evidence | Lift suggestion |
|---|---:|---|---|
| Higher | 6 | Mechanism-based, not yet confirmed: a product-orbit test can identify failures hidden by one-factor robustness averages, but it is not itself a guaranteed mitigation gain. | Make reader-grounded clinical error, not generic consistency, the primary endpoint; require locked CE confirmation before any fixed-​K listing transfer. |
| Faster | 3 | CECD requires a four-cell orbit and independent adjudication; it does not primarily reduce inference cost. Shared-cache execution only limits overhead. | Treat compute sharing as engineering, not a paper contribution. |
| Stronger | 7 | Mechanism-based, not yet confirmed: the design can reveal a compositional robustness defect that one-factor tests miss, but it neither demonstrates improved model robustness nor yet establishes that the defect exists. | Require energy–orientation dissociation and selective upstream rescue after MetaRA/composite-MR, boundary-proximity, generic two-axis, behavioral, context and late-override controls in both models. |
| Cheaper | 3 | Human clinical admission, the preferred 240-image/four-clinician product control, and final physician review add cost; the design intentionally refuses cheap automatic truth. | Use Latin-balanced one-cell-per-clinician allocation to prevent recall without a repeated full 5 x 3 review. Use 160 images only if an outcome-blind variance basis establishes adequate power. Do not claim cost reduction. |
| Broader | 5 | A shared claim schema makes CE-to-listing transfer testable, but no cross-task or cross-domain success exists and unrestricted OE/report generation is outside the frozen mechanism. | Use one closed-ontology listing task only as external-validity evidence after CECD passes; defer reports and external-knowledge claims. |

The defensible thesis axes are **Stronger** and, secondarily, **Higher**. Broader is a prospective validation axis, not an earned contribution; Faster and Cheaper must not appear in the contribution list.
With no dimension at 8 and no scientific result yet, the score supports only
**Conditional Keep**, not Accept or Strong Accept.

### Paradigm-shift probe

| Probe | Verdict | Rationale |
|---|---|---|
| First Principles | Partial | It challenges independent-to-joint robustness extrapolation in this clinical setting, but factorial non-composition is not itself a new general principle. |
| Elephant in the Room | Partial | Style/prompt instability is widely visible, but clinically admitted cross-factor interaction is usually averaged away because it is expensive to validate. |
| Technology Cycle | No | Modern medical VLMs make the failure salient, but the factorial identification and clinical-equivalence requirement were technically possible before the current model cycle. |
| Hamming's Rule | Partial | Success would change how medical VLM robustness is evaluated, but only if the defect is common, clinical, and survives simpler utilization explanations. |

**Disruptive potential:** incremental with disruptive seeds (3/8), conditional on a prevalent, clinically consequential residual rather than a renamed prompt-bias or standard factorial result.

### Feasibility

| Risk | Level | Mitigation |
|---|---:|---|
| Compute | Medium | Two local models and one GPU are sufficient; shared locks, atomic shards, staged power, and detached continuation are already implemented. Run no broad method grid before the locked gate. |
| Data | Low for VinDr; high for human evidence | VinDr is local and frozen. Independent reviewer/adjudicator availability and the fixed four-clinician product panel must be secured; automatic labelers cannot substitute. If only two clinicians are available, the randomized between-image fallback changes the estimand and narrows the paper claim. |
| Engineering | Medium | The execution DAG is static-ready and tested, but real-model canaries and future artifact drift remain fail-closed. Keep source-hash and recovery audits. |
| Timeline | High | 2026 literature is rapidly closing neighboring claims. Make CECD GO/NO-GO the first scientific decision; do not wait for all OE baselines before killing a failed mechanism. |

### Final reviewer verdict

**Conditional Keep — execute the locked product orbit as a mechanism admission
test, not as evidence that PAEL or composition failure is itself novel.**

Top actions:

1. Obtain genuine clinical-equivalence returns and execute the frozen three-stage CECD gate without changing thresholds; independently complete the recall-free human product control before using “model-specific” language.
2. If and only if CECD passes, require MetaRA/composite-MR joint-failure and
   semantic-boundary-proximity controls before any layer claim; then test an
   architecture-specific energy–orientation transition and an upstream
   spectrum/norm/marginal-preserving causal rescue.
3. Promote beyond the binary mechanism scope only if the causal rescue lowers
   PAEL by at least 20% versus matched random/ispectral controls with no more
   than 1pp clear-case loss in both models, and fixed-​K listing transfers
   without omission, brevity, negative-answer, hedge, or refusal exchange.

## 2. Tech-paper-template skeleton

### Paper-type positioning

- **Type:** conditional Mechanism Paper.
- **Rationale:** Composition failure, multimodal metamorphic testing, semantic
  rotation, interaction decomposition and orbit risk are already occupied. The
  only oral-level candidate is a causal account of how fusion selects a
  reader-harmful orientation at fixed interaction geometry.

### Thinking template

| Stage | Frozen candidate content |
|---|---|
| Research background | Medical VLM hallucination work now covers visual-context utilization (Seeing or Knowing?), late textual override (CALRD), confidence–evidence detection (CEBaG), reader disagreement (CheXthought), prompt/system-token mechanisms, and coordinated medical image–text attacks (MedFocusLeak). What remains unresolved is whether two operations that clinicians independently judge meaning-preserving can fail only when composed. The external beneficiary is a clinician or patient whose decision depends on a claim remaining invariant under clinically irrelevant rendering and wording variation. |
| Limitation 1 | Existing prompt-, image-, and layer-intervention work identifies single-factor sensitivity or generic utilization failure, but does not test whether two independently clinician-admitted equivalence operations compose without a reader-grounded clinical error. |
| Limitation 2 | A nominal render-by-wording interaction is uninterpretable if either operation changes visual support, proposition, speech act, certainty demand, answer space, or output grammar; automatic similarity and author judgement cannot establish that equivalence. |
| Limitation 3 | A nonzero mixed derivative can still be explained by marginal sensitivity, generic two-axis instability, behavioral synergy, image-ignoring perception, late textual override, prompt-copy/system-token effects, or probe geometry. |
| Key Idea / Our Goal | **Test whether cross-modal fusion selects a reader-harmful orientation of an already present render–wording interaction, and causally remove that orientation upstream while preserving its spectrum, activation norm and both marginals.** PAEL is only the frozen readout. |
| Challenge 1 | The two operations must be independently admitted as preserving visual support, proposition, speech act, certainty demand, answer space, and output grammar before any model output is eligible; otherwise the interaction is an annotation artifact. |
| Challenge 2 | The product-only effect must be estimated without fitting on the locked confirmation set or algebraically reconstructing its own error label, while separating clinical orientation from both main effects and a generic interaction with the same spectrum. |
| Challenge 3 | Even positive PAEL may be ordinary composite metamorphic fragility or semantic-boundary proximity. A mechanism claim requires a layerwise dissociation in which same-spectrum energy precedes reader-loss alignment, plus an upstream causal intervention that outperforms random/ispectral patches while simpler perception/utilization and prompt mechanisms fail to absorb it. |
| Methodology topic sentence | **CECD uses an admitted product orbit to test whether cross-modal fusion selects—and an upstream intervention can remove—a reader-harmful interaction orientation without changing its spectrum or marginals.** |
| Module A — Clinical Product Admission | Four role-isolated reviews plus independent adjudication freeze exact image-render and wording sets before model scoring. A separate recall-free control uses preferably 240 unique image-claims and four fixed clinicians, Latin-balancing the four product cells so each clinician sees each claim once; direct probability elicitation defines reader-distribution Brier loss, and signed plus non-cancelling gates prevent human harm from averaging away. This fixed-panel control does not imply reader-population generalization. |
| Module B — Locked Product Readout | A pilot→dev-fit→image-disjoint confirmation design uses 16-stratum reader-distribution Brier PAEL as a non-novel readout of observed orientation beyond an isospectral Haar reference; whole-image bootstrap and a two-model conjunction replace the target-coupled v3 claim. MetaRA-style joint failure and semantic-boundary proximity are mandatory alternatives. |
| Module C — Causal Orientation Mechanism | Layerwise energy and reader-loss alignment test whether interaction spectrum precedes harmful orientation at a model-specific fusion-to-decoder transition. An upstream spectrum/norm/marginal-preserving orientation intervention must selectively reduce PAEL; additive-state reconstruction, explicit modality-conflict directions and matched steering, prompt-head/System-PIH, HALP, reader-alias, random/ispectral and human-product controls test specificity. |
| Contribution 1 | **Conditional on all gates:** evidence that a generative medical VLM selects a reader-harmful orientation of an existing cross-modal product interaction, rather than merely creating more interaction energy (Sections 3–5). |
| Contribution 2 | **Conditional on causal rescue:** localization of the architecture-specific fusion-to-decoder transition and a spectrum/norm/marginal-preserving intervention that reduces clinical orientation without changing clean polarity or either single-axis effect (Sections 5–6). |
| Contribution 3 | The clinician admission, fixed-panel product control and reader-distribution PAEL protocol make those mechanism claims identifiable; PAEL, factorial decomposition, provenance and the protocol are explicitly not claimed as standalone metric, benchmark or systems novelty (Sections 2–4). |

### Methodology outline

#### 3.1 Clinical Product Admission

Opening: A product interaction is uninterpretable unless each axis is independently proven clinically equivalent before any model outcome is seen.

- Freeze reader-grounded atomic claims and the complete product grid.
- Independent image-equivalence and prompt-equivalence review.
- Outcome-blind clinician product control: preferably 240 unique image-claims and four fixed clinicians, with a Latin-balanced one-cell-per-clinician-per-claim allocation that preserves the within-claim 2 x 2 without repeat-view recall. Direct presence probabilities define the proper-loss contrast; signed harm, absolute/adverse-transition, assessability, and model-over-panel gates are conjunctive. Use 160 only with outcome-blind variance support. If only a two-clinician between-image fallback is feasible, limit the conclusion to an average randomized interaction rather than an instance-level product defect.
- Adjudication, attestation, computational guard exclusions, and hash closure.
- Explicit terminal reject path; no synthesized decision or automatic truth.

#### 3.2 Locked Product Identification

Opening: CECD is identified only when the joint-cell defect exceeds both marginal sensitivities and survives simpler utilization mechanisms.

- Pilot screening followed by frozen dev fitting and untouched image-disjoint two-model confirmation.
- A 16-stratum Brier-PAEL readout conditioned on both marginals and adjusted by an isospectral orientation reference; it is an estimand rather than metric novelty, and NLL/alternative nulls remain sensitivities.
- MetaRA-style joint failure with explicit single-axis cells and a frozen Yang-style semantic-boundary-proximity predictor.
- Image-cluster bootstrap, MCID, and predeclared finding/model heterogeneity guards.

#### 3.3 Adversarial Mechanism Discrimination

Opening: A product residual earns a mechanism interpretation only if simpler perception, utilization, prompt, and representation accounts fail to absorb it.

- Directional image-use and same-support swap/relevant-versus-irrelevant occlusion controls.
- CALRD/context-utilization, explicit modality-conflict direction/steering,
  System/PIH, prompt-head, HALP, and reader-alias controls selected on dev only.
- Random/norm and nonlinear/readout-rotation controls.
- Dev-selected layerwise interaction energy versus reader-loss orientation.
- Upstream spectrum/norm/marginal-preserving orientation replacement versus
  random, equal-norm, isospectral and final-logit controls; relative PAEL rescue
  must be at least 20% with at most 1pp clear-case degradation in both models.
- Joint-cell additive closure remains a causal probe unless it clears all
  method baselines; generic steering and subspace editing have no novelty claim.

### Evaluation boundary

- The core paper is the fusion-induced clinical orientation mechanism. CECD and
  PAEL are its admitted experimental substrate and readout, not the headline
  problem, benchmark, decoder, or metric.
- Closed-ontology VinDr listing is a post-gate external-validity test. Fixed-​K, matched coverage/length, omission, negative, hedge, refusal, location, and attribute metrics prevent a false transfer claim; they are not a new method.
- Report generation, unrestricted OE, external-knowledge hallucination, and evidence-bounded editing are outside the main contribution. If pursued, they require a separate scope and faithful CEBC/VLI/HalluTrace/ConRad comparisons.
- If CECD passes CE but fails listing, the claim remains limited to controlled binary clinical claims; it must not be generalized to medical VLM hallucination broadly.

### Self-consistency checks

- **Check 1, Limitations → Goal:** PASS. The goal defines the missing product object; independent admission and residual-specific identification address the two validity limitations.
- **Check 2, Goal → Challenges:** PASS. Clinical admission, locked interaction estimation, and causal specificity arise directly from identifying the stated product object.
- **Check 3, Challenges → Methodology:** PASS. Challenges 1/2/3 map one-to-one to Modules A/B/C.
- **Check 4, Methodology → Contributions:** PASS only conditionally: Contributions
  1–2 disappear if the orientation jump or selective upstream rescue fails.
  Evaluation controls, PAEL and provenance are not promoted into metric,
  benchmark, systems or generic mitigation novelty.

### Severity summary

- **0 CRITICAL, 2 MAJOR, 0 MINOR** before outcomes, with F1 automatically
  escalating to CRITICAL if generic composite failure/boundary proximity absorbs
  PAEL or causal orientation rescue fails.
- Top fixes: prove product-specific novelty against 2026 controls; secure both axis-admission reviewers and the fixed-panel recall-free product control; stop immediately on locked CECD NO-GO.
- The revised logic chain is internally consistent and no longer depends on OE/report mitigation. Scientific readiness remains conditional. Do not draft an Introduction that states any contribution as fact until the corresponding behavioral, human, mechanism and causal gates pass.
- **Pre-revision F8 was real but corrected:** the earlier OE/report claim-exchange module was not entailed by the CECD goal and mixed mechanism, mitigation, and benchmark-like validation. It is now outside the core paper rather than counted as a fourth contribution.

### ICLR oral bar

The current candidate is **not** oral-level on framing or positive PAEL alone.
An oral-caliber story requires: a nontrivial residual in both models and most
eligible findings; energy–orientation dissociation at an architecture-specific
fusion transition; an upstream matched-spectrum intervention that rescues at
least 20% of PAEL with at most 1pp clear-case loss and beats random/ispectral
controls; survival against composite-MR, boundary-proximity, marginal,
behavioral, context, prompt/system-token, reader and human controls; and one
fixed-​K listing replication with no content-budget trade. More tasks or a new
benchmark cannot compensate for a weak or absorbed mechanism.

### Next step

Do **not** invoke an Introduction drafter yet. First complete the locked CECD
admission/readout and independent human product control. A behavioral GO only
authorizes the frozen composite-MR, boundary-proximity and layerwise mechanism
tests; it is not a paper result by itself. A model NO-GO terminates the framing,
and a failed selective causal rescue kills the ICLR mechanism claim even if
PAEL is positive.

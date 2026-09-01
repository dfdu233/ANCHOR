# C45 — Causal Retraction to Image-Identifiable Phenotypes

Date: 2026-08-13  
Scope: formula-level collision audit and CPU-only prevalence screen; no GPU and
no baseline mutation.

## Executive decision

**NO-GO as the requested new ICLR-level mitigation primitive.**  The clinical
boundary is real and important: a radiograph can support an opacity pattern
without identifying its unique etiology.  However, the proposed operation—parse
an etiologic claim and deterministically replace it by a more general,
image-observable ancestor—is already the same algorithmic object as selective
abstraction / semantic backoff.  It is also a special case of evidence-bounded
minimal report editing.  A causal quotient gives an honest explanation of why
the edit is safe, but does not change the intervention.

The strongest defensible use is a **typed safety rule or baseline** for
image-only systems, not a standalone ICLR oral method.  No GPU experiment is
authorized.

## 1. Candidate and mathematical object

Let hidden disease/etiology be `D`, non-image clinical context be `H`, and an
image be sampled through the imaging channel

\[
X \sim P(X\mid D,H).
\]

For an image-only task, two latent clinical worlds are observationally
equivalent when their image laws cannot be distinguished:

\[
(d,h)\sim_X(d',h')
\iff P(X\mid d,h)=P(X\mid d',h').
\]

An image-identifiable claim must be constant on every equivalence class of
`~_X`; equivalently it factors through the quotient map
`q:(D,H)->(D,H)/~_X`.  The proposed report operator is a retraction

\[
r:\mathcal C\rightarrow\mathcal C_X,
\qquad r\circ r=r,
\qquad r(c)=c\;\text{for }c\in\mathcal C_X,
\]

where `C_X` contains observable phenotype claims.  Example:

```text
right lower-lobe pneumonia -> right lower-lobe air-space opacity
ARDS                         -> bilateral diffuse air-space opacity
CHF                          -> cardiomegaly / pulmonary vascular congestion
```

The replacement occupies the same claim slot and can preserve sentence count.
The idempotence and preservation statements are mathematically correct, but
they are properties of any deterministic projection onto a permitted
vocabulary; they are not new theorems about VLM decoding.

### Important correction to the proposal

Exact equality of the full distributions `P(X|D)` is too strong and cannot be
established from the available datasets.  In practice one would need either a
task-relative equivalence

\[
d\sim_{X,\epsilon}d' \iff
\mathrm{TV}(P(X\mid d),P(X\mid d'))\le\epsilon,
\]

or a clinically curated observability map.  The first requires estimation and
calibration; the second turns the method into an ontology/rule system.  Thus the
clean quotient is a normative formalization, not an executable training-free
detector of which generated etiology is unsupported.

## 2. Grounded phenomenon

The medical premise is well supported.  A review of pneumonia imaging states
that without clinical information radiologists cannot reliably distinguish
pneumonia from other pulmonary processes, and lists noninfectious mimics.
Radiology report standards similarly distinguish image observations in
`Findings` from a diagnostic/differential `Impression`, which may integrate
clinical context.  Pragmatic RRG explicitly argues that the usual image-to-full-
report problem lacks sufficient inputs for some expected output content and
cleans image-uninferable targets.

There is also a local behavioral substrate.  A deliberately conservative regex
screen of the complete native greedy reports counted positive mentions of
`pneumonia/ARDS/COVID/CHF/sepsis/infection/aspiration/malignancy` after excluding
nearby explicit negations:

| model / set | reports | any positive etiology mention | mention absent from reference* |
|---|---:|---:|---:|
| Huatuo / IU-Xray | 590 | 40 (6.8%) | 40 (6.8%) |
| Huatuo / MIMIC-CXR | 694 | 178 (25.6%) | 171 (24.6%) |
| Hulu / IU-Xray | 590 | 0 (0.0%) | 0 (0.0%) |
| Hulu / MIMIC-CXR | 694 | 51 (7.3%) | 36 (5.2%) |

`*` This is **not clinical hallucination truth**: a reference report can omit a
true differential and regex polarity is imperfect.  It proves only that the
target behavior is nontrivial for Huatuo/MIMIC and strongly model/dataset
dependent.  It cannot justify a mitigation result without expert or independent
etiology truth.

Representative model output from the existing cache:

```text
"... diffuse, patchy opacities ... could suggest an underlying inflammatory
or infectious process such as pneumonia."
```

The corresponding IU-Xray reference says the lungs are free of focal
infiltrates.  This is a strong automatically surfaced candidate error, but the safe replacement
(`patchy opacities`) is already present in the model sentence; stripping the
etiologic tail is ordinary span rewriting rather than a new decoder.

## 3. Formula-level collision matrix

| Work | Same phenomenon | Same mathematical/intervention object | Difference from C45 | Verdict |
|---|---|---|---|---|
| [Pragmatic Radiology Report Generation, ML4H 2023](https://arxiv.org/abs/2311.17154) | Reports contain content not inferable from supplied image; indication changes what should be mentioned | Identifies image-uninferable information and cleans it from targets | Training/data-side rather than inference-time quotient | Occupies the radiology problem statement |
| [Selective Abstraction, 2026](https://arxiv.org/abs/2602.11908) | Claims can be wrong because they are more specific than evidence supports | Decomposes text into atoms and replaces uncertain atoms with less-specific abstractions | Generic factuality and confidence-gated; C45 fixes the abstraction by an imaging observability ontology | **Direct intervention collision** |
| [Compositional Selective Specificity, 2026](https://arxiv.org/abs/2604.17487) | Local claim overcommitment | Proposes coarser semantic backoffs and emits the most specific admissible level | Agentic/text setting and calibrated selection | **Direct intervention collision** |
| [CEBC, ACL 2026](https://aclanthology.org/2026.acl-long.2142/) | Visually unsupported output mentions | Training-free, external-evidence-bounded minimal revision/suppression | C45 replaces a diagnosis with a parent phenotype rather than detector-supported object | Strong method collision |
| [HalCECE, 2025](https://doi.org/10.1007/978-3-032-08330-2_5) | Over-specialized visual concepts | Uses ontology ancestry/LCA to identify over-specialization and conceptual edits | Detection/evaluation rather than a deployed mitigation | Occupies hypernym/ancestor semantics |
| [ZINA, 2025](https://arxiv.org/abs/2506.13130) | Fine-grained multimodal hallucinated spans | Detects erroneous spans and proposes refinements | Learned editor, no identifiability quotient | Strong span-editing neighbor |
| [FactCheXcker, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Heiman_FactCheXcker_Mitigating_Measurement_Hallucinations_in_Chest_X-ray_Report_Generation_Models_CVPR_2025_paper.html) | A typed report claim exceeds what ordinary generation reliably supports | Query-specialist-update modular post-hoc correction | Measurement rather than etiology; calls a specialist instead of abstraction | Shows typed post-editing is established |

No retrieved paper uses exactly the latent imaging-channel quotient notation.
That notation is nevertheless a **cosmetic/theoretical delta** because the
realized map is the same coarsening operation as Selective Abstraction and CSS.

## 4. Why the causal language does not rescue novelty

1. **Non-identifiability does not identify the correct phenotype.**  From one
   image, the method still needs a rule/ontology or a visual verifier to know
   whether `pneumonia -> opacity` is appropriate.  The quotient alone supplies
   no executable score.
2. **Retraction is output-only.**  It does not improve visual perception or
   recover a missed finding.  It changes semantic specificity after generation.
3. **The fixed-claim-count condition is formally but not informationally
   fixed.**  Replacing `pneumonia` by `opacity` retains one syntactic claim but
   intentionally removes etiologic information.  Recent abstraction work
   explicitly models this as a risk--information tradeoff; it is misleading to
   claim that no content was reduced merely because `K` is unchanged.
4. **A universal retraction is clinically wrong.**  Some etiologies are highly
   image-characteristic in particular modalities or settings, and others become
   identifiable with multiple views, priors, pathology, or history.  The map
   must be conditioned on modality, acquisition, available context and task.
   This weakens the promised universal simplicity.
5. **The benchmark reference is not causal truth.**  MIMIC/IU report wording
   cannot determine whether a diagnosis was visually inferable or imported
   from history.  Without independent clinical adjudication, apparent gains may
   be lexical alignment to a rewritten target.

## 5. Hard-gate score

| Gate/dimension | Decision | Evidence |
|---|---|---|
| G1 grounded phenomenon | PASS | Clinical imaging ambiguity plus nonzero cached etiology generation |
| G2 no direct collision | **FAIL** | Selective Abstraction and CSS implement claim-to-ancestor semantic backoff; CEBC covers training-free evidence-bounded editing |
| G3 falsifiability | PASS in principle | Requires independently labeled image-identifiable vs context-dependent etiologies |
| G4 organic fusion | PARTIAL | Causal identifiability explains the rule but does not compute it |
| Importance | 2/3 | Clinically consequential but only one error subtype |
| Mechanistic value | 1/3 | Normative boundary, no VLM mechanism or new evidence source |
| Novelty space | 0/3 | Intervention and ontology coarsening are occupied |
| Executability | 1/3 | Rule implementation is easy; trustworthy admission is unavailable without clinical labels |

**ICLR ceiling:** as written, a useful medical safety engineering paper or
baseline, not ICLR oral.  The only potentially substantive upgrade would be to
discover a new, case-conditional means of computing the observational
equivalence class from image evidence.  That is precisely the hard perception/
identification problem; a static etiologic ontology does not solve it.

## 6. Decision and reuse boundary

- Do not launch a GPU test or add a method name.
- Retain the quotient formalization as a paper-level limitation: an image-only
  generator cannot be required to recover context-dependent etiology.
- If used as a baseline, call it `ontology ancestor rewrite`, disclose that it
  reduces semantic specificity, and compare directly with Selective
  Abstraction/CSS/CEBC.
- Do not count `pneumonia -> opacity` as preserved information merely because
  sentence or claim count is fixed.
- Search should move to a mechanism that adds or isolates case-specific visual
  evidence rather than another output abstraction/rewrite.

## Verified references

1. Nguyen et al. *Pragmatic Radiology Report Generation*. ML4H 2023.
   <https://arxiv.org/abs/2311.17154>
2. Goren et al. *When Should LLMs Be Less Specific? Selective Abstraction for
   Reliable Long-Form Text Generation*. arXiv:2602.11908, 2026.
   <https://arxiv.org/abs/2602.11908>
3. Huang et al. *Answer Only as Precisely as Justified: Calibrated Claim-Level
   Specificity Control for Agentic Systems*. arXiv:2604.17487, 2026.
   <https://arxiv.org/abs/2604.17487>
4. Mishra et al. *CEBC: Conformal Evidence-Bounded Control for
   Low-Hallucination Vision–Language Generation*. ACL 2026.
   <https://aclanthology.org/2026.acl-long.2142/>
5. Wada et al. *ZINA: Multimodal Fine-grained Hallucination Detection and
   Editing*. arXiv:2506.13130, 2025. <https://arxiv.org/abs/2506.13130>
6. Heiman et al. *FactCheXcker: Mitigating Measurement Hallucinations in Chest
   X-ray Report Generation Models*. CVPR 2025.
   <https://openaccess.thecvf.com/content/CVPR2025/html/Heiman_FactCheXcker_Mitigating_Measurement_Hallucinations_in_Chest_X-ray_Report_Generation_Models_CVPR_2025_paper.html>
7. Karadimos et al. *HalCECE: A Framework for Explainable Hallucination
   Detection Through Conceptual Counterfactuals in Image Captioning*. 2025.
   <https://doi.org/10.1007/978-3-032-08330-2_5>

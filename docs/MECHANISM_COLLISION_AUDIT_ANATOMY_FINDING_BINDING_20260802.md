# Mechanism collision audit: Anatomy–Finding Conjunctive Binding

**Audit date:** 2026-08-02  
**Scope:** 2024–2026 conference, journal, arXiv, official code and official data
releases; primary sources only  
**Mode:** literature and local-document audit; no GPU and no common-eval change  
**Decision:** **BROAD BINDING KILL.** `Reference-Frame Mediation Boundary` is
**conditional only**, and its data gate has **not passed**.

## Frozen question

The candidate asked whether a medical VLM can retain finding identity and
patient-side/location separately, yet fail to bind them during generation, and
whether a layer-selective intervention can repair the location while preserving
the finding and a fixed number of open-ended claims.

The collision test deliberately separated four contributions:

1. real CXR finding-by-patient-side/location gold;
2. layerwise decomposition of identity from location/order;
3. selective causal intervention on the location variable;
4. fixed-`K` open-ended correction that keeps claim identity/count unchanged.

The mechanism-novelty rule is stricter than finding one paper with an identical
title. A candidate is collided when established work already supplies its core
mechanism and causal tool, while adjacent domain work supplies the target data,
task and correction objective. Combining those two mature lines is an
application transfer, not an organic new mechanism, unless the target domain
breaks a substantive assumption of the established mechanism.

## Search protocol and limits

Discovery and adversarial searches covered the intersections of `CXR/radiology`
with `laterality`, `location`, `grounded report`, `finding-location error`,
`binding`, `spatial ID`, `position ID`, `layerwise`, `activation patching`,
`causal mediation`, `steering`, `fixed claim count`, and `open-ended report
correction`. Searches were triangulated through official OpenReview/ICLR, CVF,
MICCAI, ACL, arXiv, PhysioNet, BIMCV and author/project-code pages. Secondary
indexes were used only to discover titles; claims in this document come from
the primary paper, official abstract, official repository or official release.

The negative statement about fixed-`K` is bounded to the 2024–2026 primary
sources found by this audit, not a proof that no unpublished or differently
named method exists. The kill decision does not depend on that negative result:
it follows from positive, verified collisions on the mechanism and medical-task
sides.

## Closest-work matrix

Legend: `Y` = directly present; `P` = partial/adjacent; `N` = absent; `—` = not
applicable. `Fixed-K OE` means that the same number and identity set of positive
clinical claims is preserved while only the side/location binding is changed.

| Primary work | Real CXR finding↔side/location gold | Layerwise identity/location factorization | Selective causal intervention | Fixed-K OE | Collision implication |
|---|---:|---:|---:|---:|---|
| [Visual Symbolic Mechanisms, ICLR 2026 Oral](https://openreview.net/forum?id=3RQ863cRbx) | N | **Y**: content-independent spatial/position IDs bind object features | **Y**: causal mediation traces binding errors to the mechanism | N | Directly occupies the general VLM binding mechanism; a CXR-only replay is not mechanism novelty. |
| [Linear Mechanisms for Spatiotemporal Reasoning, ICLR 2026](https://openreview.net/pdf/e63a2579ef6331936b252e00cabcd1af4e1ffea0.pdf) and [official code](https://github.com/Raphoo/linear-mech-vlms) | N | **Y**: spatial IDs linearly bind object location to text activations | **Y**: intermediate-layer belief swapping/steering; mirror and attribute swaps | N | Direct collision with the proposed linear side subspace and layerwise causal steering. |
| [Dual Mechanisms of Spatial Variable Binding, arXiv 2026](https://spatial.baulab.info/) and [official code](https://github.com/Nix07/spatial-variable-binding) | N | **Y**: ordering and attribute transfer occur in distinct layer ranges | **Y**: counterfactual interchange patching and ordering amplification | N | Strongest collision: same probe→patch→selective-spatial-improvement logic on synthetic and natural images. |
| [Mechanisms of Object Localization in VLMs, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Schaumloffel_Mechanisms_of_Object_Localization_in_Vision-Language_Models_CVPR_2026_paper.pdf) and [official code](https://github.com/t9s9/vlm-loc-mechanisms) | N | **P/Y**: classification and localization share early processing but use mostly distinct sparse heads | **Y**: token ablation, attention knockout and causal mediation | N | Further removes novelty from a generic “semantic identity versus location pathway” claim. |
| [Phrase-grounded APO, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Mahmood_Phrase-grounded_APO_for_Improving_Chest_X-ray_Report_Generation_CVPR_2026_paper.html) | **Y/P**: image-grounded finding, anatomy and laterality supervision | N | **P**: inference-time parameter update with finding-veracity and grounding losses, not activation patching | **N**: deletes/flips findings and adds missed findings | Strongest medical application collision. It already represents `finding|anatomy|laterality` and corrects report errors, but does not preserve claim set/count. |
| [RadSCR, ICLR 2026](https://openreview.net/forum?id=6sOSwgCmpH) | **Y/P**: abnormality-region proposals from CXR grounding data | N | N: learned proposal critique, not hidden-state causal patching | **N**: critiques alternatives and adds potential false negatives | Occupies finding×region reasoning and correction as a medical method contribution. |
| [MAIRA-2, arXiv 2024](https://arxiv.org/abs/2406.04449) and [official model release](https://huggingface.co/microsoft/maira-2) | **Y/P**: grounded finding sentences and boxes | N | N | N | Occupies grounded free-text CXR reporting and spatial correctness/completeness evaluation. |
| [MedRegA, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/414fd191b3246a19a55741b938380136-Abstract-Conference.html) | **Y/P**: region-centric CXR instruction data | N | N | N | Occupies region identification, detection and grounded report generation; laterality/location is already an explicit clinical target. |
| [Phrase-grounded Fact-checking, MICCAI 2025](https://papers.miccai.org/miccai-2025/0693-Paper3526.html) | **Y/P**: synthetic identity/location perturbations plus image grounding | N | N: detector only | N | Detecting identity and indicated-location errors is established; omission correction is outside this work. |
| [VIHD, arXiv 2026](https://arxiv.org/abs/2605.20772) | P: medical VQA, not side gold | P: layerwise visual-dependency windows, not identity/location axes | P: targeted visual-token masking, not side-subspace patching | N | Close medical mechanism control; prevents a broad “medical layerwise intervention is new” claim. |
| [PadChest-GR, arXiv 2024/2025](https://arxiv.org/abs/2411.05085) | **Y**: finding-level categorical location and up to two radiologist box sets | N | N | — | Supplies the most defensible current finding-location gold; therefore the dataset itself is not a contribution. |
| [Chest ImaGenome, official PhysioNet release](https://physionet.org/content/chest-imagenome/1.0.0/) | **Y/P**: manual gold anatomy-object↔attribute relations; explicit left/right anatomy | N | N | — | Supplies an independent anatomy-binding replication substrate, though exact finding×side test cardinalities require access audit. |
| [Laterality-error comparison, JACR 2024](https://doi.org/10.1016/j.jacr.2024.06.014) | **Y**: radiologist-reviewed laterality mismatch cases across modalities | N | N | N | Establishes clinical laterality as an image/report QA problem and exposes dependence on an `L` marker prompt; useful evidence for a reference-frame boundary, not binding novelty. |

No primary work found in this audit combines all four columns in one experiment.
That absence does **not** rescue the broad candidate: the first three columns are
already jointly covered by the generic mechanism line plus the medical
grounding/correction line. The remaining fixed-`K` constraint is an important
anti-gaming control, but not by itself an ICLR-level mechanism contribution.

## Why the broad candidate is killed

### 1. The proposed mechanism is already explicit in general VLMs

The ICLR 2026 Oral on visual symbolic mechanisms identifies content-independent
spatial indices used to bind attributes to visual entities and causally links
binding failures to that mechanism. Kang et al. identify linearly extractable
spatial IDs attached to textual activations, then manipulate beliefs at
intermediate layers. Cui et al. go still closer: their final-token interchange
experiments separate ordering transfer from attribute transfer by layer, trace
the dominant ordering signal to projected visual tokens, identify a backup
language-backbone route, and improve natural-image spatial reasoning by
amplifying that direction. CVPR 2026 localization work separately shows sparse,
causal and largely specialized classification-versus-localization heads.

Thus the claim “identity is retained, location is separately encoded, and a
selective layerwise patch repairs their binding” is not unoccupied. Replacing
colored objects with radiographic findings and left/right spatial relations is
a new evaluation setting, not yet a new mechanism.

### 2. The clinical task and correction objective are also occupied

Phrase-grounded APO parses a CXR statement into a fine-grained structure
including finding, anatomy and laterality; its paper gives the example
`anatomical finding|yes|pleural effusion|lung|left`. A fact-checking model
estimates both finding veracity and location. The published correction rules can
drop a wrong location while retaining the finding, flip polarity, or remove the
finding; its inference-time optimization can also add previously missed
findings. RadSCR likewise constructs abnormality identity-region proposals and
critiques them against alternative abnormalities, patient images and potential
false negatives. MAIRA-2 and MedRegA already make grounded reporting and region
reasoning first-class tasks.

These papers do not contain the proposed activation decomposition, but that
missing piece is exactly what the general VLM papers already provide. The
union is too close for a broad medical-binding paper to survive a
mechanism-level novelty review.

### 3. Fixed-K remains useful, but is a control rather than the discovery

The closest medical methods do not hold positive-content count fixed. APO can
remove an erroneous left pleural effusion and later add a right pleural
effusion; the authors explicitly expect previously missed findings to appear.
RadSCR has a dedicated false-negative branch and produces a completed report.
Therefore matched claim count/identity would be a valuable experiment: it can
prove that apparent hallucination gains are not caused by shorter reports,
deleted claims or a precision–recall trade.

However, `fixed K` specifies what a valid correction is allowed to change. It
does not establish a new computational mechanism. It should be retained as an
evaluation contract for any future OE intervention, not sold as the paper's
main novelty.

## Dataset and construct gate

### Real patient-side/location gold exists in principle

- **PadChest-GR:** 4,555 studies, 7,037 positive and 3,422 negative finding
  sentences. Every positive finding has categorical finding type and location,
  plus up to two independent radiologist box sets. This is the preferred
  finding-linked source.
- **Chest ImaGenome gold:** 500 unique patients with manually validated or
  corrected anatomy-object↔attribute relations and dual-annotated anatomy
  boxes. Explicit objects include left/right lung zones. Silver data are
  automatically derived and should not define the primary truth.
- **MS-CXR:** 1,162 radiologist-verified phrase-box pairs across eight findings.
  It has no published categorical patient-side cross-tab and is too small for a
  high-cardinality three-finding-by-two-side primary test.

These releases show that “no side gold exists” is false. They do not establish
that the required finding×left/right×patient cells are large enough; that must
be counted from the actual official annotations with patient-disjoint splits.

### The medically distinctive pivot requires more than a side label

The only potentially non-cosmetic boundary is not ordinary spatial binding but
a **reference-frame conversion**:

```text
visible lesion / pixel hemifield
            ↓
acquisition and display orientation + side marker
            ↓
patient anatomical side
            ↓
textual laterality bound to a clinical finding
```

Generic binding papers typically control object order in the displayed image.
Clinical laterality refers to the patient, and may require transforming display
coordinates using acquisition metadata or an embedded side marker. The JACR
study is suggestive: its CXR prompt explicitly told the system that the `L`
marker denotes the patient's left side when the initial answer was wrong.

This is a three-variable problem—finding identity, visible spatial locus and
patient reference frame—only if the experiment can independently observe or
intervene on all three. A report word `left`, a bounding-box centre, or a
horizontally flipped PNG cannot establish that construct.

## Conditional survivor: Reference-Frame Mediation Boundary

This is **not approved as a method mainline**. It survives only as a cheap,
data-gated falsification probe.

### Required data gate — currently not passed

The admitted manifest must link, for the same image and finding:

1. expert finding identity and polarity;
2. expert patient-side label and preferably a finding box;
3. original DICOM orientation/acquisition metadata;
4. observable side-marker status or an independently verified mapping from
   stored pixel coordinates to patient anatomical side;
5. patient-disjoint train/dev/test membership.

It must then certify adequate `finding × patient side × display/orientation`
cells. PadChest-GR appears sufficient for items 1–2, but the audit has not
verified that its released files expose items 3–4 at the necessary resolution
or adequate cross-cell cardinality. Chest ImaGenome likewise does not by itself
certify an exact display-frame counterfactual. Therefore the data gate is
**UNPASSED**.

### Required discovery gate

Continue only if a CPU-first schema/count audit passes and a minimal behavioral
test shows all of the following:

- finding identity is intact;
- lesion hemifield/display location is intact;
- patient-side prediction fails selectively when the reference-frame cue is
  changed, hidden or made inconsistent;
- the failure exceeds matched image/prompt and answer-length drift;
- at least two medical VLMs share the pattern;
- the result cannot be reproduced by the established generic spatial-ID
  intervention without modeling the reference-frame mediator.

Only then is a layerwise study justified. The causal test must patch or mediate
the reference-frame variable while matching finding identity and visible
location, with random-direction, norm, prompt-paraphrase and ordinary spatial
steering controls. OE evaluation must preserve claim identity/count first;
laterality correction is scored separately from finding deletion or addition.

### Kill conditions

Kill the survivor immediately if any of these holds:

- no trustworthy orientation/marker variable can be joined to side gold;
- the only counterfactual is a synthetic flip that also changes anatomy,
  marker validity or image distribution;
- patient-side error is fully explained by loss of lesion localization or
  finding identity;
- generic spatial-ID directions repair the task without a distinct
  reference-frame mediator;
- gains arise from deleting claims, changing `K`, shortening answers or using
  marker text as a trivial OCR shortcut.

## Final reviewer-style verdict

**BROAD ANATOMY–FINDING CONJUNCTIVE BINDING: KILL.** The core mechanism,
layerwise causal tool and selective spatial repair are already established in
general VLM work; finding-location grounding and report correction are already
established in CXR work; suitable clinical gold datasets already exist. The
remaining combination is too readily described as applying ICLR 2026 binding
mechanisms to PadChest-GR and comparing with APO/RadSCR.

**REFERENCE-FRAME MEDIATION BOUNDARY: CONDITIONAL ONLY.** It could become a
substantive medical boundary because patient laterality is not necessarily the
same variable as displayed left/right order. At present, the required
orientation/marker-linked expert manifest and exact counterfactual have not
been demonstrated. **Data gate: UNPASSED. No GPU or mainline commitment is
justified.**

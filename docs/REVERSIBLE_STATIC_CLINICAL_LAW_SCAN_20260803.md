# Reversible static clinical-law scan

**Frozen:** 2026-08-03  
**Decision:** **ALL-NO-GO; no GPU experiment is authorized by this scan.**

## 1. Search contract

This was a substrate-first search for a mechanism in static medical images that
simultaneously has:

1. public, independently defined clinical truth;
2. a bidirectional counterfactual whose two clinical actions are known before
   observing model outputs;
3. a native single-image input path for HuatuoGPT-Vision, Hulu-Med and
   LLaVA-Med;
4. an OE intervention that holds atomic claim identity and claim count fixed;
5. a mechanism-level delta beyond robustness, calibration, abstention or a new
   medical dataset transfer.

The search excluded the already rejected or occupied reader, style,
source-domain, ROI, laterality, counting, temporal-inversion, device-CAZC,
CECD, Specificity, measurement, and evidence-source-erasure branches.  It also
did not inspect sealed model outcomes and ran no GPU code.

Three construction analyses guided the search.  The ViT pattern suggested
redefining the primitive from a disease label to a clinically lawful state
transition; SigLIP suggested removing accidental coupling between disease truth
and image observability; Model Collapse suggested demanding a state variable
and a reversible trajectory rather than another mitigation module.  These are
analytical reconstructions, not claims about the exemplar authors' discovery
histories.

## 2. Frozen decision table

| Candidate | Independent truth and cardinality | Bidirectional intervention | Closest mechanism collision | Hard-gate verdict |
|---|---|---|---|---|
| **Gradability--Commitment Collapse (GCC)** on EyeQ | **Annotation-count pass; image-access fail.** 28,792 annotation rows; official split has 8,614/12,048 patients and zero patient overlap. Every `quality x any-DR` cell has at least 523 train and 1,025 test images. Original EyePACS pixels require Kaggle access and are not local. | Start from a gradable fundus image and toggle a preregistered strong acquisition artifact on/off. Patient DR truth and the fixed DR claim remain unchanged; only `gradable -> ungradable -> gradable` is licensed. | Cheng et al., *npj Digital Medicine* 2025 already evaluate original/artifact pairs, disease errors, and ungradable-image detection; Ouaari et al. 2026 benchmark 16 VLMs over seven modalities, seven corruptions and embedding displacement; CVPR 2023 already defines image-quality-aware diagnosis; BCEA 2026 already formalizes answer/abstain/acquire with claim-specific visual re-examination. | **NO-GO.** The only remaining statement is layerwise `quality is encoded but erased at commitment`, which reuses the locally failed reader two-plane erasure hypothesis with a new clarity label. A new substrate plus probes is not a new mechanism. |
| **Cross-Magnification Evidence Conservation (CMEC)** on BreaKHis/pathology | **Fail.** Public BreaKHis contains 7,909 downloadable images but only 82 patients, so it cannot meet a 100-patient-disjoint-per-cell gate. Images at 40x/100x/200x/400x are not guaranteed to be the identical field of view. | Magnification up/down would preserve specimen diagnosis, but not pixel content or independently adjudicated visibility of each atomic claim. | MR-PLIP (CVPR 2025) directly introduces cross-resolution pathology-language alignment; MLLM-HWSI (CVPR 2026) explicitly aligns cell, patch, region and WSI evidence for open-ended reasoning. | **Direct/cosmetic collision plus substrate failure.** Layer probing or applying this to three generic medical VLMs would not change the occupied mechanism. |
| **Diagnostic-Sign Likelihood-Ratio Asymmetry (SDLA)** on ISIC/Derm7pt | **Insufficient.** ISIC 2018 publishes 2,594 images with five attribute masks and 10,015 diagnosis entries, but the public task does not supply a same-lesion, pathology-certified pair in which one sign is independently toggled in both directions. Cell counts alone cannot certify a counterfactual. | Erasing a sign mask destroys evidence; donor insertion invents tissue. Natural matched lesions are associations, not bidirectional counterfactuals. | VL-MedGuide 2025 already performs concept perception followed by diagnostic reasoning on Derm7pt; MAKE 2025 and concept-adaptive VLM work already align dermatologic concepts and diagnoses; DermFM-Zero 2026 reports sparse clinical concepts and targeted suppression of artifact-induced bias. | **NO-GO.** Without a truthful reversible edit it becomes concept-bottleneck correlation or post-hoc calibration; with a generative edit it requires clinician admission and no longer has independent truth. |

Scores before hard gates (`I/M/N/E`, each 0--3) were GCC `3/2/1/3`, CMEC
`2/2/0/1`, and SDLA `2/2/1/1`.  Hard-gate failures override these scores.

## 3. Candidate 1 audit: Gradability--Commitment Collapse

### Proposed law

Image quality and patient disease truth are different variables.  When a fixed
image is made clinically ungradable, a claim such as `diabetic retinopathy is
visible` must move from supported/refuted to undetermined.  The patient's DR
label does not change.  Restoring the original image must restore gradability.
This would preserve claim identity, polarity target and positive-claim count;
only the permissible commitment changes.

### Frozen cardinality result

The official EyeQ repository was inspected at commit
`1dcfabb08cd31045de1a4a3ee82e48f6833697aa`.  Annotation hashes are:

```text
train 34f39f8923f62a13fcaeccef7e5da7ec607b45805e7736042fa274b2731b9480
test  209135bb0dff2d6203c445ee010cbdcb925edb5338a040dc6ae62c06649047b3
```

`quality=0/1/2` denotes good/usable/reject in the release.

| Split | Images | Patients | q0/no DR | q0/DR | q1/no DR | q1/DR | q2/no DR | q2/DR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 12,543 | 8,614 | 6,342 | 2,005 | 1,353 | 523 | 1,544 | 776 |
| Test | 16,249 | 12,048 | 5,967 | 2,504 | 3,200 | 1,358 | 2,195 | 1,025 |

The patient identifier was parsed from the released left/right image name; the
official train/test patient overlap is zero.  Referable DR (`grade >= 2`) is
also adequately populated: the smallest train cell is 420 and the smallest
test cell is 872.  These are annotation counts, not an execution certificate:
the EyeQ repository redirects users to the original EyePACS images on Kaggle,
and no EyeQ/EyePACS pixels were found locally during this audit.

### Why adequate data do not authorize an experiment

The exact behavioral phenomenon is already established.  The 2025
[medical-image artifact study](https://www.nature.com/articles/s41746-025-02108-w)
tests clean images and weak artifacts for disease detection, strong artifacts
for poor-quality detection, and real fundus images stratified as high quality,
weak artifact, and ungradable.  It reports both false-positive and
false-negative transitions and explicitly notes that structured prompts can
suppress poor-quality acknowledgements.  [Image Quality-Aware Diagnosis via
Meta-Knowledge Co-Embedding](https://openaccess.thecvf.com/content/CVPR2023/html/Che_Image_Quality-Aware_Diagnosis_CVPR_2023_paper.html)
already makes quality a causal auxiliary variable for diagnosis.  [BCEA](https://arxiv.org/abs/2606.16667)
already treats claim verification as answer/abstain/acquire and includes
claim-specific visual interventions with risk--coverage guarantees.

The closest new neighbor is [Assessing VLM Reliability for Medical Image
Quality Evaluation Under Corruption and Bias](https://arxiv.org/abs/2607.01973)
(Ouaari et al., 2026).  It benchmarks 16 VLMs across seven modalities, seven
corruption types and five severity levels; it additionally relates score change
to embedding displacement and tests contextual bias in quality judgments.
Therefore neither a broader model/corruption grid nor a hidden-state displacement
correlation is available as the novelty delta.

A layerwise residual could still be scientifically descriptive, but the unique
prediction would be identical in form to the reader-clarity two-plane mechanism
already falsified locally: quality/clarity is decoded before the final layer and
then erased by language commitment.  Changing independent disagreement labels
to independent gradability labels does not rescue a rejected causal mechanism.
There is therefore no mechanism-level delta sufficient for an ICLR mainline.

## 4. Candidate 2 audit: Cross-Magnification Evidence Conservation

[BreaKHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
has pathology labels and images at 40x, 100x, 200x and 400x, but only 82
patients.  This fails the frozen minimum of 100 patient-disjoint examples per
primary counterfactual cell before model capability is considered.  Moreover,
the images at different magnifications are samples from a tumor, not a released
registration proving an identical field and exact reversible scale action.

The remaining conceptual space is occupied at mechanism level:

- [MR-PLIP, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Albastaki_Multi-Resolution_Pathology-Language_Pre-training_Model_with_Text-Guided_Visual_Representation_CVPR_2025_paper.html)
  explicitly learns multi-resolution and cross-resolution pathology-language
  alignment on 34 million image-language pairs.
- [MLLM-HWSI, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Alawode_MLLM-HWSI_A_Multimodal_Large_Language_Model_for_Hierarchical_Whole_Slide_CVPR_2026_paper.pdf)
  aligns cell, patch, region and WSI evidence and uses cross-scale consistency
  for VQA, reasoning, captioning and report generation.

Thus `single-scale loses diagnostic evidence; cross-scale restores it` is not
a remaining contribution.  Generic Huatuo/Hulu/LLaVA probing would only move
the same mechanism to weaker, non-pathology-native models.

## 5. Candidate 3 audit: Diagnostic-Sign Likelihood-Ratio Asymmetry

The attractive clinical idea is that the presence and absence of a visual sign
need not carry symmetric diagnostic likelihood ratios.  A VLM could perceive a
sign correctly yet over-upgrade it to a definitive disease, or incorrectly use
its absence to rule disease out.  This is richer than testing the words
`yes/no`.

The public substrate does not identify the required causal test.  The
[ISIC 2018 release](https://challenge.isic-archive.com/data/) provides 2,594
training images with five attribute masks and 10,015 diagnosis rows, but no
registered same-lesion pair with one dermoscopic sign truthfully added and
removed.  Mask deletion is evidence destruction; donor or diffusion insertion
creates an image whose clinical and pathology truth must be re-adjudicated.
Natural `sign x diagnosis` cells estimate association and cannot distinguish a
VLM mechanism from disease prevalence, lesion subtype, skin type, acquisition
or annotation selection.

The non-causal version is already crowded: [VL-MedGuide](https://arxiv.org/abs/2508.06624)
uses Derm7pt concept perception followed by explainable diagnostic reasoning;
[MAKE, MICCAI 2025](https://papers.miccai.org/miccai-2025/0520-Paper0976.html)
aligns structured diagnostic concepts with visual features; and
[DermFM-Zero](https://arxiv.org/abs/2602.10624) learns sparse clinical concepts
and suppresses artifact-induced concept bias.  Consequently, a concept score,
concept bottleneck, or empirical likelihood-ratio correction would be an
incremental method without the missing reversible identification.

## 6. Other branches closed before candidacy

- Query negation and positive/negative wording are directly occupied by
  [NegBench, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Alhamoud_Vision-Language_Models_Do_Not_Understand_Negation_CVPR_2025_paper.pdf),
  which includes medical data, and by NegVQA.  This cannot be relabelled as a
  clinical rule-in/rule-out mechanism without independent likelihood-ratio
  interventions.
- Zoom/crop evidence acquisition is occupied by BCEA and [Perception Magnifier,
  ACL 2026](https://aclanthology.org/2026.acl-long.2059/), and generic ROI work
  is excluded by the project contract.
- Pathology co-occurrence counterfactuals are already studied with
  radiologist-vetted synthetic CXR counterfactuals by
  [RoentMod](https://www.nature.com/articles/s41746-026-02497-6).

## 7. Frozen consequence

No candidate in this scan passes independent truth, bidirectional intervention,
mechanism novelty and native execution simultaneously.  The correct action is:

1. do not launch Huatuo, Hulu or LLaVA-Med runs for these candidates;
2. retain EyeQ only as a future external **boundary-control dataset** for any
   independently discovered commitment mechanism, not as a new mainline;
3. do not revive magnification or diagnostic-concept branches by adding hidden
   probes, more prompts, an LLM judge, or a generative editor;
4. reopen only if a public dataset supplies a registered same-subject reversible
   intervention with independent clinical truth and a distinguishing prediction
   not already made by the closest work.

This is an ALL-NO-GO result, not evidence that no reversible static clinical law
exists.  It means no such law retrieved and audited here is currently both
identifiable and novel enough to justify scarce model runs.

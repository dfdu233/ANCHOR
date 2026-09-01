# Substrate-first bidirectional mechanism scan

**Date:** 2026-08-03  
**Mode:** public/local data and literature only; no GPU and no sealed-outcome
inspection.  The search excludes reader erasure, style/domain centers, generic
ROI/visual reliance, the current Specificity pack, missing-modality/MARC, and
generic laterality/counting.

## 0. Decision

The initial scan retained one **conditional CPU-first** candidate.  The
subsequent frozen-cardinality audit closes it, so this scan now ends in
**ALL-NO-GO**:

1. **Clinical Action-Zone Collapse (CAZC)** on RANZCR CLiP — **NO-GO after the
   frozen cardinality gate**: the published label census leaves only NGT and
   CVC with at least 100 examples in every action state; ETT-abnormal has 79,
   so the preregistered requirement of three usable relation families cannot
   be met.  The missing per-image anatomical-landmark truth is a second,
   independent failure.
2. **Temporal Sign Collapse** on MS-CXR-T — **direct-collision NO-GO** because
   bidirectional order reversal is already the method and evaluation target of
   MIDL 2025 bidirectional loss and CVPR 2026 TILA.
3. **Cardiac Phase-Arrow Collapse** on EchoNet-Dynamic — **NO-GO for the current
   paper** because current target medical models do not share a native video
   contract, while EchoMLLM and general video-hallucination work already occupy
   cyclic keyframe/temporal grounding.

No VinDr-only candidate survived.  Its local truth is strong for finding
polarity and boxes, but all bidirectional constructions available from it
reduce to the explicitly excluded reader, style, ROI, laterality, counting, or
generic spatial-binding branches.  No GPU work is authorized by this scan.

## 1. Construction logic

This scan transfers only two problem-construction moves from the mechanism
discovery exemplars:

- **ViT-style unit redefinition:** replace “a finding is present” with a more
  clinical primitive whose two directions are independently truthable:
  `device relative to action boundary` or `state change under ordered images`.
- **SigLIP-style accidental-coupling audit:** ask whether the language decision
  is unnecessarily coupled to device/finding prevalence when the clinical
  truth is a relation or signed transition.

These are analytical construction paths, not claims about how the exemplar
authors discovered their papers.

## 2. Candidate 1 — Clinical Action-Zone Collapse (post-gate NO-GO)

### Question and mechanism

> Does a medical VLM retain the device trace and landmark-relative endpoint,
> but collapse the clinically defined `normal / borderline / abnormal` action
> zones into a device-type or ICU prevalence prior when it verbalizes device
> position?

The causal variable is not generic ROI attention.  It is the signed relation
between a device endpoint/path and the device-specific anatomical action
boundary.  The three states have different clinical consequences: no
repositioning, desirable but non-urgent repositioning, and immediate
repositioning.

The distinguishing prediction is a representation/output dissociation:

1. an aligned geometry readout moves monotonically when only the distal device
   segment crosses the action boundaries;
2. final `normal/borderline/abnormal` commitment remains pinned to the native
   device prior, or skips the borderline state;
3. device-presence logits and unrelated findings remain unchanged.

The simple alternative is perception failure: if neither device endpoint nor
landmark-relative geometry is recoverable before generation, there is no
action-zone collapse to mitigate.

### Data availability

[RANZCR CLiP](https://www.nature.com/articles/s41597-021-01066-8) contains
30,083 NIH ChestXray14 radiographs from 3,791 patients, 50,612 image-level
annotations and 17,999 manual line annotations.  ETT, NET and CVC are labelled
with clinically defined normal, borderline and abnormal categories; NET also
has an incompletely imaged state.  About 30% of radiographs were double-labelled
and 10% triple-labelled, with a consensus procedure and experienced-reader
review.  Labels and polyline coordinates are distributed through Kaggle after
registration and licence acceptance.  The files are not currently local.

The released training CSV contains `PatientID`; a publicly reproduced census
reports 79/1,138/7,240 ETT abnormal/borderline/normal,
279/529/4,797 NET abnormal/borderline/normal, and
3,195/8,460/21,324 CVC abnormal/borderline/normal.  These counts are only an
access lead; the official CSV must be hashed and recounted before any claim.

The phenomenon is independently grounded: a 2026 diagnostic study evaluated
three general-purpose MLLMs on 4,813 RANZCR images and found uniformly poor
abnormal-position performance, with MCC at most 0.028 and balanced accuracy
0.41--0.53
([primary abstract](https://pubmed.ncbi.nlm.nih.gov/42279469/)).  A separate
clinical study attributes CVC errors to inability to follow the catheter path
and relate the tip to anatomical landmarks
([full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13302893/)).  These ground
the failure but do not establish the proposed representation-to-action-zone
mechanism.

### Minimal CPU certificate

Download only `train.csv` and `train_annotations.csv` first and emit a
hash-bound, patient-disjoint manifest.  No image/model run is needed for this
certificate.

1. Recount every `device x action-zone` cell and unique patients.
2. Restrict the primary set to exactly one target device/polyline and no other
   same-type device; preserve multi-device cases as a preregistered stress set.
3. Verify that annotation polylines have stable ordered endpoints, are not
   truncated at the image boundary, and correspond to the image-level state.
4. Require at least three independently usable relation families with at least
   100 patient-disjoint examples in every primary state.  ETT-abnormal is
   expected to fail this threshold and may not be rescued by oversampling.
5. Fit a dev-only nuisance classifier using device identity, number of devices,
   patient image count, line length, image dimensions and label prevalence.  A
   balanced accuracy above 0.60 kills the matched natural comparison.
6. Before creating pixels, specify a reversible endpoint/path transform
   `normal -> borderline -> abnormal` and its exact inverse while keeping the
   rest of the image and output contract fixed.  If the published polyline lacks
   enough anatomical-landmark truth to define that transform, the candidate
   stops at CPU.

The cardinality condition is already false under the published label census:
ETT contributes only 79 abnormal cases, while NGT and CVC are the only two
device families with at least 100 normal, borderline, and abnormal examples.
Because the threshold and the requirement for three relation families were
frozen before this check, neither oversampling ETT nor silently reducing the
family count is allowed.  Access to the official CSV would improve provenance
but cannot turn a 79-case class into a 100-patient class.

The final item is independently fatal: CLiP supplies device polylines and
expert action labels, but does not publish complete pixel landmark annotations
for every device.  A classifier or VLM-estimated landmark cannot define the
counterfactual truth.

### Fatal collisions

| Closest work | Occupied component | Remaining delta |
|---|---|---|
| 2026 RANZCR MLLM diagnostic study | Presence and normal/abnormal position performance, reader comparison, prompt sensitivity and error taxonomy | No bidirectional endpoint intervention or hidden geometry-to-action transition |
| Device localization/measurement models | Detect tip, carina/distance and position category | No language overcommitment mechanism, but they are mandatory geometry baselines |
| General VLM spatial-binding/localization mechanisms | Identity/location separation and causal intervention | CAZC survives only if the clinically defined three-zone action boundary, not generic location, explains the failure |
| Medical grounding and phrase-grounded correction | Finding/location verification and correction | A location-aware verifier alone is not novel; correction must preserve device identity and unrelated claims |

The collision risk is **high but not yet direct**.  The candidate becomes a
cosmetic spatial-binding replay if the result is merely “the model cannot see
the tip,” or if a deterministic landmark tool supplies the entire answer.

### GPU gate

GPU would have been authorized only if the CPU certificate passed and at least two blinded
clinicians admit at least 90% of a 120-pair edit pack as realistic, action-zone
correct and anatomy-preserving in **both** directions.  Then a 50-case,
two-model teacher-forced pilot must show:

- endpoint/landmark geometry is decodable above a locked threshold before the
  final answer;
- the final three-state output loses at least 0.05 AUROC relative to that
  geometry readout under a patient-cluster bootstrap;
- the loss is action-zone-specific, not device-presence, prompt, output-length
  or generic spatial-localization drift;
- reversing the edit reverses the action-state margin while preserving device
  identity and unrelated clinical claims.

Failure of any item kills CAZC.  The cardinality and landmark-truth items fail,
so the gate is permanently closed for this formulation.  No mitigation is
authorized by a behavior-only failure rate.

## 3. Candidate 2 — Temporal Sign Collapse (direct-collision NO-GO)

### Substrate and exact counterfactual

[MS-CXR-T](https://physionet.org/content/ms-cxr-t/1.0.0/) provides 1,326
prior/current image pairs for consolidation, edema, pleural effusion, pneumonia
and pneumothorax with `improving / stable / worsening` labels.  The release
includes label quality and pair identifiers; a subset was independently
re-annotated by a second radiologist.  Images inherit MIMIC-CXR access terms;
the current unauthenticated file endpoint returns HTTP 403, so project-specific
credential access must be checked before calling it executable.

The counterfactual is unusually clean and exact: swap the two image identities.
`improving <-> worsening`, while `stable -> stable`.  It changes no pixels,
finding identity, image quality or claim count.

The minimal CPU certificate would hash the pair IDs, validate chronological
order, invert labels, count patient-disjoint `finding x state x direction`
cells, and require at least 100 units in each primary cell.  No synthetic image
admission is needed.

### Why it is killed

This mechanism and intervention are already occupied.  The MIDL 2025 paper
[A Bidirectional Loss Approach to Imparting Order Sensitivity to Multi-Image
Chest X-ray Encoders](https://openreview.net/forum?id=fEAAi7wwXb) explicitly
reports that models yield inconsistent predictions after reversing image order,
trains on reversed pairs with inverted labels, and evaluates improved order
sensitivity.  CVPR 2026
[TILA](https://arxiv.org/abs/2604.04563) incorporates temporal inversion in
pretraining, fine-tuning and inference and introduces an inversion-aware
evaluation protocol.  It even discusses the limits of clinical symmetry for
recovery versus worsening.

Layerwise probing, a larger VLM, OE language or stricter fixed-coverage controls
would not change the mechanism-level collision.  They would be useful analysis
inside a temporal CXR paper, not a new hallucination mechanism.

**GPU gate: permanently closed for this proposal.**  TILA is a baseline/data
resource, not a candidate mainline.

## 4. Candidate 3 — Cardiac Phase-Arrow Collapse (current-paper NO-GO)

### Substrate and counterfactual

[EchoNet-Dynamic](https://echonet.github.io/dynamic/) releases 10,030 apical
four-chamber echocardiography videos under a free non-commercial research-use
agreement.  It includes clinically obtained and expert-verified ejection
fraction, end-systolic volume, end-diastolic volume, and endocardial tracings at
end-systole/end-diastole.  Official code is MIT-licensed
([repository](https://github.com/echonet/dynamic)).

An exact two-frame counterfactual uses the same expert ED and ES frames:

```text
ED -> ES : contraction / decreasing LV volume
ES -> ED : filling / increasing LV volume
```

The pixels are identical as a set; only temporal role is reversed.  EF magnitude
is an invariant control.  This separates phase identity, motion direction and
quantitative function more cleanly than a generated lesion edit.

The minimal CPU certificate is to obtain `FileList` and `VolumeTracings`, verify
two distinct expert frame indices per video, reject duplicate/ambiguous phases,
count patient-unique EF strata and create byte-identical forward/reverse frame
pairs.  Require at least 300 videos per EF stratum and audit whether the selected
frames truly belong to a single contiguous cycle.

### Fatal collision and execution mismatch

[EchoMLLM, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1001/)
already defines cyclic temporal ambiguity as a central echocardiography MLLM
problem, introduces cycle/pathology-conditioned keyframe grounding and report
generation, and uses cycle-aware reinforcement learning.  General
[VidHalluc, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_VidHalluc_Evaluating_Temporal_Hallucinations_in_Multimodal_Large_Language_Models_for_Video_Understanding_CVPR_2025_paper.pdf)
already evaluates temporal sequence hallucinations.  The remaining exact ED/ES
order test is an elegant diagnostic, but currently a boundary case rather than
an oral-level mechanism.

More importantly, HuatuoGPT-Vision and LLaVA-Med do not share a native video
input contract; converting ED/ES to a left/right panel turns temporal-role
reasoning into already-collided generic spatial/order binding.  A result from
one video-capable model cannot satisfy the project's two-model native-output
gate.

**GPU gate: closed for the current paper.**  Reopen only if two medical
video-MLLMs with native ordered-frame inputs and accessible checkpoints are
frozen, and a second-pass collision audit shows a prediction not already made
by EchoMLLM or temporal-order benchmarks.  Merely showing that reversal changes
the answer is insufficient.

## 5. Ranked hard-gate table

Scores use `I/M/N/E` in `[0,3]` and
`0.30 I + 0.30 M + 0.20 N + 0.20 E`; a hard-gate failure overrides the score.

| Rank | Candidate | I | M | N | E | Score | Decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | CAZC / RANZCR | 3 | 3 | 2 | 2 | 2.6 | **NO-GO**; only two three-state families meet the frozen count and landmark/edit truth is absent |
| 2 | Cardiac Phase-Arrow / EchoNet | 2 | 2 | 1 | 1 | 1.6 | **NO-GO current paper**; model-contract and collision failures |
| 3 | Temporal Sign / MS-CXR-T | 3 | 2 | 0 | 2 | 1.9 | **Direct-collision NO-GO** despite excellent substrate |

The numeric ranking records the pre-gate attractiveness of the questions, not
their current viability.  After enforcing the frozen requirements, all three
branches are NO-GO.  Reopening CAZC requires a different dataset that already
contains three adequately powered device families *and* independently verified
device-to-landmark geometry; changing the current thresholds is prohibited.

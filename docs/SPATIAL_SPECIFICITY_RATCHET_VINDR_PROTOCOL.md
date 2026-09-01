# Spatial Specificity Ratchet: VinDr reader-box feasibility protocol

## Decision

The current VinDr R8/R9/R10 boxes **do not authorize a Spatial Specificity
Ratchet experiment**.  They provide a clean image-coordinate agreement screen,
but not yet independent clinical truth for `unilateral/bilateral` or
`upper/lower lung zone`.

The read-only audit found 5,112 localized findings that all three fixed readers
called positive, spanning 3,537 images.  Every box matched its image-level
positive label and every coordinate lay within the DICOM dimensions.  However,
only pulmonary fibrosis passed the frozen two-class count/consistency screen;
the progression rule requires at least two findings.  Patient/study identifiers
and all local orientation tags are absent.  Consequently the split is
image-disjoint but patient-disjointness cannot be verified.

This is a useful negative result: the attractive raw counts do not by themselves
make a clinical modifier benchmark.

## Frozen question and claim ceiling

The admissible question is:

> Conditional on three readers agreeing that a finding is present, do their
> independently drawn boxes support a reproducible *image-coordinate extent*
> attribute that could later be clinically admitted as a nested claim?

The current artifact can support only:

> R8/R9/R10 agree or disagree about whether their boxes occupy one versus both
> image hemifields, or one versus multiple normalized image-height regions.

It cannot support patient left/right, unilateral/bilateral disease, upper/lower
lung zone, lobe, etiology, severity, or a VLM hallucination claim.  No ontology
edge is inferred from finding co-occurrence and no model output, RadGraph label,
LLM judge, detector, or generated grounding is used as truth.

## Coordinate definitions

Only claims with `3/3` parent-finding support and valid boxes from all of R8,
R9, and R10 enter the screen.

- `single_image_hemifield`: all of one reader's boxes remain on one side of a
  five-percent guarded image midline.  The side identity is discarded, making
  the label invariant to horizontal reflection.
- `both_image_hemifields`: separated lateral box centers or one box spanning
  both outer hemifields.  This is box coverage, not clinical bilaterality.
- `upper_image_region`, `lower_image_region`, and
  `multiple_image_height_zones`: strict normalized image-height labels with a
  central ambiguity band.  These are not lung regions or lobes.
- Borderline, central, and crossing layouts are `ambiguous`; they are never
  forced into a class.

The primary findings were frozen before looking at the derived split counts:
lung opacity, nodule/mass, pleural effusion, and pulmonary fibrosis.  Aortic
enlargement and cardiomegaly are excluded because crossing the image midline is
not a clinically meaningful bilateral attribute.  Other lesion and
calcification are too semantically heterogeneous for this claim.

## Source and geometry audit

The official VinDr-CXR description says the released set contains PA chest
radiographs, three independent annotations for every training image, and local
finding boxes ([PhysioNet](https://physionet.org/content/vindr-cxr/1.0.0/)).
The local files are stricter in a different sense: `PatientID`,
`StudyInstanceUID`, `ViewPosition`, `PatientOrientation`, `Laterality`,
`ImageLaterality`, and `ImageOrientationPatient` are non-null in `0/3,537`
audited DICOM headers.  Therefore:

- PA view is a dataset-source contract, not inferred from local tags;
- anatomical left/right is forbidden;
- image-level hashing is reproducible, but patient-level leakage cannot be
  audited from this release.

The fixed `20/20/60` SHA-256 image split contains 994 pilot, 1,026 development,
and 3,092 test claim rows.  A repeated image always remains in one split across
all findings.

## Label-only result

| Finding | 3/3 parent positive | Fully definite horizontal | Unanimous one hemifield | Unanimous both hemifields | Coordinate gate |
|---|---:|---:|---:|---:|---|
| Lung opacity | 162 | 160 | 122 | 5 | fail: both class underpowered |
| Nodule/mass | 167 | 165 | 113 | 27 | fail: both class underpowered |
| Pleural effusion | 425 | 423 | 313 | 38 | fail: pilot/dev both class underpowered |
| Pulmonary fibrosis | 617 | 615 | 324 | 116 | pass |

The coordinate screen requires, per finding, at least 75% fully definite reader
labels, at least 65% unanimity among fully definite cases, and at least 10/10/20
unanimous examples of *each* horizontal class in pilot/dev/test.  Two findings
must pass; only one passed, so the branch fails before model inference.

Vertical labels are weaker and clinically less defensible.  Fully definite
fractions are 42.0% (opacity), 71.3% (nodule/mass), 58.8% (effusion), and 74.9%
(fibrosis).  Their observed image-height imbalance must not be retold as an
upper/lower lung-zone result.

## Why box style is a serious confound

The horizontal label is almost perfectly coupled to how a reader chose to draw
boxes.  Among the four primary findings, every `both_image_hemifields` reader
annotation except one uses multiple boxes; most `single_image_hemifield`
annotations use one box.  This can reflect genuine bilateral extent, but it can
also reflect lesion multiplicity, box granularity, or a reader's annotation
style.  The present CSV cannot separate these causes.

Other required controls are:

- stratify by reader, number of boxes, box area, central crossing, and multiple
  lesion instances;
- never let a target-dependent crop or segmentation define its own truth;
- retain ambiguous cases in coverage denominators;
- treat the official PA-view statement as a source-level constant and report
  the absent local view tags;
- do not call the hash split patient-disjoint until a patient mapping exists.

## Hard admission gates before any GPU experiment

All must pass without changing thresholds after viewing model outputs:

1. At least two frozen findings pass the coordinate count/consistency gate.
2. A source-authorized patient grouping produces patient- and image-disjoint
   splits; substituting `patient_id=image_id` is forbidden.
3. Two blinded chest radiologists independently label at least 50 examples per
   proposed clinical modifier class.  `single/both image hemifields ->
   unilateral/bilateral` must achieve class-wise PPV at least 0.90 and agreement
   kappa at least 0.70, including separate one-box and multi-box strata.
4. Any upper/lower claim must be referenced to independently validated lung
   anatomy, not raw image height.  Lobar wording remains forbidden on a frontal
   radiograph unless separately labeled.
5. Reader identity, box count, box area, and centrality alone must not explain
   the admitted attribute.  A box-style-only grouped predictor is a mandatory
   baseline.
6. The clinical admission and thresholds are frozen on pilot/development; test
   is not used for edge selection.

Until these gates pass, no teacher-forced layer probe or mitigation run is
authorized.

## Mechanism test if the substrate is later repaired

For a clinically admitted edge `finding present -> finding + spatial modifier`,
the ratchet prediction is not simply that the child is wrong.  Conditional on
the parent being supported, the decoder's child-minus-parent commitment should
increase toward the modifier token while image evidence for the added
constraint does not increase.  A causal test must preserve parent identity and
polarity while changing only the incremental modifier commitment.

For OE, evaluate at fixed positive-finding count `K`:

1. extract a draft's parent findings and explicit modifiers;
2. score modifier overrefinement against the independent reader distribution;
3. back off only an unsupported modifier, never delete its parent finding;
4. keep `K`, finding coverage, polarity, answer budget, and refusal rate fixed;
5. report modifier precision/recall, parent retention, omission, answer length,
   and clinical usefulness.

This design can test overrefinement without winning by silence.  It still needs
to beat matched-compute full-output controls and the closest editing/grounding
baselines.

## Collision audit

- [FINER (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_FINER_MLLMs_Hallucinate_under_Fine-grained_Negative_Queries_CVPR_2026_paper.html)
  already establishes hallucination under subtle fine-grained negative
  queries.  Our only possible delta is a generation-time parent-to-child
  escalation mechanism under independent clinical truth.
- [ZINA (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Wada_ZINA_Multimodal_Fine-grained_Hallucination_Detection_and_Editing_CVPR_2026_paper.html)
  already detects, types, and edits fine-grained hallucinated spans.  Span
  detection or generic modifier deletion is not novel.
- [CounterVHD (arXiv:2606.28520)](https://arxiv.org/abs/2606.28520) already
  extracts medical entities and uses factual/counterfactual grounding
  uncertainty.  An external grounder over spatial attributes is therefore a
  baseline, not the contribution.
- [CEBC (ACL 2026)](https://aclanthology.org/2026.acl-long.2142/) already uses
  conformally calibrated detector evidence for minimal editing.  A thresholded
  modifier backoff method directly collides.

No mechanism-equivalent work was retrieved under these searches for the narrow
claim of *autoregressive child-over-parent commitment growth measured against
independent reader-level spatial support*.  That novelty space remains only if
the clinical truth gates above are first satisfied.

## Reproduction

The derived manifest is restricted-data metadata and remains outside Git:

```bash
PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
  anchor/corrected_sgta/prepare_vindr_spatial_reader_manifest_v1.py \
  --labels-csv /home/dbw/datasets/physionet/vindr-cxr/1.0.0/annotations/image_labels_train.csv \
  --bbox-csv /workspace/vinbigdata/train.csv \
  --image-root /workspace/vinbigdata/train \
  --output-dir /home/dbw/datasets/physionet/vindr-cxr/1.0.0/spatial_reader_v1
```

Artifacts:

- `spatial_reader_manifest_v1.jsonl`: deterministic reader-distribution rows;
- `summary_v1.json`: exact source hashes, coordinate/metadata audit, split
  counts, per-finding counts, frozen gates, and explicit formal blockers.

Focused validation:

```bash
PYTHONPATH=anchor /opt/miniconda3/envs/huatuo/bin/python \
  -m pytest -q tests/test_vindr_spatial_reader_manifest_v1.py
```

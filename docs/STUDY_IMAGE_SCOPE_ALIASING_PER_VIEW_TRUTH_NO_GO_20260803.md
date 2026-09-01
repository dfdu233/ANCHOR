# Study–Image Scope Aliasing: independent per-view truth audit

**Frozen:** 2026-08-03 UTC  
**Mode:** annotation/protocol audit and CPU joins only; model outputs unopened;
no GPU  
**Decision:** **STRICT NO-GO.**  No legally usable current source supplies the
paired, claim-matched three-state per-view truth required by the frozen gate.

This audit does not infer a negative or an unobservable state from a missing
box, report label, scene-graph node, classifier score, VLM answer, or judge
output.  A positive box proves visibility only for the annotated image.  It
does not determine the state of a sibling view.

## Frozen admission contract

A source is admissible only if it can produce all of the following without
reading any model output:

1. a stable patient, study, image/view and clinical-claim identity;
2. an independent expert state for **each member of a paired view set**:
   `visible-support`, `visible-refute`, or `unobservable`;
3. at least 100 paired claim–view cells;
4. at least two findings with at least 30
   `one-view-unobservable / other-view-visible` cells apiece, plus matched
   all-view-clear controls;
5. patient- and study-disjoint development/test partitions;
6. a license or DUA path that authorizes the local use.  An unofficial mirror
   cannot override the source dataset's agreement.

The first two requirements are construct gates.  Large aggregate dataset
counts cannot compensate for failing them.

## Reproducible local result: Tam et al. annotations

The strongest locally available public CXR finding boxes were joined by exact
DICOM UUID to the complete MIMIC-CXR metadata table.

| Input | SHA-256 | Size |
|---|---|---:|
| `221v2hiqualnihsplit.json` | `8b80ca98ac6b28ccd00a897c863a57acb9b519025acbe565bff307a8446df668` | 692,582 B |
| `mimic-cxr-2.0.0-metadata.csv.gz` | `6a3748ce77724c0dfe7d2def8f47643e989e3bbf0795bc13b89c1578e1649d6b` | 16,546,905 B |

The join is complete and one-to-one:

- 354 annotated images, 458 boxes and two findings;
- 154 pneumonia-only images, 187 pneumothorax-only images and 13 images with
  both finding categories;
- 255 AP and 99 PA images; no annotated lateral view;
- 249 patients and **354 distinct studies**;
- **zero studies contain two annotated images**.

Consequently this source admits 354 positive image–finding examples but
**zero paired labelled studies and zero paired ambiguity cells**.  Joining an
unannotated same-study image would not fix the gate: absence from the released
box set is not an expert state for that sibling image.  The public annotation
release is therefore useful positive grounding gold, but cannot test
Study–Image Scope Aliasing.

## Candidate audit

| Source | What is genuinely independent | Fatal gap for this question | Gate |
|---|---|---|---|
| [MS-CXR v1.1.0](https://physionet.org/content/ms-cxr/1.1.0/) | 1,162 radiologist-verified positive phrase–box pairs over eight findings | The release is a positive grounding benchmark, not exhaustive annotation of every sibling image.  No explicit paired-view `refuted/unobservable` state is published.  Its files are PhysioNet credentialed and currently absent locally. | **NO-GO** |
| [REFLACX v1.0.0](https://physionet.org/content/reflacx-xray-localization/1.0.0/) | 2,616 frontal CXRs with radiologist image-level certainty and mandatory abnormality ellipses | The protocol first restricted sampling to studies with only one frontal CXR and the radiologist saw that frontal image.  A lateral or other sibling, if present in MIMIC, was not read under the same protocol.  It has no paired-view truth. | **NO-GO** |
| [Chest ImaGenome v1.0.0](https://physionet.org/content/chest-imagenome/1.0.0/) | Manually corrected anatomy boxes on 1,000 frontal images and a 500-patient gold audit | Finding attributes/relations originate in reports and NLP; manual image annotation primarily corrects anatomy boxes.  The first and second exams are longitudinal studies, not complementary views of one study.  Missing attributes/boxes are explicitly ambiguous. | **NO-GO** |
| [PadChest-GR](https://arxiv.org/abs/2411.05085) | Expert-curated image-grounded positive findings, negative finding sentences and up to two reader box sets | The release does not publish a claim-linked state saying that a same-study sibling view is unobservable.  A negative sentence is clinical absence, not invisibility on one projection.  Official access is request-only under the PadChest agreement. | **NO-GO** |
| [VinDr-Mammo v1.0.0](https://physionet.org/content/vindr-mammo/1.0.0/) | 5,000 four-view exams, double-read/arbitrated breast assessment, and image-specific finding boxes | Public schema has no stable lesion/finding identity linking CC and MLO boxes, and no exhaustive per-view visibility state.  Breast BI-RADS is attached to both images of a breast and is therefore breast-scoped; a missing box in the other view is not a negative. | **NO-GO** |
| REFLACX-derived/MIMIC-EYE tables | Exact MIMIC IDs and image-level labels for the displayed frontal image | Integration adds metadata, not a second independent reader decision on the sibling image. | **NO-GO** |

### Access and mirror boundary

Unauthenticated official file endpoints for MS-CXR, REFLACX, Chest ImaGenome
and VinDr-Mammo returned HTTP 401 on 2026-08-03.  This is not a scientific
reason to reject them, but it prevents a local cardinality audit.  More
importantly, their public protocols already show that downloading the files
would not create the missing paired three-state field.

PadChest's research-use agreement prohibits redistributing the dataset or its
download link.  Public third-party Hugging Face copies were found, including a
copy that declares a permissive license, but no source-owner authorization was
retrieved.  They were not downloaded or admitted: a mirror uploader cannot
relicense restricted clinical data.  No DUA was bypassed.

## Mammography does not rescue the construct

VinDr-Mammo is the largest natural multi-view candidate, but its two annotation
levels expose the same scope problem rather than solve it:

```text
breast-level BI-RADS  -> shared across CC and MLO
image-level box       -> positive visibility on that image only
missing box           -> unknown, not refuted/unobservable
```

This is especially unsafe because incomplete mammography detection labels are
an established methodological problem; treating unlabeled lesions as
background is exactly the assumption addressed by work such as
[BRAIxDet](https://arxiv.org/abs/2301.13418).

Mammography also fails directional admission for the present model family.
[MammoVQA](https://www.nature.com/articles/s41467-025-66507-z) evaluates six
general and six medical LVLMs, including LLaVA-Med-7B, and reports that most
models are near random on mammographic interpretation.  The same paper already
separates image-scoped from exam-scoped labels and uses multi-image inputs for
exam questions.  Thus generic claims that "label scope matters" or
"multi-view input helps" are occupied, while a decoder-level mitigation on
Huatuo/Hulu/LLaVA-Med would be uninterpretable until a mammography-specific
model passes lesion-level visual admission.  No such admission result exists
here.

## CT slice–volume fallback

Volume-level reports/labels in CT-RATE-like data cannot be copied to slices as
truth.  Key-slice resources such as DeepLesion provide a positive lesion mark
on a selected slice but are not exhaustive `refuted/unobservable` audits of all
other slices.  Exhaustive 3-D segmentations can define mask intersection with
a slice, but that is a geometric operator label rather than physician-admitted
clinical observability; it also changes the paper from study/report scope
aliasing to established slice/volume MIL and 3-D VLM selection.  No reviewed
source found here simultaneously supplies a natural report claim, exhaustive
per-slice clinical visibility, and two adequate finding families.  CT is
therefore not an admissible fallback.

## Collision boundary

The broad premise is already covered from several directions:

- REFERS and multi-view RRG treat the study, rather than a single radiograph,
  as the image–report supervision unit.
- KCLVA and View-PNDF explicitly study view-specific report content or
  view-specific neurons; "different views produce different answers" is not a
  remaining contribution.
- MammoVQA explicitly distinguishes image-, breast- and exam-scoped labels and
  evaluates multi-image exam input.
- multiple-instance/multiple-label learning already formalizes why a bag label
  cannot be copied to every instance; incomplete-annotation mammography work
  makes the same warning operational.

No mechanism-equivalent work was retrieved for the narrower conjunction of
independent three-state per-view clinical truth, a fixed-claim evidence-set
counterfactual, and image-versus-study hallucination identifiability.  That
delta is real but **unexecutable on all audited sources**.  A missing substrate
is not a method contribution.

## Frozen decision

| Requirement | Admitted evidence | Decision |
|---|---:|---|
| paired claim–view cells | 0 | fail (`>=100` required) |
| findings with `>=30` ambiguity cells | 0 | fail (`>=2` required) |
| explicit support/refute/unobservable | none | fail |
| patient/study-disjoint split | feasible for several datasets | insufficient |
| directional model admission | contradicted for LLaVA-Med on mammography; absent elsewhere | fail |

**Study–Image Scope Aliasing is closed as a current experimental branch.**
There is no GPU authorization, no exact-parent training authorization, and no
current physician-pack launch.  A future reopening would require a genuinely
new blinded annotation study in which at least two radiologists independently
read every view in isolation and assign the three frozen states to stable
claim identities, followed by adjudication and the original cardinality audit.
Until those labels exist, neither report copying nor missing boxes may be used
to manufacture the desired effect.

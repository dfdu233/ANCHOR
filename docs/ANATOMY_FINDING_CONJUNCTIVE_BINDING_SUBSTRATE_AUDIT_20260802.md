# Anatomy–Finding Conjunctive Binding substrate audit

**Date:** 2026-08-02  
**Mode:** local read-only inventory plus official primary-release audit; CPU only  
**Decision:** **F6 KILL + F7 KILL for the current workspace. No GPU run and no
formal manifest are authorized.**

This decision does not reject the binding hypothesis. It rejects the claim that
the hypothesis can presently be tested with true patient-side/location gold.
Synthetic flips, bbox-centre heuristics, report parsers, and LLM judges are not
accepted substitutes.

## Fatal-gate decision

### F6 — construct/gold validity: KILL now

No local dataset supplies an expert-admitted tuple

`(patient_id, image_id, finding, patient-side/location, finding support, box/anatomy)`.

The available VinDr DICOMs have no patient-side metadata, as established by the
previous full 15,000-image audit. The local IU-Xray and MedHEval assets do not
contain a systematic, linkable, expert finding-by-side reference. Pixel x
coordinates cannot define patient laterality, and an exact word such as
`left` in unreviewed report text cannot by itself define visual truth.

PadChest-GR would satisfy the construct conditionally because its categorical
location labels are linked to individual finding sentences and its positive
findings were reviewed and boxed by a 14-radiologist team. Chest ImaGenome's
manual gold object–attribute relations are also a defensible anatomy-binding
replication substrate. Neither annotation release is locally available. MS-CXR does not publish a
structured patient-side field: its radiologist-verified phrase and box are
excellent grounding gold, but deriving side would still require a separately
admitted literal/location mapping.

### F7 — access/cardinality: KILL now

- Local annotations for PadChest-GR, Chest ImaGenome, and MS-CXR are all
  absent. There are 1,677 local MIMIC-CXR JPGs whose DICOM IDs can be joined
  exactly to future Chest ImaGenome/MS-CXR annotations, but the inaccessible
  annotation IDs are needed to determine the actual overlap.
- PadChest-GR is a 46 GB request-only download. Its official paper publishes
  global `right/left/bilateral` location counts, but no finding-by-side-by-split
  cross-tab. Therefore the required three findings with at least 100
  patient-disjoint cases on each side cannot be certified from the paper.
- Chest ImaGenome and MS-CXR require PhysioNet credentialing, CITI training,
  and a signed project DUA; matching MIMIC-CXR images must also be obtained
  under the applicable credentialed license. No credentials were entered or
  tested in this audit.
- MS-CXR is too small for the frozen gate even before side stratification: its
  patient-level test split has only 8–53 total examples per finding. Splitting
  any finding into left and right can only reduce those cells.

These are short-circuit failures. No threshold relaxation, automatic side
inference, or GPU exploratory run is allowed.

## Reproducible local inventory

Search roots were `/home/dbw/ANCHOR`, `/home/dbw/datasets`, and `/workspace`.
The exact release filenames and dataset-name variants returned no Chest
ImaGenome, MS-CXR, or PadChest-GR annotation file and no matching image tree.

| Candidate | local annotation | locally linkable candidate images | usable finding×side rows |
|---|---:|---:|---:|
| PadChest-GR | 0 | 0 | 0 |
| Chest ImaGenome gold/silver | 0 | 1,677 MIMIC JPG candidates; overlap unknown | 0 |
| MS-CXR v1.1.0 | 0 | same 1,677 MIMIC JPG candidates; overlap unknown | 0 |

The current filesystems have about 360 GB free. A 46 GB PadChest-GR download is
storage-feasible while retaining more than 100 GB free, but access approval and
the missing cross-tab remain scientific prerequisites.

### Exact local MIMIC ID join feasibility

`data/medheval/images/pXX/pXXXXXXXX/sXXXXXXXX/<dicom-id>.jpg` contains exactly
1,677 JPGs from 282 patients and 1,515 studies (2,882,216,078 bytes). All 1,677
filename stems are unique, all match the five-group MIMIC DICOM UUID form, and
there are no duplicate paths. The SHA-256 of the sorted canonical
`patient_id,study_id,dicom_id,relpath,bytes` index is
`cd8674442641914f9224f1e42f519f9aeb171f616d6948c7cdb30fb20300de2c`.
This is a technically clean join substrate:

1. join Chest ImaGenome `image_id` or MS-CXR `file_name` stem to the local JPG
   stem exactly;
2. independently verify the local `pXXXXXXXX` and `sXXXXXXXX` components
   against annotation patient/study IDs;
3. fail on any ID disagreement, duplicate, missing checksum or cross-split
   patient;
4. use only Chest ImaGenome/MS-CXR expert annotations as truth.

The present exact overlap with the Chest ImaGenome 500-patient gold set and
MS-CXR's 1,162 pairs is **unknown**, because neither annotation ID list is
locally accessible. It must not be estimated from filenames, reports, MedHEval
questions or model answers. MedHEval annotations are never location truth.
Prior MedHEval/RULE use of a joined image must be recorded in the future
manifest so dev/test selection cannot depend on already observed model outputs.

## Candidate-by-candidate audit

### 1. PadChest-GR — conditional GO for metadata admission only

Primary sources: [official BIMCV release page](https://bimcv.cipf.es/bimcv-projects/padchest-gr/),
[PadChest-GR paper v2](https://arxiv.org/pdf/2411.05085).

Verified facts:

- 4,555 studies: 3,185 train, 455 validation, and 915 test. The paper states
  that all studies from one patient remain within one partition.
- 7,037 positive and 3,422 negative finding sentences; 6,217 positive findings
  have official boxes and 820 do not.
- A team of 14 radiologists performed quality control and finding box
  annotation. Positive findings can have up to two independent reader box sets.
- Finding-level categorical locations exist. Published global counts include
  `right: 845 images / 1,207 boxes`, `left: 645 / 896`, and
  `bilateral: 419 / 929`.
- Important large finding groups include cardiomegaly 500, pleural effusion
  419, nodule 409, atelectasis 263, pleural thickening 242, interstitial pattern
  220, and alveolar pattern 204 studies. These totals are **not** side cells.
- Access is free for research but request-only through an official Google form;
  the official page reports a 46 GB download and requires the PadChest Research
  Use Agreement. The agreement prohibits redistribution and re-identification.

What remains unknown until the official annotations are obtained:

- exact `finding × left/right × split × unique patient` counts;
- whether a location list can contain conflicting or nested side labels;
- exact patient, study, image and reader identifiers in the release schema;
- duplicate findings within studies and duplicate patients across candidate
  cells;
- box-to-location consistency and whether all selected unilateral findings
  have at least one official box.

Hence PadChest-GR is **not a scientific GO**. It is the only current
**access-first candidate**. The paper's aggregate 845/645 side counts must not
be distributed across findings by assumption.

### 2. Chest ImaGenome gold — conditional replication candidate

Primary source: [PhysioNet Chest ImaGenome v1.0.0](https://physionet.org/content/chest-imagenome/1.0.0/).

Verified facts:

- The dataset includes `patient_id`, `study_id`, `image_id`, AP/PA viewpoint,
  anatomy objects and object–attribute relations. Object names explicitly
  include patient-side anatomy such as `left upper lung zone`.
- A manual gold set contains corrected object–attribute relations for the first
  study of 500 unique patients. Anatomy boxes were dual-annotated for 1,000
  images from the first and second studies of those patients.
- Published gold relation frequencies include left lung 1,453 and right lung
  1,436; left/right lower zones 609/580 and left/right hilar structures
  571/572. These are relation frequencies, **not finding-by-side unique-patient
  counts** and cannot establish the cell quota.
- The released train/validation/test split is patient-level. Gold patients are
  explicitly excluded from silver training/validation via `images_to_avoid`.
- Access is restricted to credentialed users with CITI training and a signed
  DUA. The underlying MIMIC-CXR images are a separate credentialed dependency.

Chest ImaGenome gold is preferable to silver for truth. Silver scene graphs
are automatically constructed and may only support sensitivity analysis. Even
after access, a 500-patient gold set may fail the three-findings-by-two-sides
quota; no paper table resolves this.

### 3. MS-CXR — KILL as primary binding substrate

Primary source: [PhysioNet MS-CXR v1.1.0](https://physionet.org/content/ms-cxr/1.1.0/).

Verified facts:

- 1,162 radiologist-verified phrase–box pairs from 851 subjects covering eight
  findings. The release has COCO JSON/CSV annotations but images must be
  separately downloaded from MIMIC-CXR/JPG.
- The official 70:15:15 split is patient-level and stratified by finding and
  gender.
- Test totals before side stratification are: atelectasis 8, cardiomegaly 53,
  consolidation 15, edema 8, lung opacity 12, pleural effusion 14, pneumonia
  30, and pneumothorax 36.
- Annotation schema contains category, phrase, box, image and split, but no
  published categorical patient-side field or side cross-tab.
- Access requires PhysioNet credentialing, CITI training and DUA.

MS-CXR cannot meet a three-finding, 100-patient-per-side gate. Even under the
impossible best case in which every pair were unilateral and perfectly balanced,
a finding needs at least 200 total pairs. Only cardiomegaly (333) and
pneumothorax (245) reach 200, so at most two findings can qualify; their
locked-test totals are at most 53 before dividing by side. MS-CXR remains a
useful phrase-grounding auxiliary benchmark, not the primary conjunctive
binding substrate.

## Minimum admissible manifest after PadChest-GR access

The builder must fail closed unless the source schema provides every field
below without deriving clinical truth from free text or image geometry:

```text
patient_id
study_id
image_id
official_split
image_relpath + checksum
finding_id + frozen parent category
finding_polarity = present
location_labels_raw
patient_side = left | right
side_provenance = official_categorical_location
official_bbox_set(s) + reader_id(s)
view_position
prior/current role
```

Admission rules:

1. Keep only exactly one official unilateral side: left XOR right. Exclude
   bilateral, both-side, side-unspecified, conflicting, dextrocardia/situs and
   non-lateral anatomy.
2. Require at least one expert box attached to the same positive finding;
   never infer patient side from the box centre.
3. Deduplicate at patient level before counting. Preserve the official
   patient-disjoint split and checksum every image/annotation input.
4. Freeze finding parents and all exclusions before reading model outputs.
5. Require at least three findings with both left and right cells containing
   at least 100 distinct patients. Report train/dev/test counts separately;
   do not pool the locked test to rescue a dev shortage.
6. Exclude progression/history-only claims unless the current image was
   independently annotated as supporting the finding.
7. Emit no manifest when any quota or provenance field fails. An automatic
   parser or LLM may inspect schema but may not create, correct or adjudicate
   finding, polarity, side, or support.

The first executable action after legitimate access should be a CPU-only schema
and cell-count audit. It must produce only counts and exclusion reasons. A VLM
runner is authorized only after that audit yields a formal GO.

## Estimated cost

| Stage | Data/storage | CPU/wall time | GPU | decision |
|---|---:|---:|---:|---|
| Current local audit | none | completed, minutes | 0 | KILL |
| PadChest-GR access/download | 46 GB official package; reserve at least 100 GB free | approval latency unknown; download about 8–80 min at 100–10 MB/s; verify/extract 0.5–2 h | 0 | metadata-only |
| PadChest-GR schema/cell audit | no derived image cache | about 5–30 min | 0 | GO/KILL gate |
| Chest ImaGenome gold audit | small annotation archive plus selectively fetched 500 gold images; exact size unavailable before access | 0.5–2 h after access | 0 | replication gate |
| MS-CXR audit | small annotations; reuse only exact-ID matches among 1,677 local JPGs, selectively fetch the remainder of up to 1,162 images | under 1 h after access | 0 | auxiliary only |
| Minimal admitted behavior/probe screen | 600–1,200 real images, two models | preprocessing 1–3 h | roughly 4–8 GPU-h | only after GO |
| Layerwise interchange + OE validation | same images plus matched pairs, third model | additional analysis | roughly 12–24 GPU-h | paper gate |

Approval latency is external and cannot be honestly estimated. The current
workspace has enough disk for PadChest-GR, but downloading it without an
approved request and accepted agreement is not authorized.

## Final decision

- **PadChest-GR:** `F6 conditional pass / F7 current KILL`; request-and-count
  candidate only.
- **Chest ImaGenome gold:** `F6 conditional pass / F7 current KILL`; replication
  candidate only.
- **MS-CXR:** `F6 unresolved side schema / F7 cardinality KILL`; auxiliary only.
- **Current Anatomy–Finding Conjunctive Binding experiment:** **KILL.** No
  minimal truthful manifest exists locally, so no GPU or causal claim is
  permitted.

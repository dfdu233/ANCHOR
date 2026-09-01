# SISC outcome-blind truth/substrate gate

> Frozen: 2026-08-03 UTC. Decision: **NO-GO before model outcomes or GPU**.
> This audit did not inspect either completed report-generation output file and
> did not modify the shared evaluation system.

## Frozen question and admission rule

Study--Image Supervision Collision (SISC) is only a viable mechanism if a
study-level report copied onto individual images creates positive supervision
for claims that an independent expert says are not assessable, or are refuted,
on that particular image. The preregistered one-day gate required all of:

1. at least 100 paired-view studies;
2. at least three findings;
3. at least 30 independently established view-exclusive or view-unassessable
   study--claim instances per finding;
4. explicit separation of `visible`, `refuted`, and `unassessable`;
5. patient/study-disjoint splitting.

A shared study report can audit target duplication, but cannot define
image-level truth. Absence of a grounding box is not a negative label and is
not an `unassessable` label. These locks were implemented before inspecting any
model outcome.

## Local substrate result

The source is `data/mmedrag/test/report/mimic_test.json`, SHA-256
`be8b34a034bf96639596f973bd26abeaf964a113b596e4c0d4bc06f3cd8a0fe7`.
The outcome-free compiler finds:

| Quantity | Result |
|---|---:|
| image rows / unique images | 700 / 700 |
| images present locally | 694 |
| unique studies | 653 |
| unique subjects | 219 |
| studies with at least two image rows | **44** |
| rows in those studies | 91 |
| maximum images per study | 3 |
| rows with official view metadata locally | **0** |
| within-study report-target identity | 100% |

The 44 paired studies establish that a study target was duplicated onto image
rows, but they are below the frozen 100-study floor and cannot establish that
any duplicated claim was wrong for a view. A deterministic patient-level
70/15/15 split produces zero subject and zero study leakage, so splitting is
not the limiting factor.

## Independent truth inventory

### MS-CXR

The official release describes 1,162 radiologist-verified image--sentence box
pairs over eight findings. It is a useful source of *positive visible*
groundings. It is not a complete per-study annotation of every sibling image,
and therefore an image with no MS-CXR box cannot be marked `refuted` or
`unassessable`. The annotations were not present locally; the authenticated
file endpoint also returned HTTP 403 under the current regional/access policy.
Even successful access would not by itself satisfy the three-state incidence
contract. See the [official MS-CXR description](https://physionet.org/content/ms-cxr/1.1.0/).

### Tam et al. MIMIC-CXR expert boxes

The public board-certified-radiologist release contains 354 images and 458
boxes for pneumonia and pneumothorax (source SHA-256
`8b80ca98ac6b28ccd00a897c863a57acb9b519025acbe565bff307a8446df668`).
Only one image overlaps the 700-row local MIMIC report cohort, and its study has
no sibling image row. One `visible` image--finding record is therefore
admitted; no missing box is converted to another state. This yields zero
view-exclusive pairs. See the [public annotation repository](https://github.com/leotam/MIMIC-CXR-annotations).

### Chest ImaGenome

No Chest ImaGenome files are present locally. More importantly, the silver
scene graphs are constructed by extracting attributes from study reports and
attaching them to frontal images; they would reproduce the target-side source
whose collision is under test. The 500-patient gold attributes validate and
correct report text relations, while the image annotation is primarily anatomy
bounding boxes. It is not an independent three-state finding decision for each
view in a paired study. It is therefore inadmissible for this gate even if
downloaded. See the [official construction and gold-standard description](https://physionet.org/content/chest-imagenome/1.0.0/).

### Other local material

- The local IU X-ray material contains many multi-image studies, but it also
  supplies one shared report and has no independent per-image finding truth.
  Using it would merely enlarge the duplicated-target substrate.
- VinDr-CXR supplies strong image-local reader labels and boxes, but the local
  release has no paired same-study view sets on which to define sibling-view
  crossover.
- Neither model-generated reports, CheXbert/RadGraph extraction from a shared
  report, nor an image classifier was admitted as truth.

## Machine-readable decision

The compiler emitted:

- `corrected_runs/sisc_truth_gate_v1/study_view_manifest.jsonl`: 700 rows;
- `corrected_runs/sisc_truth_gate_v1/claim_view_truth_candidates.jsonl`: one
  independently boxed `visible` record;
- `corrected_runs/sisc_truth_gate_v1/sisc_feasibility.json`: frozen gate and
  provenance audit.

The final gate failures are:

- paired studies: 44 < 100;
- eligible findings: 0 < 3;
- per-finding independent view-exclusive/unassessable count: 0 < 30;
- `refuted` and `unassessable` cannot be separated because neither is present.

Patient/study-disjoint splitting alone passes. Overall decision is **NO-GO**;
`gpu_authorized=false` and `outcomes_opened=false`.

## Nearest-prior collision

The broad observation that reports summarize multiple views and lack
single-image diagnosis allocation is already occupied.

- **KCLVA** (MIUA 2025, DOI
  [10.1007/978-3-031-98688-8_14](https://doi.org/10.1007/978-3-031-98688-8_14))
  explicitly motivates MIMIC/IU multi-view learning from one report, extracts
  view-specific terms, and introduces view-specific attention/many-to-many
  contrastive learning. Its caption loss still uses the full report and its
  endpoints are ordinary NLG metrics, so it does not supply our required
  independent claim--view truth or supervision-incidence causal test.
- **View-PNDF** ([arXiv:2606.31099](https://arxiv.org/abs/2606.31099)) studies
  disagreement between reports generated independently from different views,
  identifies view-specific neurons, and merges view reports. This further
  occupies generic view-consistency and view-specific generation claims.
- **LLM-RG4** (AAAI 2025) compares single-view, multi-view, and longitudinal
  evidence contexts. It occupies the claim that richer input context improves
  generation.

Consequently SISC could only remain novel through the conjunction of (i)
independent per-image `visible/refuted/unassessable` truth, (ii) exact-parent
children differing only in exploded versus incidence-aware supervision, and
(iii) a matched-content wrong-view crossover endpoint. The present truth gate
fails before this conjunction can be tested. It must not be repackaged as
"multi-view helps."

## Reopening condition

SISC may reopen only after acquiring a genuinely new truth substrate, most
plausibly a blinded physician annotation study over at least 100 MIMIC studies
with paired frontal/lateral or complementary views. Each image must be judged
alone, without the report or sibling view, for at least three frozen findings
using exactly `visible/refuted/unassessable`; a second reader and blinded
adjudication must establish at least 30 qualifying study--claim incidences per
finding. Full MIMIC metadata must then bind DICOM, study, subject, and view
before the split is frozen.

Until that substrate exists, no threshold relaxation, report-derived weak
label, absent-box heuristic, model pseudo-label, multi-view generation run, or
GPU exact-parent training is authorized for SISC.

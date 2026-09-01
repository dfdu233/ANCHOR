# Causal image-use common protocol v1

## Scientific role

This protocol is the backend-neutral behavioral qualification gate that must run
before any PCEM representation capture. It asks whether a frozen model/finding
cell uses the image stably under four matched conditions:

1. `original`;
2. `swap`: a different-patient, same-label image;
3. `target_mask`: the finding-relevant region is masked;
4. `irrelevant_mask`: an equal-area irrelevant region is masked.

The point estimands and released behavior taxonomy follow Lotfinia et al.,
*Vision-language models for chest radiography do not always need the image*
(arXiv:2606.17710v2), using the official repository at commit
`6acd5639f06c7ac89c890f67a7e1eef335726d47`. The local adaptation replaces an
i.i.d. case bootstrap with a patient/episode-cluster bootstrap. It is therefore
`common_protocol`, not a claim of exact paper-native reproduction.

Passing establishes only stable behavioral image use. It does not establish
projection-conditioned binding, layerwise erasure, mitigation efficacy, or a
paper claim.

## Frozen record contract

The executable evaluator is
`anchor/medeval/evaluate_causal_image_use.py`. Each JSONL row is one condition
for one `(model_id, finding, case_id)` cell and must include:

The machine protocol identifier is `anchor-causal-image-use-triad-v1`.

```text
case_id, model_id, finding, cluster_id, view, condition
ground_truth, ground_truth_provenance, decision, parser_version
raw_text_sha256, question_sha256, prompt_sha256
reference_contract_sha256, swap_manifest_sha256
source_image_sha256, condition_image_sha256
source_subject_hash, condition_subject_hash
swap_label_preserved
target_region_defined, irrelevant_region_defined, region_provenance
mask_sha256, mask_area_pixels
```

All four conditions must agree on frozen semantic and provenance fields. The
swap must change both image and patient while preserving the reference label.
The two masks must differ, retain the source patient, and cover exactly equal
pixel area. One run binds exactly one reference contract, one swap manifest,
and parser `upstream-normalized-explicit-binary-v1`.

Raw text, image bytes, subjects, prompts, masks, references, and manifests are
represented by lowercase SHA-256 values. Patient identifiers are never written
to the analysis artifact.

## Truth and region provenance

The evaluator preserves weaker sources for diagnostics but admits only:

- truth: `independent_clinical` or `expert_image_annotation`;
- regions: `expert_box` or `clinician_validated_segmentation`.

`report_derived`, `model_derived`, and `automatic_unvalidated` inputs can still
produce a descriptive behavior category, but their provenance gate is false
and they cannot qualify a PCEM cell. This prevents circular model-derived truth
or unvalidated saliency regions from certifying image use.

## Estimands and behavior taxonomy

For originally correct and parseable cases, the evaluator reports original
accuracy, unrelated-image answer rate, causal grounding rate, irrelevant-mask
stability, and a diagnostic grounding-specificity premium. It reproduces the
released categories:

- `unstable`: irrelevant-mask stability below 0.70;
- `ignores_image`: CGR = 0, UAR = 1, and stability = 1;
- `uses_image`: CGR > 0 with bootstrap lower bound > 0 and stability at least
  0.90;
- otherwise `other` or `not_evaluable`.

Scientific CLI runs use 10,000 patient/episode-cluster bootstrap replicates.
Every target metric requires at least 30 eligible cases, and every condition
requires at least 95% parse rate. AP and PA are reported separately as
diagnostics; subgroup cells do not silently inherit the pooled power gate.

## Fail-closed admission

A model/finding cell enters the PCEM image-use set only if all of the following
hold:

```text
official category == uses_image
eligible cases >= 30 for CGR, UAR, and stability
parse rate >= 0.95 in each of four conditions
trusted ground-truth provenance only
trusted region provenance only
```

The cross-model gate additionally requires every preregistered target cell to
pass and at least two model families. Even after it passes, the output fixes:

```text
representation_capture_authorized=false
image_download_authorized=false
gpu_authorized=false
paper_claim_authorized=false
```

Independent echo construct validity and the geometry-by-view behavioral gate
must authorize subsequent PCEM stages. The evaluator cannot authorize those
steps itself.

## Invocation

```bash
PYTHONPATH=. .venv-full/bin/python -m anchor.medeval.evaluate_causal_image_use \
  --input /path/to/frozen_conditions.jsonl \
  --output /path/to/causal_image_use.json \
  --target-model huatuo \
  --target-model hulu \
  --target-finding cardiomegaly
```

No real-model run is currently authorized: the ECHO substrate remains
access-blocked, and this protocol creates no images, labels, masks, or model
outputs.

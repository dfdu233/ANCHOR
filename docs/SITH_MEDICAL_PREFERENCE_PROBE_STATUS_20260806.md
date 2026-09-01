# SITH medical preference probe status — 2026-08-06

## Decision

Raw SITH-style VO singular directions are rejected as the current main route
for auditing "what images a medical VLM prefers" on the available local setup.

Scope of this decision:

- model: HuatuoGPT-Vision-7B local CLIP ViT-L/14-336 visual tower;
- data: 500-image VinBigData/VinDr-like balanced DICOM sample from
  `/workspace/vinbigdata`;
- layers: final four ViT layers, 20--23;
- directions: both right/input and left/output singular vectors, rank 16 per
  head;
- controls: sign-invariant text matching, same-count random hidden directions,
  three activation response statistics, and same-manifest pooled-feature
  positive control.

This does not prove that every SITH-inspired idea is impossible. It does reject
the specific claim that raw VO singular directions, named by CLIP text prompts
and projected onto images, provide a reliable medical preference/shortcut
coordinate system in this setting.

## Evidence chain

### 1. Weight-space concept naming fails the random-direction control

Artifact:
`corrected_runs/sith_medical_preference_probe_v1/huatuo_clip_vit_l20_23_rank16_right_v2.json`

- protocol:
  `sith-medical-preference-weight-probe-v2-sign-random-control`
- fingerprint:
  `6f6f095912dc86bb0d9cdc50a9c969b15d8d4bec06de26ab3f2e69065a0fb0eb`
- observed shortcut top-1 rate: `0.7754`
- random shortcut top-1 rate: `0.7764`
- observed-minus-random shortcut enrichment: `-0.0010`

The striking "device/artifact" naming pattern is reproduced by random hidden
directions passed through the same projection and text-matching path. Therefore
it is not evidence of a learned medical shortcut preference.

Left singular vector naming showed the same qualitative pattern:

- artifact:
  `corrected_runs/sith_medical_preference_probe_v1/huatuo_clip_vit_l20_23_rank16_left_v2.json`
- device/artifact top-1 rate: `0.7266`
- shortcut top-1 rate: `0.7813`

### 2. Real-image activation of right singular directions does not beat random

Artifact:
`corrected_runs/sith_medical_preference_probe_v1/vindr_activation_huatuo_l20_23_rank16_n500_v2.json`

- protocol:
  `sith-vindr-activation-probe-v2-response-and-top-random-controls`
- fingerprint:
  `4e34dbb328b06e4c1f3e1d1e3d5bdfc0df9ffe07a39670e23741a2cf92890daa`
- sample: 100 each from No finding, Cardiomegaly, Pleural effusion,
  Aortic enlargement, Pulmonary fibrosis.

Maximum layer/response mean observed-minus-random absolute AUC advantage:
`+0.0031`.

Across CLS, mean-patch and max-patch response statistics, the average advantage
stays near zero and changes sign. Best observed direction versus best random
direction is also unstable, so isolated high-AUC directions should be treated as
multiple-comparison hits.

### 3. Left singular directions also do not show stable image-label advantage

Artifact:
`corrected_runs/sith_medical_preference_probe_v1/vindr_activation_huatuo_l20_23_rank16_left_n500_v2.json`

Maximum layer/response mean observed-minus-random absolute AUC advantage:
`+0.0035`.

This is again too small and unstable to support a preference/shortcut coordinate
claim.

### 4. The null result is not because pooled CLIP image features lack disease signal

Artifact:
`corrected_runs/sith_medical_preference_probe_v1/vindr_zeroshot_control_huatuo_n500.json`

- protocol: `sith-vindr-zeroshot-control-v1`
- fingerprint:
  `909fff68eb4b9d59e2939aeb91d12c0c6996ad9164e997c3910ff2e8ba960c81`

Same-manifest positive controls:

| finding | zero-shot prompt AUC | pooled-feature ridge AUC |
|---|---:|---:|
| Cardiomegaly | 0.566 | 0.719 |
| Pleural effusion | 0.539 | 0.681 |
| Aortic enlargement | 0.546 | 0.774 |
| Pulmonary fibrosis | 0.564 | 0.612 |

The visual tower contains pathology information in pooled features, but raw
SITH VO directions do not recover it beyond random-direction controls.

## Interpretation

The likely failure mode is conceptual, not just engineering:

1. CLIP text-nearest naming is dominated by projection/text-space priors; random
   hidden directions already map heavily to device/artifact phrases.
2. Per-head VO singular vectors are useful weight-basis objects, but in this
   medical CLIP tower they are not aligned enough with image-level pathology or
   shortcut variation to serve as a calibrated concept coordinate system.
3. The original DG/style/source-center risk remains: without true metadata and
   causal intervention, "style preference" claims are too easy to hallucinate
   from pretty clusters.

## Consequence for the project

Do not make raw SITH-VO the main paper mechanism.

Recommended next route:

1. Keep SITH as a negative baseline / cautionary mechanism audit.
2. Replace "weight singular vectors as concepts" with an activation-grounded
   preference audit:
   - pooled/patch hidden probes;
   - true metadata groups where available: view position, portable/AP, hospital
     or source, DICOM photometric/render tags;
   - error/confidence correlation;
   - intervention only after metadata-grounded correlation survives random,
     label and length/confidence controls.
3. If a second local medical CLIP becomes available, rerun the same scripts
   unchanged before generalizing beyond Huatuo.


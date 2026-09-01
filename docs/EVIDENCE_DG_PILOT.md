# Evidence-DG Pilot

## Status

This is a hypothesis audit, not a mitigation result. It asks whether public
dataset source shift remains visible inside HuatuoGPT-Vision and whether that
shift predicts CE or report-generation errors.

Runner: `anchor/corrected_sgta/run_huatuo_evidence_dg_probe.py`

The first report smoke at
`corrected_runs/huatuo_evidence_dg/report_n1_smoke/` is invalidated because
its teacher-forcing path omitted Huatuo's image placeholder. The corrected
runner asserts exactly one image token. No invalid evidence statistic is used
below.

## Protocol

### Report/OE

- Model: `/home/dbw/models/HuatuoGPT-Vision-7B`
- Sources: MIMIC-CXR, IU-Xray, and `chexpert_subset_unverified`
- Samples: 16 unique images per source, 48 total
- Prompt: the existing MMed-RAG zero-shot radiology-report prompt
- Decoding: native greedy, 256 new-token limit
- Reference use: teacher-forced diagnostics and evaluation only
- Output: complete free-form reports; no binary parser or label logits

Artifacts:

- `corrected_runs/huatuo_evidence_dg/report_n16_controls_v1/`
- Text evaluation for the first 8/source checkpoint:
  `corrected_runs/huatuo_evidence_dg/report_n8_v1/text_eval/`

### CE

- Sources: RULE/MIMIC and RULE/IU-Xray
- Samples: 32 unique images per source, 64 total
- Sampling: 16 Yes and 16 No per source; one question per image
- Prompt: unchanged RULE question with only the dataset `<image>` marker
  removed because Huatuo inserts its own marker
- Decoding: native greedy complete sentences, 48 new-token limit
- Primary quality label: RULE normalized generated-sentence correctness
- Decision-first accuracy and parse rate: diagnostics

Artifact: `corrected_runs/huatuo_evidence_dg/ce_n16_balanced_v1/`

The directory name records the first checkpoint; the identical run was resumed
to 32 samples per source. Its config and summary record the final `n=64`.

## Features And Controls

Each sample stores:

- pixel intensity and radial-frequency descriptors;
- pooled vision-tower and post-projector features;
- pooled final-decoder representation;
- a 19-dimensional token-evidence trajectory containing NLL, entropy,
  top-1 margin, target top-1 rate, and zero-visual support;
- teacher-forced reference NLL under intermediate-layer mixing;
- norm-matched layer-mixing and final-logit temperature controls.

Source probes use four-fold stratified logistic regression. Quality probes
remove each source's mean quality inside the training fold before fitting, so a
probe cannot succeed merely by recognizing that one dataset is easier.

## Results

### Source Identification

| Task | Pixel intensity | Radial | Vision pre | Vision post | Decoder final | Evidence trajectory |
|---|---:|---:|---:|---:|---:|---:|
| Report, chance 33.3% | 81.25% | 77.08% | 68.75% | 70.83% | 56.25% | 41.67% |
| CE, chance 50% | 95.31% | 92.19% | 85.94% | 87.50% | 71.88% | 59.38% |

Public dataset source is strongly encoded in pixels and visual features. The
signal weakens toward the compact generation-evidence trajectory.

### Source-Controlled Quality Prediction

Report ROUGE-L:

- pixel intensity: Spearman 0.215, `p=0.142`;
- visual pre/post: -0.201/-0.133;
- decoder final: -0.315, `p=0.029`;
- evidence trajectory: 0.275, `p=0.058`.

The lone negative decoder correlation is not an actionable reliability signal,
and the evidence trajectory does not pass the pilot threshold.

CE RULE-normalized correctness:

- pixel intensity: 0.065;
- radial: -0.288, `p=0.021`;
- visual pre/post: 0.008/0.017;
- decoder final: 0.103;
- evidence trajectory: 0.213, `p=0.091`.

The n16 visual-quality signal disappeared at n32. The surviving radial
correlation is negative, isolated, and does not transfer to OE.

### NLL Control

On reports, leave-one-source-out `layer 7, alpha 0.1` lowered reference NLL by
0.027-0.029, but temperature 1.2 lowered it by 0.037-0.075. Norm matching
reduced the layer effect to 0.018-0.022.

On CE, layer mixing also lowered reference NLL, but temperature scaling matched
or exceeded it on the IU-Xray held-out fold and was comparable on MIMIC.
Temperature scaling cannot alter greedy argmax tokens. The NLL gain is therefore
primarily a calibration effect, not evidence that source-robust layer transport
improves generation.

## Bad Cases

- Reports average roughly 159-180 words while references are much shorter.
  They frequently add differential diagnoses, disease explanations, and
  unsupported devices.
- A CheXpert case with bibasilar atelectasis and effusions gained unsupported
  pneumonia, edema, catheters, and ventilation details.
- An IU-Xray case with a left lower-lobe infiltrate gained a large right
  effusion and multiple metastatic/granulomatous nodules.
- Hallucinations occur at both near-zero and high zero-visual support. Direct
  visual-support/ROUGE-L correlation is 0.034.
- In CE, correct and wrong outputs do not differ significantly in visual
  support, generated NLL, reference NLL, temperature gain, or layer gain.
- Decision-first parsing marks some semantically explicit natural sentences as
  unparsed. The final quality probe therefore uses RULE's normalized
  generated-sentence primary metric and reports decision-first separately.

## Decision

Supported:

1. Public CXR datasets have a large measurable source shift.
2. Huatuo's visual representation preserves source identity.
3. Source identity attenuates but remains detectable in the decoder.

Not supported:

1. Pixel or visual source distance is a stable predictor of hallucination.
2. The compact evidence trajectory predicts CE and OE quality.
3. Intermediate-layer NLL gains establish an evidence-DG mechanism.
4. The current evidence is strong enough to launch 128/full mitigation runs.

The experiment stops at 32 CE samples per source and 16 reports per source.
Continuing the same probe would violate the registered breadth-first rule.
DG remains a useful evaluation setting, but public-dataset style identity is
not yet a demonstrated causal mitigation target.

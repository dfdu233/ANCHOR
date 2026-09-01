# Huatuo Source-Shift Round-Trip Pilot

## Question

This pilot tests the missing causal bridge in image-only DG:

1. Does a source-derived public CXR style shift cause Huatuo errors?
2. Does projection to Huatuo's PubMedVision-CXR envelope repair those errors?
3. Is matched-source repair better than an unrelated public CXR envelope?

The VLM, prompt, and decoding remain fixed. Every condition emits one complete
natural-language answer. No Yes/No logits, voting, candidate selection, or
reference-aware selection is used.

## Bank Correction

The historical `ct__chest.npy` is a CT/chest 2-D amplitude center and is not
used. The matched bank is the existing high-precision PubMedVision-CXR radial
bank built from 5,846 single-image caption-confirmed chest radiographs. Its
base `q05/q95` envelope is used for exact projection.

MIMIC, IU-Xray, and the unverified CheXpert proxy each contribute 500
model-visible images to separate control banks. They are public-dataset
proxies, not claimed to be pure hospital acquisition domains.

## Intervention

For a test image descriptor `r`, the CheXpert-proxy domain residual is:

```
delta = median(CheXpert proxy) - median(PubMedVision CXR)
r_shift = r + 0.25 * delta
```

The matched repair clips `r_shift` to the PubMedVision base envelope. The
mismatched repair clips it to the IU-Xray envelope. Reconstruction applies
isotropic radial gains and preserves the original phase.

Direct interpolation to the IU median was rejected during preflight because it
reduced PubMedVision exceedance and made matched repair 94-100% identity. The
domain-residual shift is retained because it increases source exceedance on
average while remaining source-data-derived.

## Image Audit

On 16 RULE/MIMIC images:

| Condition | PubMed exceedance | PSNR | Edge correlation |
|---|---:|---:|---:|
| Native | 0.0280 | - | 1.0000 |
| Shift 0.25 | 0.0590 | 26.96 | 0.9947 |
| Matched repair | 0.0027 | 27.64 | 0.9939 |
| Mismatched repair | 0.0002 | 27.33 | 0.9852 |

The stronger 0.5 shift reaches only 21.24 dB mean PSNR and is not used for
generation. Mismatched repair also reduces PubMed exceedance, so geometric
return alone cannot establish model-source specificity.

## CE Result

RULE/MIMIC uses eight stable-hash samples:

| Condition | Decision-first accuracy | RULE normalized |
|---|---:|---:|
| Native | 5/8 | 4/8 |
| Shift | 5/8 | 4/8 |
| Matched repair | 5/8 | 4/8 |
| Mismatched repair | 6/8 | 4/8 |

The shift causes no real decision flip, so there is no CE degradation for
matched repair to recover. The mismatched apparent rescue is parser-only: a
semantically negative native sentence changes to an explicit leading `No`.

Matched repair also adds an unsupported right pleural effusion to item 1188.

## OE Result

MedHEval visual OE/MIMIC uses four report-generation samples and 256-token
greedy decoding:

| Condition | BLEU | ROUGE-L | METEOR | Token F1 |
|---|---:|---:|---:|---:|
| Native | 0.0126 | 0.1394 | 0.2277 | 0.2150 |
| Shift | 0.0143 | 0.1601 | 0.2447 | 0.2366 |
| Matched repair | 0.0118 | 0.1665 | 0.2238 | 0.2268 |
| Mismatched repair | 0.0116 | 0.1566 | 0.2122 | 0.2210 |

The lexical signal is not clinically safe. On item 415, the reference reports
bullous basal disease, a Dobbhoff catheter in the stomach, no pleural
effusions, and upper-normal heart size. Matched repair invents a right
pneumothorax, partial lung collapse, leftward tracheal deviation, multiple rib
fractures, and subcutaneous emphysema.

Pinned RadGraph, RaTEScore, and CheXbert checkpoints remain unavailable, so the
clinical-metric block is explicitly recorded rather than replaced by
unregistered weights.

## Decision

The pilot stops before 16/32:

- source-away shift: passes only on average;
- CE causal degradation: fails;
- matched CE recovery: fails;
- OE catastrophic-harm gate: fails;
- matched-bank specificity: fails the geometric control.

This closes the most direct paired causal test without supporting source-bank
image calibration as a Huatuo hallucination mitigation method.

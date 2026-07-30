# Alignment-contraction mechanism audit

This directory contains compact, publishable diagnostics from a controlled
Qwen2.5-VL-7B experiment. Large raw traces and merger checkpoints remain
outside Git under `/root/autodl-fs/data/dbw/anchor_alignment_contraction_v1/`.

## Controlled training

- Source: 2,048 strict-CXR PubMedVision instruction records.
- Matched and image-permuted branches have identical image/text marginals,
  zero fixed pairs, and zero same-PMC/figure-group pairs.
- Both train only the 44,574,464-parameter visual merger for 250 steps,
  seed 42, effective batch 8, and learning rate `5e-6`.
- Matched fingerprint:
  `1d57e127c42513684a33256014378918dc5d983c0e85458cfecd1f0c59147b80`.
- Permuted fingerprint:
  `3e6e104b6d8057bd1b325337cc09cd9ba61f3401d9d53f94655c79b25f0d8064`.

## Result

On 128 strict-CXR source-held-out records, mean complete-answer NLL was
1.21636 (base), 1.21378 (matched), and 1.21312 (permuted).
Matched-minus-permuted was +0.00066 with source-group cluster-bootstrap 95%
CI `[-0.00020, 0.00154]`.

On 40 selected frontal MIMIC development images from 38 patients, normalized
style susceptibility was 0.3175 (matched) versus 0.3211 (permuted), difference
-0.0036 with patient-cluster bootstrap 95% CI `[-0.0301, 0.0113]`. Reusable
style variance was 2.129% versus 2.221%, difference -0.092 percentage points,
CI `[-0.394, 0.150]`.

The prespecified gate therefore failed: correct pairing did not cause
detectable style contraction at the merger under this budget. The result does
not rule out conditional style priors or explicit same-content orbit
training.

See `docs/ALIGNMENT_STYLE_CONTRACTION_AUDIT.md` for the theorem, limitations,
and claim ceiling.

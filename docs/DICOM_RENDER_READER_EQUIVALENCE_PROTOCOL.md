# The Renderer Is a Hidden Reader

## Frozen question

Can clinically admissible renderings of one unchanged DICOM move a medical
VLM's signed claim evidence by an amount comparable with one independent
radiologist vote?

This is an equivalence-class test, not a source-domain or generic style claim.
It does not infer a training-domain center from DICOM metadata.  A positive
result means that the model is not invariant on a clinically audited image
orbit; it does not by itself establish hallucination mitigation.

## Frozen Huatuo pilot

- Dataset: the image-disjoint VinDr pilot split from the fixed R8/R9/R10 reader
  panel.
- Findings: aortic enlargement, cardiomegaly, pleural effusion, and pulmonary
  fibrosis.
- Sample: 10 claims per finding and reader-vote bin, for 160 claims total.
- Coordinate: FP32 next-token `Yes - No` polarity.  Commitment is
  `max(Yes, No) - Maybe` and is secondary.
- Unit: a finding-specific robust slope of polarity over 0/3, 1/3, 2/3, and
  3/3 reader support, with image-cluster bootstrap.

The running artifact is fingerprinted by
`aa4030ad49d00cdd4bcc8839202c3de457ad5703b3413beb22a09a0cbadcaa66`.
The runner SHA-256 is
`743625bb31f8aa7b09aa13b599715cc72b55a69686ae228da1eb2e40d48332f6`;
the analyzer SHA-256 is
`5a60d33b8788e399d0a90d92d7694f90794e0ed0f38104a1a0b632c5ea7ddee8`.

## Clinical render contract

Every view applies the DICOM rescale transform and correct MONOCHROME1/2
polarity first.  The primary candidates are native LINEAR windowing,
center shifts of plus or minus 0.05 window widths, width multipliers 0.8 and
1.25, and conservative blank-border zoom.  A transform is globally eligible
only when it is available and passes the fixed audit in at least 95% of pilot
claims.  The audit uses a label-independent central-thorax saturation region;
reader boxes are used only to ensure that cropping removes no annotated
lesion.

Non-native SIGMOID, polarity inversion, and 32-by-32 content loss are secondary
controls and can never drive the discovery gate.  An exact lossless duplicate
defines the numerical noise floor.  Token flips and the maximum-minus-minimum
render orbit are descriptive only.

The pre-model CPU audit found 0/160 failures.  All source VOI declarations are
LINEAR.  Native LINEAR, center-minus-0.05W, and width-times-1.25 pass 160/160;
center-plus-0.05W passes 159/160.  Width-times-0.8 and blank-border zoom pass
only 118/160 and are therefore ineligible under the frozen 95% rule.

## Fail-closed mechanism gate

Within each finding, a deterministic image-disjoint half A selects one primary
transform by absolute median paired effect.  Half B must independently satisfy
all of the following:

1. The baseline reader-step slope is positive and its cluster-bootstrap 95%
   confidence interval excludes zero.
2. The selected transform has the same signed effect in half B, its confidence
   interval excludes zero, and at least 65% of paired cases agree in sign.
3. The absolute held-out effect is at least 0.5 reader steps.  A joint
   image-cluster bootstrap that re-estimates the reader slope must have a
   direction-aware lower magnitude bound greater than 0.25 reader steps.
4. The paired effect exceeds the exact-duplicate numerical floor.
5. The signed effect remains nonzero in the high-baseline-margin half of held-
   out cases.

At least three of the four frozen findings must pass.  A positive Huatuo pilot
only authorizes replication in Hulu and a physician-blinded audit of at least
40 render orbits.  The mechanism is promoted only if at least two models pass.

## Frozen result

Huatuo failed `0/4` findings.  The reader-step slope had a positive confidence
interval for aortic enlargement, cardiomegaly, and pleural effusion, but not
pulmonary fibrosis.  More importantly, every transform selected on half A
failed the half-B magnitude/direction or high-margin confirmation.  The exact
duplicate floor passed, so this is not a numerical-readout failure.

Several descriptive max-minus-min orbit tests would have passed.  They were
predeclared as non-gating because selecting extrema over many transforms
inflates sensitivity without requiring a reproducible renderer direction.
The result therefore rejects a stable “renderer is a hidden reader” mechanism
in Huatuo and does not authorize a Hulu raw-render replication.

## Role after the collision audit

Even if this gate passes, raw render stability or render-contrastive decoding
is not promoted as the paper contribution: SPCD, LENS, VGS-Decoding, UniVRSE,
and related perturbation decoders already occupy that method neighborhood.
This pilot becomes a calibrated substrate and direct baseline for the narrower
Clinical-Equivalence Composition Defect test, which crosses admitted render
operations with speech-act-preserving paraphrases and asks whether their
two-way interaction adds held-out clinical-error information beyond clean
margin and both marginal sensitivities.

Render symmetrization and a clinical-orbit certificate remain baselines.  Any
later method must reduce reader-grounded overcommitment by at least 20% and
reader Brier by at least 5%, keep clear-case performance within one percentage
point, and keep OE claim count fixed.  If gains come from shorter answers,
refusal, universal hedging, claim deletion, or ordinary full-orbit averaging,
the method fails.

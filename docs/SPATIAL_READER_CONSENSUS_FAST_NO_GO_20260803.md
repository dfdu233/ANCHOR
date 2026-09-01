# Spatial Reader Consensus: fast NO-GO

Date: 2026-08-03 UTC

## Question

Does a medical VLM preserve *where* independent readers locate the same positive
finding, but erase this spatial-consensus signal when converting image evidence
into committed language?

This is intentionally narrower than reader-vote clarity: all analyzed claims are
3/3 positive, while the candidate target is agreement among the three readers'
independent boxes.

## Independent substrate audit

`/workspace/vinbigdata/train.csv` contains 15,000 images, exactly three readers
per image, and 5,266 image/finding groups for which all three readers supplied a
positive box.  For each reader pair, boxes were matched one-to-one by maximum
intersection and scored with an area-normalized matched Dice coefficient.  The
median pair score has substantial variation (mean 0.708, SD 0.213, 5th--95th
percentile 0.236--0.911).  The variation is not merely a tiny-data artifact:
pulmonary fibrosis has 633 unanimous-positive groups, pleural thickening 365,
pleural effusion 439, and lung opacity 167.

Thus the target is measurable in VinDr, although it can conflate lesion
multiplicity with ambiguity and is not itself a clinical uncertainty label.

## Outcome-blind minimum screen

The existing frozen Huatuo development capture contains 160 unanimous-positive
claims (20 for each of eight findings; 152 unique images) at decoder layers
7/14/21/28.  Spatial consensus was binarized by the within-finding median, and
evaluated with five-fold stratified image-group cross-validation.  The fixed
probe was StandardScaler -> 32-dimensional PCA -> class-balanced logistic
regression (`C=0.1`).

Cross-validated AUROC:

| representation | L7 | L14 | L21 | L28 |
|---|---:|---:|---:|---:|
| claim state | 0.461 | 0.432 | 0.413 | 0.412 |
| visual-token mean | 0.526 | 0.455 | 0.530 | 0.496 |
| visual-token SD | 0.531 | 0.432 | 0.487 | 0.377 |

Finding-centered Spearman associations between median spatial consensus and
the supported-vs-undetermined logit margin were -0.125, -0.147, -0.028, and
0.004 at L7/L14/L21/L28 (all uncorrected p > 0.06).  There is neither reliable
decodability nor a clean early-to-late erasure pattern.

## Collision boundary

CheXthought already supplies multi-reader visual attention, predicts human-human
and human-AI disagreement, and reports that visual-attention hints recover
missed findings and reduce hallucinations (arXiv:2604.26288).  Generic pathology
localization and phrase-grounded report fact checking are also occupied.  A
paper based only on VinDr box dispersion would therefore be an underpowered
application-level variant unless a distinct causal mediator were first shown.

## Decision

**Strict NO-GO for the current portfolio.**  Do not launch new GPU captures or
present spatial box consensus as a replacement for the failed reader-clarity
mechanism.  This result does not claim that spatial agreement is clinically
irrelevant; it says the current model/substrate offers no credible mechanism or
novelty signal strong enough to justify further paper-track spending.

